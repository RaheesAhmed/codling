"""
CODLING Trainer - Complete Training Pipeline
=============================================

A comprehensive training pipeline for CODLING SSM Language Model with:
- Mixed precision training (BF16/FP16)
- Gradient checkpointing for memory efficiency
- Learning rate scheduler (cosine with warmup)
- Optimizer setup (AdamW with weight decay)
- Checkpoint saving/loading
- Training metrics tracking
- DDP (DistributedDataParallel) support
- Colab T4 GPU detection and optimization

Usage:
    from trainer import Trainer, TrainerConfig
    
    config = TrainerConfig(
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        output_dir="./checkpoints",
        learning_rate=3e-4,
        max_steps=10000,
        warmup_steps=500,
        gradient_accumulation_steps=4,
    )
    
    trainer = Trainer(config)
    trainer.train()

Author: CODLING Team
"""

import os
import math
import time
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Iterable
from contextlib import nullcontext
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler

# For gradient checkpointing
from torch.utils.checkpoint import checkpoint_sequential

# For debugging and profiling
from torch.profiler import profile, ProfilerActivity, record_function

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainerConfig:
    """
    Configuration for the CODLING Trainer.
    
    Attributes:
        model: The model to train (nn.Module)
        train_dataloader: Training DataLoader
        eval_dataloader: Evaluation DataLoader (optional)
        output_dir: Directory to save checkpoints
        max_steps: Maximum training steps
        eval_steps: Evaluate every N steps
        save_steps: Save checkpoint every N steps
        logging_steps: Log metrics every N steps
        learning_rate: Initial learning rate
        min_learning_rate: Minimum learning rate for cosine decay
        weight_decay: Weight decay for AdamW
        warmup_steps: Number of warmup steps
        gradient_accumulation_steps: Number of gradient accumulation steps
        max_grad_norm: Maximum gradient norm for clipping
        batch_size: Batch size per device
        seq_length: Sequence length
        use_gradient_checkpointing: Enable gradient checkpointing
        use_compile: Enable torch.compile for speedup
        precision: Training precision ("bf16", "fp16", "fp32")
        save_total_limit: Maximum number of checkpoints to keep
        resume_from: Path to checkpoint to resume from
        local_rank: Local rank for distributed training
        world_size: World size for distributed training
        log_dir: Directory for logs
        seed: Random seed
    """
    # Model and data
    model: nn.Module = None
    train_dataloader: DataLoader = None
    eval_dataloader: Optional[DataLoader] = None
    
    # Output
    output_dir: str = "./checkpoints"
    
    # Training steps
    max_steps: int = 10000
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 10
    
    # Optimizer
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 0.1
    warmup_steps: int = 500
    warmup_ratio: float = 0.0  # Alternative to warmup_steps (ratio of max_steps)
    
    # Gradient
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Batch
    batch_size: int = 8
    seq_length: int = 2048
    
    # Memory optimization
    use_gradient_checkpointing: bool = True
    use_compile: bool = False
    
    # Precision
    precision: str = "bf16"  # "bf16", "fp16", "fp32"
    
    # Checkpointing
    save_total_limit: int = 3
    resume_from: Optional[str] = None
    
    # Distributed
    local_rank: int = -1
    world_size: int = 1
    
    # Logging
    log_dir: Optional[str] = None
    seed: int = 42
    
    # Colab-specific
    colab_detect: bool = True  # Auto-detect Colab environment
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Handle warmup ratio
        if self.warmup_ratio > 0 and self.warmup_steps == 0:
            self.warmup_steps = int(self.max_steps * self.warmup_ratio)
        
        # Validate precision
        if self.precision not in ["bf16", "fp16", "fp32"]:
            raise ValueError(f"Invalid precision: {self.precision}")


# ============================================================================
# Metrics Tracker
# ============================================================================

class MetricsTracker:
    """
    Tracks and logs training metrics.
    
    Attributes:
        history: List of metric dictionaries
        current_epoch: Current epoch number
        current_step: Current global step
    """
    
    def __init__(self, log_dir: Optional[str] = None):
        self.history: List[Dict[str, float]] = []
        self.current_epoch = 0
        self.current_step = 0
        self.log_dir = log_dir
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    def update(self, metrics: Dict[str, float], step: int):
        """Update metrics for current step."""
        self.current_step = step
        metrics["step"] = step
        metrics["epoch"] = self.current_epoch
        self.history.append(metrics)
        
        # Save to file periodically
        if self.log_dir and step % 100 == 0:
            self.save()
    
    def save(self):
        """Save metrics history to file."""
        if not self.log_dir:
            return
        
        filepath = os.path.join(self.log_dir, "metrics.json")
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def get_average(self, metric_name: str, last_n: int = 100) -> float:
        """Get average of a metric over last N steps."""
        values = [m.get(metric_name, 0) for m in self.history[-last_n:]]
        return sum(values) / len(values) if values else 0.0


# ============================================================================
# Learning Rate Scheduler
# ============================================================================

class CosineWarmupScheduler:
    """
    Cosine learning rate scheduler with warmup.
    
    Learning rate schedule:
    - Warmup: Linear increase from 0 to learning_rate over warmup_steps
    - Cosine decay: Decrease to min_learning_rate over remaining steps
    
    Formula:
        if step < warmup_steps:
            lr = learning_rate * (step / warmup_steps)
        else:
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            lr = min_learning_rate + (learning_rate - min_learning_rate) * \
                 0.5 * (1 + cos(π * progress))
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        learning_rate: float,
        min_learning_rate: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.current_step = 0
        
        # Save initial learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self):
        """Update learning rate."""
        self.current_step += 1
        lr = self.get_lr()
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = lr * (base_lr / self.learning_rate) if self.learning_rate > 0 else base_lr
    
    def get_lr(self) -> float:
        """Calculate learning rate for current step."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.learning_rate * (self.current_step / self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / \
                       (self.max_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            return self.min_learning_rate + \
                   (self.learning_rate - self.min_learning_rate) * \
                   0.5 * (1 + math.cos(math.pi * progress))
    
    def state_dict(self):
        """Return scheduler state."""
        return {
            'current_step': self.current_step,
            'warmup_steps': self.warmup_steps,
            'max_steps': self.max_steps,
            'learning_rate': self.learning_rate,
            'min_learning_rate': self.min_learning_rate,
        }
    
    def load_state_dict(self, state_dict: Dict):
        """Load scheduler state."""
        self.current_step = state_dict['current_step']


# ============================================================================
# Utility Functions
# ============================================================================

def setup_distributed():
    """
    Initialize distributed training.
    
    Returns:
        tuple: (local_rank, world_size)
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return local_rank, world_size
    return -1, 1


def cleanup_distributed():
    """Clean up distributed training."""
    destroy_process_group()


def detect_colab_gpu() -> Dict[str, Any]:
    """
    Detect GPU capabilities, especially for Colab T4.
    
    Returns:
        dict: GPU information including:
            - device_name: GPU name
            - memory_total: Total memory in GB
            - compute_capability: CUDA compute capability
            - is_colab: Whether running on Google Colab
            - recommended_precision: Recommended precision for training
    """
    result = {
        'device_name': 'Unknown',
        'memory_total': 0,
        'compute_capability': (0, 0),
        'is_colab': False,
        'recommended_precision': 'fp32',
    }
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        return result
    
    # Get GPU info
    result['device_name'] = torch.cuda.get_device_name(0)
    result['memory_total'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    result['compute_capability'] = torch.cuda.get_device_capability(0)
    
    # Check for Colab
    try:
        import google.colab
        result['is_colab'] = True
    except ImportError:
        pass
    
    # Recommend precision based on GPU
    compute_cap = result['compute_capability']
    if compute_cap >= (8, 0):  # A100, H100
        result['recommended_precision'] = 'bf16'
    elif compute_cap >= (7, 0):  # V100, T4, RTX Ampere
        result['recommended_precision'] = 'bf16'
    elif 'T4' in result['device_name']:
        result['recommended_precision'] = 'bf16'  # T4 supports BF16
    else:
        result['recommended_precision'] = 'fp16'
    
    return result


def get_autocast_context(precision: str, device_type: str = 'cuda'):
    """
    Get autocast context based on precision.
    
    Args:
        precision: Precision string ("bf16", "fp16", "fp32")
        device_type: Device type for autocast
    
    Returns:
        autocast context manager
    """
    if precision == "bf16":
        return autocast(dtype=torch.bfloat16, device_type=device_type)
    elif precision == "fp16":
        return autocast(dtype=torch.float16, device_type=device_type)
    else:
        return nullcontext()


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters
    
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


# ============================================================================
# Main Trainer Class
# ============================================================================

class Trainer:
    """
    CODLING Trainer - Complete training pipeline for CODLING language models.
    
    This trainer provides:
    - Mixed precision training (BF16/FP16)
    - Gradient checkpointing for memory efficiency
    - Learning rate scheduling (cosine with warmup)
    - Checkpoint saving/loading
    - Distributed training support
    - Comprehensive metrics tracking
    
    Example:
        >>> config = TrainerConfig(
        ...     model=model,
        ...     train_dataloader=train_loader,
        ...     output_dir="./checkpoints",
        ...     learning_rate=3e-4,
        ...     max_steps=10000,
        ... )
        >>> trainer = Trainer(config)
        >>> trainer.train()
    
    Attributes:
        config: Trainer configuration
        model: The model being trained
        optimizer: The optimizer
        scheduler: Learning rate scheduler
        scaler: Gradient scaler for mixed precision
        metrics: Metrics tracker
    """
    
    def __init__(self, config: TrainerConfig):
        """
        Initialize the trainer.
        
        Args:
            config: Trainer configuration
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize distributed training
        self.is_distributed = config.world_size > 1
        if self.is_distributed:
            self.local_rank, self.world_size = setup_distributed()
        else:
            self.local_rank = config.local_rank
            self.world_size = config.world_size
        
        # Detect GPU capabilities
        if config.colab_detect:
            self.gpu_info = detect_colab_gpu()
            logger.info(f"GPU Detected: {self.gpu_info['device_name']}")
            logger.info(f"Memory: {self.gpu_info['memory_total']:.2f} GB")
            logger.info(f"Recommended Precision: {self.gpu_info['recommended_precision']}")
            
            # Auto-configure precision if not specified
            if config.precision == "bf16" and self.gpu_info['recommended_precision'] == "bf16":
                logger.info("Using BF16 precision")
        
        # Set random seed
        self._set_seed(config.seed)
        
        # Initialize model
        self.model = config.model
        self.model.to(self.device)
        
        # Apply gradient checkpointing
        if config.use_gradient_checkpointing:
            self._enable_gradient_checkpointing()
        
        # Wrap with DDP if distributed
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = CosineWarmupScheduler(
            optimizer=self.optimizer,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            learning_rate=config.learning_rate,
            min_learning_rate=config.min_learning_rate,
        )
        
        # Initialize mixed precision scaler
        self.scaler = GradScaler() if config.precision in ["bf16", "fp16"] else None
        
        # Initialize metrics tracker
        self.metrics = MetricsTracker(config.log_dir)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Compile model if requested
        if config.use_compile:
            logger.info("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)
        
        # Resume from checkpoint if specified
        if config.resume_from:
            self._load_checkpoint(config.resume_from)
        
        # Log model info
        total_params = count_parameters(self.model)
        trainable_params = count_parameters(self.model, trainable_only=True)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(self.model, 'module'):
            # DDP wrapped model
            model = self.model.module
        else:
            model = self.model
        
        # Enable gradient checkpointing for all layers
        if hasattr(model, 'enable_gradient_checkpointing'):
            model.enable_gradient_checkpointing()
        else:
            # Manual implementation
            for module in model.modules():
                if hasattr(module, 'gradient_checkpointing'):
                    module.gradient_checkpointing_enable()
        
        logger.info("Gradient checkpointing enabled")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """
        Create AdamW optimizer with weight decay.
        
        Weight decay is applied to:
        - All parameters except bias and LayerNorm
        - LayerNorm parameters are excluded from weight decay
        
        Returns:
            Configured AdamW optimizer
        """
        # Separate parameters with and without weight decay
        no_decay = ['bias', 'LayerNorm', 'layer_norm', 'norm']
        
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': self.config.weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0,
            },
        ]
        
        return optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
    
    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor, 
                      pad_token_id: int = -100) -> torch.Tensor:
        """
        Compute cross-entropy loss with padding token handling.
        
        Args:
            logits: Model predictions [batch, seq_len, vocab_size]
            labels: Ground truth labels [batch, seq_len]
            pad_token_id: Token ID used for padding
        
        Returns:
            Cross-entropy loss
        """
        # Shift logits and labels for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten
        loss_fct = nn.CrossEntropyLoss(ignore_index=pad_token_id)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        return loss
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Dictionary containing 'input_ids' and 'labels'
        
        Returns:
            Dictionary of metrics
        """
        self.model.train()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        labels = batch.get('labels', input_ids).to(self.device)
        
        # Forward pass with mixed precision
        if self.config.precision in ["bf16", "fp16"]:
            with autocast(dtype=torch.bfloat16 if self.config.precision == "bf16" 
                         else torch.float16):
                outputs = self.model(input_ids)
                loss = self._compute_loss(outputs.logits, labels)
        else:
            outputs = self.model(input_ids)
            loss = self._compute_loss(outputs.logits, labels)
        
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Return unscaled loss for logging
        loss_value = loss.item() * self.config.gradient_accumulation_steps
        
        # Calculate tokens per second
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        tokens_per_step = batch_size * seq_len * self.config.gradient_accumulation_steps
        time_per_step = time.time() - self._step_start_time
        
        return {
            'loss': loss_value,
            'tokens_per_second': tokens_per_step / time_per_step if time_per_step > 0 else 0,
            'learning_rate': self.scheduler.get_lr(),
        }
    
    def _optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        if self.scaler:
            # Unscale gradients
            self.scaler.unscale_(self.optimizer)
        
        # Clip gradients
        if self.is_distributed:
            # All-reduce gradients across devices
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )
        else:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )
        
        # Optimizer step
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        
        # Update learning rate
        self.scheduler.step()
    
    def train(self):
        """
        Main training loop.
        
        This method runs the complete training process:
        1. Iterate over training dataloader
        2. Compute loss and perform backpropagation
        3. Apply gradient accumulation
        4. Update parameters with optimizer
        5. Log metrics
        6. Evaluate periodically
        7. Save checkpoints
        """
        logger.info("Starting training...")
        logger.info(f"Max steps: {self.config.max_steps}")
        logger.info(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        logger.info(f"Precision: {self.config.precision}")
        
        # Training loop
        while self.global_step < self.config.max_steps:
            self.epoch += 1
            
            for batch_idx, batch in enumerate(self.config.train_dataloader):
                self._step_start_time = time.time()
                
                # Training step
                metrics = self._training_step(batch)
                
                # Accumulate gradients
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self._optimizer_step()
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        self._log_metrics(metrics)
                    
                    # Evaluation
                    if self.config.eval_dataloader and \
                       self.global_step % self.config.eval_steps == 0:
                        eval_metrics = self.evaluate()
                        self._log_metrics(eval_metrics, prefix="eval_")
                    
                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self._save_checkpoint(f"checkpoint-{self.global_step}")
                    
                    # Check for max steps
                    if self.global_step >= self.config.max_steps:
                        break
        
        # Final save
        self._save_checkpoint("final")
        logger.info("Training complete!")
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on the evaluation dataset.
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Running evaluation...")
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.config.eval_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch.get('labels', input_ids).to(self.device)
                
                # Forward pass
                if self.config.precision == "bf16":
                    with autocast(dtype=torch.bfloat16):
                        outputs = self.model(input_ids)
                        loss = self._compute_loss(outputs.logits, labels)
                elif self.config.precision == "fp16":
                    with autocast(dtype=torch.float16):
                        outputs = self.model(input_ids)
                        loss = self._compute_loss(outputs.logits, labels)
                else:
                    outputs = self.model(input_ids)
                    loss = self._compute_loss(outputs.logits, labels)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
        
        logger.info(f"Evaluation - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
        
        self.model.train()
        
        return {
            'eval_loss': avg_loss,
            'eval_perplexity': perplexity,
        }
    
    def _log_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """Log metrics to console and file."""
        # Add prefix if specified
        if prefix:
            metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
        
        # Update tracker
        self.metrics.update(metrics, self.global_step)
        
        # Log to console
        metrics_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) 
                                  else f"{k}: {v}" 
                                  for k, v in metrics.items()])
        logger.info(f"Step {self.global_step} | {metrics_str}")
    
    def _save_checkpoint(self, checkpoint_name: str):
        """
        Save training checkpoint.
        
        Args:
            checkpoint_name: Name for the checkpoint
        """
        if self.is_distributed and self.local_rank != 0:
            return  # Only save on rank 0
        
        checkpoint_dir = os.path.join(self.config.output_dir, checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Determine model for saving
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        # Save model
        model_path = os.path.join(checkpoint_dir, "model.pt")
        torch.save(model_to_save.state_dict(), model_path)
        
        # Save optimizer
        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
        torch.save(self.optimizer.state_dict(), optimizer_path)
        
        # Save scheduler
        scheduler_path = os.path.join(checkpoint_dir, "scheduler.pt")
        torch.save(self.scheduler.state_dict(), scheduler_path)
        
        # Save scaler if using mixed precision
        if self.scaler:
            scaler_path = os.path.join(checkpoint_dir, "scaler.pt")
            torch.save(self.scaler.state_dict(), scaler_path)
        
        # Save training state
        training_state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
        }
        state_path = os.path.join(checkpoint_dir, "training_state.pt")
        torch.save(training_state, state_path)
        
        # Save config
        config_path = os.path.join(checkpoint_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        logger.info(f"Checkpoint saved: {checkpoint_dir}")
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
    
    def _load_checkpoint(self, checkpoint_path: str):
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
        """
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        # Load model
        model_path = os.path.join(checkpoint_path, "model.pt")
        if os.path.exists(model_path):
            model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
            model_to_load.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # Load optimizer
        optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
        
        # Load scheduler
        scheduler_path = os.path.join(checkpoint_path, "scheduler.pt")
        if os.path.exists(scheduler_path):
            scheduler_state = torch.load(scheduler_path, map_location=self.device)
            self.scheduler.load_state_dict(scheduler_state)
        
        # Load scaler
        scaler_path = os.path.join(checkpoint_path, "scaler.pt")
        if os.path.exists(scaler_path) and self.scaler:
            self.scaler.load_state_dict(torch.load(scaler_path, map_location=self.device))
        
        # Load training state
        state_path = os.path.join(checkpoint_path, "training_state.pt")
        if os.path.exists(state_path):
            training_state = torch.load(state_path, map_location=self.device)
            self.global_step = training_state['global_step']
            self.epoch = training_state['epoch']
        
        logger.info(f"Checkpoint loaded. Resuming from step {self.global_step}")
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to save disk space."""
        if not os.path.exists(self.config.output_dir):
            return
        
        # Get all checkpoint directories
        checkpoints = [
            d for d in os.listdir(self.config.output_dir)
            if os.path.isdir(os.path.join(self.config.output_dir, d))
        ]
        
        # Sort by modification time
        checkpoints.sort(
            key=lambda x: os.path.getmtime(os.path.join(self.config.output_dir, x)),
            reverse=True
        )
        
        # Remove old checkpoints
        if len(checkpoints) > self.config.save_total_limit:
            for checkpoint in checkpoints[self.config.save_total_limit:]:
                checkpoint_dir = os.path.join(self.config.output_dir, checkpoint)
                import shutil
                shutil.rmtree(checkpoint_dir)
                logger.info(f"Removed old checkpoint: {checkpoint}")
    
    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def summary(self) -> Dict[str, Any]:
        """Get training summary."""
        return {
            'total_steps': self.config.max_steps,
            'current_step': self.global_step,
            'current_epoch': self.epoch,
            'learning_rate': self.scheduler.get_lr(),
            'trainable_parameters': self.get_trainable_parameters(),
            'precision': self.config.precision,
            'gradient_checkpointing': self.config.use_gradient_checkpointing,
            'torch_compile': self.config.use_compile,
        }


# ============================================================================
# Utility Functions for Training
# ============================================================================

def create_simple_dataset(
    vocab_size: int = 32000,
    num_samples: int = 10000,
    seq_length: int = 512,
) -> Dataset:
    """
    Create a simple random dataset for testing.
    
    Args:
        vocab_size: Vocabulary size
        num_samples: Number of samples
        seq_length: Sequence length
    
    Returns:
        Simple Dataset
    """
    class SimpleDataset(Dataset):
        def __init__(self, num_samples, seq_length, vocab_size):
            self.num_samples = num_samples
            self.seq_length = seq_length
            self.vocab_size = vocab_size
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Generate random tokens (skip special tokens at beginning)
            input_ids = torch.randint(
                4, self.vocab_size, (self.seq_length,), dtype=torch.long
            )
            return {'input_ids': input_ids, 'labels': input_ids.clone()}
    
    return SimpleDataset(num_samples, seq_length, vocab_size)


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a DataLoader with optimal settings.
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
    
    Returns:
        Configured DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True,
    )


# ============================================================================
# Example Usage
# ============================================================================

def example_training():
    """
    Example training script demonstrating usage.
    """
    # Import model
    from codling.codling.model import CodlingConfig, CodlingForCausalLM
    
    # Create model config
    config = CodlingConfig(
        vocab_size=32000,
        d_model=768,
        n_layers=12,
        d_state=128,
    )
    
    # Create model
    model = CodlingForCausalLM(config)
    
    # Create dataset and dataloader
    train_dataset = create_simple_dataset(
        vocab_size=config.vocab_size,
        num_samples=1000,
        seq_length=512,
    )
    
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=4,
    )
    
    # Create trainer config
    trainer_config = TrainerConfig(
        model=model,
        train_dataloader=train_dataloader,
        output_dir="./codling_checkpoints",
        max_steps=100,
        learning_rate=3e-4,
        warmup_steps=10,
        gradient_accumulation_steps=4,
        use_gradient_checkpointing=True,
        precision="bf16",
        logging_steps=5,
        save_steps=50,
    )
    
    # Create trainer
    trainer = Trainer(trainer_config)
    
    # Train
    trainer.train()
    
    # Print summary
    print("Training Summary:")
    print(json.dumps(trainer.summary(), indent=2))


if __name__ == "__main__":
    example_training()
