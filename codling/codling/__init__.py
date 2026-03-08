"""
CODLING - Complete SSM Language Model
=====================================

CODLING is a state-of-the-art SSM-based language model combining:
- Mamba/S4: Selective State Space Models for efficient sequence modeling
- Hyena: Subquadratic token mixer for hybrid architectures
- Based: Buffer-based Linear Attention for efficient attention
- LongRoPE: Rotary Position Embeddings with extended context support

This package provides the complete model architecture including:
- CodlingConfig: Model configuration dataclass
- CodlingModel: Core language model without LM head
- CodlingForCausalLM: Full causal language model with output projection
- CodlingRMSNorm: LLaMA-style RMSNorm normalization
- CodlingMLP: LLaMA-style SwiGLU feedforward network
- CodlingAttention: Optional linear/based attention
- CodlingHyena: Optional Hyena token mixer
- LongRoPE: Extended context rotary embeddings

Configuration Options:
---------------------
- vocab_size: 50K-100K for code (default: 50,304)
- d_model: 512-2048 (default: 1024)
- n_layers: 12-48 (default: 24)
- d_state: 128-256 for SSM state dimension (default: 128)
- d_expand: 2-4 expansion factor for FFN (default: 2)
- use_hyena: Enable Hyena token mixer (hybrid mode)
- use_linear_attn: Enable Based linear attention (hybrid mode)
- use_longrope: Enable LongRoPE for 1M context
- max_position_embeddings: 1024-1M (default: 8192)

Model Sizes (approximate):
--------------------------
- Small (130M): d_model=768, n_layers=24
- Medium (350M): d_model=1024, n_layers=24
- Large (760M): d_model=1280, n_layers=32
- XL (1B+): d_model=2048, n_layers=48

Usage:
------
    from codling.codling.model import CodlingConfig, CodlingForCausalLM
    
    # Create model configuration
    config = CodlingConfig(
        vocab_size=50304,
        d_model=1024,
        n_layers=24,
        d_state=128,
    )
    
    # Initialize model
    model = CodlingForCausalLM(config)
    
    # Forward pass
    input_ids = torch.randint(0, config.vocab_size, (1, 512))
    outputs = model(input_ids)
    logits = outputs.logits

Supported Features:
------------------
- torch.compile for speed optimization
- Gradient checkpointing for memory efficiency
- BF16/FP16 mixed precision training
- Weight tying for output projection
- sliding window attention (for Based hybrid mode)
- KV caching for autoregressive generation

Paper References:
----------------
- Mamba: "Mamba: Linear-time Sequence Modeling with Selective State Spaces" (2024)
- Hyena: "Hyena: Subquadratic Inference and Training for State Space Models" (2023)
- Based: "Long Context Attention with Linear Complexity" (2024)
- LongRoPE: "LongRoPE: Extending LLM Context Window Beyond 100K Tokens" (2024)
- SwiGLU: "GLU Variants Improve Transformer" (2020)
- RMSNorm: "Root Mean Square Layer Normalization" (2019)

Version: 1.0.0
"""

# Core model exports
from codling.codling.model import (
    # Configuration
    CodlingConfig,
    
    # Core model
    CodlingModel,
    CodlingForCausalLM,
    
    # Components
    CodlingRMSNorm,
    CodlingMLP,
    CodlingRotaryEmbedding,
    LongRoPE,
    
    # Token mixers
    CodlingAttention,
    CodlingHyena,
    
    # SSM blocks
    CodlingSSMBlock,
    
    # Factory functions
    create_codling_model,
    create_codling_model_from_pretrained,
)

# Trainer exports
from codling.codling.trainer import (
    # Configuration
    TrainerConfig,
    
    # Main trainer
    Trainer,
    
    # Utilities
    CosineWarmupScheduler,
    MetricsTracker,
    detect_colab_gpu,
    count_parameters,
    create_simple_dataset,
    create_dataloader,
    
    # Distributed utilities
    setup_distributed,
    cleanup_distributed,
)

# Inference exports
from codling.codling.inference import (
    # Generator
    CodlingGenerator,
    LongContextGenerator,
    
    # Configuration
    GenerationConfig,
    GenerationMetrics,
    
    # State management
    CachedStates,
    PromptCache,
    
    # Callbacks
    StreamCallback,
    ProgressCallback,
    
    # Sampling
    SamplingStrategy,
    
    # Factory functions
    create_generator,
    stream_generate,
)

# Version info
__version__ = "1.0.0"
__author__ = "CODLING Team"

# __all__ for clean imports
__all__ = [
    # Configuration
    "CodlingConfig",
    
    # Core model
    "CodlingModel",
    "CodlingForCausalLM",
    
    # Components
    "CodlingRMSNorm",
    "CodlingMLP",
    "CodlingRotaryEmbedding",
    "LongRoPE",
    
    # Token mixers
    "CodlingAttention",
    "CodlingHyena",
    
    # SSM blocks
    "CodlingSSMBlock",
    
    # Factory functions
    "create_codling_model",
    "create_codling_model_from_pretrained",
    
    # Trainer
    "Trainer",
    "TrainerConfig",
    "CosineWarmupScheduler",
    "MetricsTracker",
    "detect_colab_gpu",
    "count_parameters",
    "create_simple_dataset",
    "create_dataloader",
    "setup_distributed",
    "cleanup_distributed",
    
    # Inference
    "CodlingGenerator",
    "LongContextGenerator",
    "GenerationConfig",
    "GenerationMetrics",
    "CachedStates",
    "PromptCache",
    "StreamCallback",
    "ProgressCallback",
    "SamplingStrategy",
    "create_generator",
    "stream_generate",
]
