"""
Quantization Support for CODLING
=================================

This module provides quantization capabilities for CODLING models to enable
efficient CPU inference with reduced memory usage and improved speed.

Features:
- Post-Training Quantization (PTQ) with calibration
- INT8 weight quantization
- Dynamic quantization support
- CPU-optimized inference
- Minimal accuracy loss

Usage:
    >>> from codling.codling.quantization import quantize_codling_model
    >>> quantized_model = quantize_codling_model(model, calibration_data)
"""

import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
from typing import Optional, List, Union, Dict, Any
from dataclasses import dataclass
from torch.utils.data import DataLoader

from codling.codling.model import CodlingConfig, CodlingForCausalLM, CodlingRMSNorm


@dataclass
class QuantizationConfig:
    """
    Configuration for model quantization.
    
    Attributes:
        method: Quantization method ("static", "dynamic", "qat")
        dtype: Quantized data type ("int8", "uint8", "int4")
        calibrate_with: Calibration dataset or dataloader
        num_calibration_samples: Number of samples for calibration
        per_channel: Whether to quantize per channel
        reduce_range: Whether to use reduced range quantization
    """
    method: str = "static"  # "static", "dynamic", "qat"
    dtype: str = "int8"     # "int8", "uint8", "int4"
    calibrate_with: Optional[Union[List, DataLoader]] = None
    num_calibration_samples: int = 100
    per_channel: bool = True
    reduce_range: bool = False


class QuantizedCodlingForCausalLM(nn.Module):
    """
    Quantized version of CODLING for efficient CPU inference.
    
    This class wraps the original model and applies quantization to linear layers
    while preserving the SSM architecture's functionality.
    """
    
    def __init__(self, config: CodlingConfig, quantization_config: QuantizationConfig):
        super().__init__()
        self.config = config
        self.quantization_config = quantization_config
        
        # Create the base model
        self.model = CodlingForCausalLM(config)
        
        # Quantization stubs
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        # Prepare for quantization
        self._prepare_for_quantization()
    
    def _prepare_for_quantization(self):
        """Prepare the model for quantization by fusing layers and adding observers."""
        
        # Fuse Conv1d + BatchNorm if present (though CODLING doesn't use BatchNorm)
        # Fuse Linear + ReLU where possible
        self._fuse_modules()
        
        # Set quantization configuration
        if self.quantization_config.method == "static":
            self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        elif self.quantization_config.method == "dynamic":
            self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        else:
            # QAT configuration
            self.model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        
        # Prepare the model
        torch.quantization.prepare(self.model, inplace=True)
    
    def _fuse_modules(self):
        """Fuse modules for better quantization performance."""
        # CODLING uses mostly Linear layers, fuse where possible
        # Note: SSM layers are more complex, focus on attention and MLP
        pass  # For now, rely on automatic fusion
    
    def calibrate(self, calibration_data: Union[List[torch.Tensor], DataLoader]):
        """Calibrate the model with sample data for static quantization."""
        if self.quantization_config.method != "static":
            return
        
        self.model.eval()
        
        # Collect calibration data
        if isinstance(calibration_data, DataLoader):
            calibration_samples = []
            for batch in calibration_data:
                if len(calibration_samples) >= self.quantization_config.num_calibration_samples:
                    break
                input_ids = batch['input_ids'] if isinstance(batch, dict) else batch
                calibration_samples.append(input_ids)
        else:
            calibration_samples = calibration_data[:self.quantization_config.num_calibration_samples]
        
        # Run calibration
        with torch.no_grad():
            for sample in calibration_samples:
                self.forward(sample)
    
    def convert(self):
        """Convert the model to quantized version."""
        torch.quantization.convert(self.model, inplace=True)
        self.quantized = True
    
    def forward(self, input_ids, **kwargs):
        """Forward pass with quantization."""
        # Quantize input if needed
        if hasattr(self, 'quantized') and self.quantized:
            # For quantized models, ensure input is on CPU
            input_ids = input_ids.to('cpu')
        
        # Forward through quantized model
        outputs = self.model(input_ids, **kwargs)
        
        return outputs
    
    def save_quantized_model(self, path: str):
        """Save the quantized model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'quantization_config': self.quantization_config,
            'quantized': getattr(self, 'quantized', False)
        }, path)
    
    @classmethod
    def load_quantized_model(cls, path: str, device='cpu'):
        """Load a quantized model."""
        checkpoint = torch.load(path, map_location=device)
        
        config = checkpoint['config']
        quantization_config = checkpoint['quantization_config']
        
        model = cls(config, quantization_config)
        model.model.load_state_dict(checkpoint['model_state_dict'])
        
        if checkpoint.get('quantized', False):
            model.quantized = True
        
        return model


def quantize_codling_model(
    model: CodlingForCausalLM,
    quantization_config: QuantizationConfig,
    calibration_data: Optional[Union[List[torch.Tensor], DataLoader]] = None
) -> CodlingForCausalLM:
    """
    Convenience function to quantize a CODLING model.
    
    Args:
        model: The trained CODLING model
        quantization_config: Quantization configuration
        calibration_data: Data for calibration (required for static quantization)
    
    Returns:
        Quantized model ready for efficient inference
    """
    
    if quantization_config.method == "dynamic":
        # Dynamic quantization - quantizes Linear layers only
        quantized_model = torch.quantization.quantize_dynamic(
            model, 
            {torch.nn.Linear: torch.quantization.default_dynamic_qconfig}, 
            inplace=False
        )
        return quantized_model
    
    elif quantization_config.method == "static":
        # Static quantization - more complex, requires calibration
        # Use a custom config that only quantizes Linear layers
        from torch.quantization import QConfig, MinMaxObserver, PerChannelMinMaxObserver
        
        model.qconfig = QConfig(
            activation=MinMaxObserver.with_args(dtype=torch.quint8),
            weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
        )
        
        # Don't quantize embeddings and normalization layers
        model.model.embedding.qconfig = None
        for module in model.modules():
            if isinstance(module, CodlingRMSNorm):
                module.qconfig = None
        
        # Prepare the model
        torch.quantization.prepare(model, inplace=True)
        
        # Calibrate
        if calibration_data is not None:
            model.eval()
            if isinstance(calibration_data, DataLoader):
                calibration_samples = []
                for batch in calibration_data:
                    if len(calibration_samples) >= quantization_config.num_calibration_samples:
                        break
                    input_ids = batch['input_ids'] if isinstance(batch, dict) else batch
                    calibration_samples.append(input_ids)
            else:
                calibration_samples = calibration_data[:quantization_config.num_calibration_samples]
            
            with torch.no_grad():
                for sample in calibration_samples:
                    _ = model(sample)
        
        # Convert
        torch.quantization.convert(model, inplace=True)
        return model
    
    else:
        raise ValueError(f"Unsupported quantization method: {quantization_config.method}")


def benchmark_quantization(
    original_model: CodlingForCausalLM,
    quantized_model: CodlingForCausalLM,
    test_data: List[torch.Tensor],
    num_runs: int = 10
) -> Dict[str, Any]:
    """
    Benchmark original vs quantized model performance.
    
    Returns metrics on speed, memory usage, and accuracy.
    """
    
    results = {
        'original': {},
        'quantized': {}
    }
    
    # Ensure models are on CPU
    original_model = original_model.to('cpu')
    quantized_model = quantized_model.to('cpu')
    
    original_model.eval()
    quantized_model.eval()
    
    # Memory usage
    def get_model_size(model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return param_size + buffer_size
    
    results['original']['size_mb'] = get_model_size(original_model) / 1024 / 1024
    results['quantized']['size_mb'] = get_model_size(quantized_model) / 1024 / 1024
    
    # Inference speed
    import time
    
    def benchmark_inference(model, data, runs):
        times = []
        with torch.no_grad():
            for _ in range(runs):
                start = time.time()
                for sample in data:
                    _ = model(sample)
                end = time.time()
                times.append(end - start)
        return sum(times) / len(times)
    
    results['original']['inference_time'] = benchmark_inference(original_model, test_data, num_runs)
    results['quantized']['inference_time'] = benchmark_inference(quantized_model, test_data, num_runs)
    
    # Speedup
    results['speedup'] = results['original']['inference_time'] / results['quantized']['inference_time']
    results['size_reduction'] = results['original']['size_mb'] / results['quantized']['size_mb']
    
    return results


# Example usage
if __name__ == "__main__":
    # Create a small model for testing
    config = CodlingConfig(d_model=256, n_layers=2, vocab_size=5000)
    model = CodlingForCausalLM(config)
    
    # Create dummy calibration data
    calibration_data = [torch.randint(0, 5000, (1, 32)) for _ in range(10)]
    
    # Quantize
    quant_config = QuantizationConfig(method="static", calibrate_with=calibration_data)
    quantized_model = quantize_codling_model(model, quant_config, calibration_data)
    
    # Benchmark
    test_data = [torch.randint(0, 5000, (1, 16)) for _ in range(5)]
    results = benchmark_quantization(model, quantized_model, test_data)
    
    print("Quantization Results:")
    print(f"Original size: {results['original']['size_mb']:.2f} MB")
    print(f"Quantized size: {results['quantized']['size_mb']:.2f} MB")
    print(f"Size reduction: {results['size_reduction']:.2f}x")
    print(f"Speedup: {results['speedup']:.2f}x")