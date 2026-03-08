"""
CODLING Hyena Operator Module
==============================

Subquadratic token mixer replacing attention with O(n log n) or O(n) operations.

Based on: Poli et al., 2023 - "Hyena: Subquadratic Inference and Training for 
State Space Models"

Key Features:
- HyenaOperator: Core operator implementing the Hyena attention replacement
- HyenaLayer: Complete layer matching attention interface for drop-in replacement
- MultiHeadHyena: Multi-head variant combining multiple Hyena heads
- FilterFunction: Learned convolution kernels via MLP (tanh GELU)
- GatedDepthwiseConv1d: Multi-scale depthwise convolutions with gating
- fft_conv1d: FFT-based convolution for O(n log n) complexity
- SinusoidalPositionEmbedding: Positional embeddings for filter functions

The Hyena formula: Hyena(q, k, v) = depthwise_gated_conv(φ(q), φ(k)) ⊙ v

This provides:
- O(n log n) or O(n) complexity vs O(n²) for attention
- Same interface as attention for drop-in replacement
- Causal convolution support for autoregressive generation
- Full gradient flow support
- Configurable filter order and depth

Usage:
    # As drop-in replacement for attention
    from codling.hyena import HyenaLayer
    
    layer = HyenaLayer(d_model=512, num_heads=8, causal=True)
    output = layer(x)  # Same interface as attention
"""

from .hyena import (
    # Core operator
    HyenaOperator,
    
    # Layer (attention replacement)
    HyenaLayer,
    
    # Multi-head variant
    MultiHeadHyena,
    
    # Filter function
    FilterFunction,
    
    # Gated depthwise convolution
    GatedDepthwiseConv1d,
    
    # FFT convolution utility
    fft_conv1d,
    
    # Positional embeddings
    SinusoidalPositionEmbedding,
    
    # Factory function
    create_hyena_layer,
)

__all__ = [
    # Core operator
    "HyenaOperator",
    
    # Layer (attention replacement)
    "HyenaLayer",
    
    # Multi-head variant
    "MultiHeadHyena",
    
    # Filter function
    "FilterFunction",
    
    # Gated depthwise convolution
    "GatedDepthwiseConv1d",
    
    # FFT convolution utility
    "fft_conv1d",
    
    # Positional embeddings
    "SinusoidalPositionEmbedding",
    
    # Factory function
    "create_hyena_layer",
]

__version__ = "1.0.0"
__author__ = "CODLING Team"
__paper__ = "Poli et al., 2023 - Hyena: Subquadratic Inference and Training for State Space Models"
