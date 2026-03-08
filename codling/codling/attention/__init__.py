"""
CODLING Linear Attention Module
===============================

Optional hybrid component for CODLING - Based (Buffer-based Linear Attention) implementation.
Provides 4.5x faster inference than standard attention while maintaining competitive quality.

Based on: Arora et al., 2024 - "Long Context Attention with Linear Complexity"

Key Features:
- LinearAttention: Core linear attention with feature map approximation
- Tiled Flash Linear Attention (TFLA): Memory-efficient chunked computation
- Buffer-based state management for streaming
- Feature map: ELU+1 approximation for kernel-based attention
- Causal masking support for autoregressive generation

This is an OPTIONAL component - can be enabled in the full model for hybrid attention + SSM.
"""

from .linear_attn import (
    LinearAttention,
    TiledFlashLinearAttention,
    BasedLinearAttention,
    causal_mask,
    elu_feature_map,
    apply_causal_mask,
    BufferBasedState,
)

__all__ = [
    "LinearAttention",
    "TiledFlashLinearAttention",
    "BasedLinearAttention",
    "BufferBasedState",
    "causal_mask",
    "elu_feature_map",
    "apply_causal_mask",
]

__version__ = "1.0.0"
