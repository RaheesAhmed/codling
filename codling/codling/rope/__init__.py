"""
RoPE (Rotary Position Embedding) module for CODLING.

This module provides various implementations of Rotary Position Embeddings
to support different context lengths and efficiency requirements:

- RotaryPositionEmbedding: Standard RoPE implementation
- LongRoPE: Extended RoPE for 1M+ token context (DeepSeek 2024)
- NTKScaledRoPE: NTK-aware RoPE scaling
- DualChunkLongRoPE: Chunked attention with dual RoPE
- create_long_rope: Factory function for easy LongRoPE creation
- apply_rope_scaling: Utility for dynamic RoPE application

Usage:
    from codling.rope import LongRoPE, create_long_rope
    
    # Create RoPE for 1M context
    rope = create_long_rope(
        dim=128,
        training_length=4096,
        target_length=1048576
    )
    
    # Get embeddings
    cos, sin = rope(seq_len=8192)
    
    # Apply to query/key
    q_rotated, k_rotated = rope.apply_rotary_pos_emb(q, k, cos, sin)
"""

from .lrope import (
    LongRoPE,
    DualChunkLongRoPE,
    NTKScaledRoPE,
    RotaryPositionEmbedding,
    create_long_rope,
    apply_rope_scaling,
)

__all__ = [
    # Core RoPE implementations
    "RotaryPositionEmbedding",
    "LongRoPE",
    "NTKScaledRoPE",
    "DualChunkLongRoPE",
    
    # Factory functions
    "create_long_rope",
    "apply_rope_scaling",
]

# Version info
__version__ = "1.0.0"
__longrope_version__ = "DeepSeek-2024"
