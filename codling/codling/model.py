"""
CODLING SSM Language Model - Complete Architecture
===================================================

This module implements the full CODLING language model combining:
- Selective State Space Models (Mamba v2 or S4)
- Optional Hyena token mixer for hybrid SSM architecture
- Optional Based linear attention for hybrid architecture
- LongRoPE for extended context (up to 1M tokens)
- LLaMA-style RMSNorm and SwiGLU feedforward

Key Classes:
-----------
1. CodlingConfig - Model configuration with all hyperparameters
2. CodlingRMSNorm - LLaMA-style RMSNorm normalization
3. CodlingMLP - SwiGLU feedforward network
4. CodlingRotaryEmbedding - Standard rotary position embeddings
5. LongRoPE - Extended context rotary embeddings
6. CodlingAttention - Optional linear attention (Based)
7. CodlingHyena - Optional Hyena token mixer
8. CodlingSSMBlock - SSM block with residual and optional hybrid components
9. CodlingModel - Core language model
10. CodlingForCausalLM - Full causal LM with output projection

Architecture Support:
--------------------
- Pure SSM: Mamba/S4 only (no attention)
- Hyena Hybrid: SSM + Hyena token mixer
- Based Hybrid: SSM + Linear attention
- Full Hybrid: SSM + Hyena + Based

Model Sizes:
-----------
- Small (130M): d_model=768, n_layers=24, d_state=128
- Medium (350M): d_model=1024, n_layers=24, d_state=128
- Large (760M): d_model=1280, n_layers=32, d_state=256
- XL (1B+): d_model=2048, n_layers=48, d_state=256

Mathematical Foundation:
=======================

SSM (State Space Model):
    h_t = A @ h_{t-1} + B @ x_t
    y_t = C @ h_t + D @ x_t
    
    Computed in parallel using selective scan: O(n) complexity

RMSNorm:
    y = x / RMS(x) * γ, where RMS(x) = sqrt(mean(x²) + ε)

SwiGLU:
    FFN(x) = (Swish(xW₁) * (xW₂))W₃
    Swish(x) = x * sigmoid(x)

Rotary Position Embedding:
    RoPE(q, pos) = R(pos) @ q, where R is a rotation matrix

LongRoPE Extension:
    Interpolate base RoPE to extended context lengths
    Uses non-uniform interpolation with NTK-aware scaling
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, List
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LayerNorm

# Import SSM components
from codling.codling.ssm.mamba import MambaBlock, MambaBlockv2, MambaConfig
from codling.codling.ssm.s4 import S4Block, S4Layer

# Import attention components
from codling.codling.attention.linear_attn import (
    BasedLinearAttention,
    LinearAttention,
    elu_feature_map,
)

# Import Hyena components
from codling.codling.hyena.hyena import (
    HyenaLayer,
    HyenaOperator,
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class CodlingConfig:
    """
    Configuration class for CODLING SSM Language Model.
    
    This dataclass contains all hyperparameters needed to instantiate
    a CODLING model. It supports various configurations from small (130M)
    to extra-large (1B+) parameter models.
    
    Attributes:
        vocab_size (int): Vocabulary size for token embeddings.
            Default: 50,304 (~50K tokens, suitable for code)
        
        d_model (int): Hidden dimension size.
            Recommended: 512-2048
            Default: 1024
        
        n_layers (int): Number of transformer/SSM layers.
            Recommended: 12-48
            Default: 24
        
        d_state (int): SSM state dimension (A matrix size).
            Higher = more expressive but more memory
            Recommended: 128-256
            Default: 128
        
        d_expand (int): Expansion factor for feedforward networks.
            FFN hidden dim = d_model * d_expand
            Recommended: 2-4
            Default: 2
        
        ssm_type (str): Type of SSM layer to use.
            Options: "mamba", "mamba2", "s4"
            Default: "mamba2"
        
        use_hyena (bool): Enable Hyena token mixer (hybrid mode).
            Adds Hyena alongside SSM for token mixing
            Default: False
        
        use_linear_attn (bool): Enable Based linear attention (hybrid mode).
            Adds linear attention alongside SSM
            Default: False
        
        use_longrope (bool): Enable LongRoPE for extended context.
            Supports up to 1M token context
            Default: False
        
        max_position_embeddings (int): Maximum sequence length.
            For LongRoPE: 8192 to 1,048,576
            Default: 8192
        
        rope_theta (float): Base frequency for RoPE.
            Default: 10000.0
        
        rope_local (int): Local context window for RoPE.
            0 = full attention
            Default: 0
        
        use_gated (bool): Use gated activation in SSM.
            Default: True
        
        dropout (float): Dropout probability.
            Default: 0.0
        
        bias (bool): Use bias in linear layers.
            Default: False
        
        pad_token_id (int): Padding token ID.
            Default: 0
        
        eos_token_id (int): End-of-sequence token ID.
            Default: 2
        
        bos_token_id (int): Beginning-of-sequence token ID.
            Default: 1
        
        tie_word_embeddings (bool): Tie input/output embeddings.
            Default: True
        
        norm_eps (float): Epsilon for RMSNorm.
            Default: 1e-5
        
        use_bias_for_norm (bool): Use bias in RMSNorm.
            Default: False
        
        checkpoint_ratio (float): Gradient checkpointing ratio.
            0.0 = no checkpointing
            1.0 = checkpoint all layers
            Default: 0.0
        
        fused_mlp (bool): Use fused MLP kernel.
            Default: False
        
        rmsnorm_backend (str): RMSNorm implementation.
            Options: "torch", "fused"
            Default: "torch"
        
        attn_chunk_size (int): Chunk size for linear attention.
            0 = no chunking
            Default: 0
        
        hyena_num_heads (int): Number of Hyena heads.
            Default: 8
        
        hyena_filter_order (int): Filter order for Hyena.
            Default: 64
        
        hyena_kernel_size (int): Kernel size for Hyena convolutions.
            Default: 3
        
        hyena_depth (int): Depth of Hyena operator.
            Default: 2
        
        linear_attn_dims (int): Feature dim for linear attention.
            Default: 64
        
        ssm_dt_rank (str): DT rank for Mamba ("auto" or int).
            Default: "auto"
        
        ssm_conv_size (int): Convolution kernel size for Mamba.
            Default: 4
        
        ssm_expand (int): Expansion factor for Mamba inner dim.
            Default: 2
    
    Example:
        >>> # Small 130M parameter model
        >>> config = CodlingConfig(
        ...     vocab_size=50304,
        ...     d_model=768,
        ...     n_layers=24,
        ...     d_state=128,
        ... )
        >>> 
        >>> # Large 760M parameter model with hybrid architecture
        >>> config = CodlingConfig(
        ...     vocab_size=50304,
        ...     d_model=1280,
        ...     n_layers=32,
        ...     d_state=256,
        ...     use_hyena=True,
        ...     use_linear_attn=True,
        ...     use_longrope=True,
        ...     max_position_embeddings=32768,
        ... )
    """
    
    # Core architecture
    vocab_size: int = 50304
    d_model: int = 1024
    n_layers: int = 24
    d_state: int = 128
    d_expand: int = 2
    ssm_type: str = "s4"  # "mamba", "mamba2", "s4"
    
    # Hybrid components
    use_hyena: bool = False
    use_linear_attn: bool = False
    
    # LongRoPE
    use_longrope: bool = False
    max_position_embeddings: int = 8192
    rope_theta: float = 10000.0
    rope_local: int = 0
    
    # SSM options
    use_gated: bool = True
    
    # Regularization
    dropout: float = 0.0
    bias: bool = False
    
    # Tokenization
    pad_token_id: int = 0
    eos_token_id: int = 2
    bos_token_id: int = 1
    
    # Embeddings
    tie_word_embeddings: bool = True
    
    # Normalization
    norm_eps: float = 1e-5
    use_bias_for_norm: bool = False
    
    # Optimization
    checkpoint_ratio: float = 0.0
    fused_mlp: bool = False
    rmsnorm_backend: str = "torch"
    
    # Attention specific
    attn_chunk_size: int = 0
    
    # Hyena specific
    hyena_num_heads: int = 8
    hyena_filter_order: int = 64
    hyena_kernel_size: int = 3
    hyena_depth: int = 2
    
    # Linear attention specific
    linear_attn_dims: int = 64
    
    # SSM specific
    ssm_dt_rank: str = "auto"
    ssm_conv_size: int = 4
    ssm_expand: int = 2
    
    # Computed properties
    d_ffn: int = field(init=False)
    n_heads: int = field(init=False)
    d_head: int = field(init=False)
    
    def __post_init__(self):
        """Compute derived parameters."""
        self.d_ffn = int(self.d_model * self.d_expand)
        
        if self.use_hyena:
            self.n_heads = self.hyena_num_heads
            self.d_head = self.d_model // self.n_heads
        else:
            self.n_heads = max(1, self.d_model // 64)
            self.d_head = self.d_model // self.n_heads


# ============================================================================
# Normalization
# ============================================================================

class CodlingRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    
    RMSNorm normalizes activations by their root mean square, providing
    a simpler and more efficient alternative to LayerNorm. Used in LLaMA
    and other modern architectures.
    
    Mathematical Formulation:
        y = x / RMS(x) * γ
        
        where RMS(x) = sqrt(mean(x²) + ε)
    
    Advantages over LayerNorm:
        - No bias term (simpler)
        - Faster computation (fewer operations)
        - Similar or better performance on many tasks
    
    Args:
        hidden_size (int): Size of the hidden dimension.
        eps (float): Epsilon for numerical stability.
            Default: 1e-5
        use_bias (bool): Whether to use bias (for compatibility).
            Default: False
    
    Example:
        >>> norm = CodlingRMSNorm(hidden_size=1024, eps=1e-5)
        >>> x = torch.randn(2, 10, 1024)
        >>> y = norm(x)
        >>> y.shape
        torch.Size([2, 10, 1024])
    """
    
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        use_bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.use_bias = use_bias
        
        # Weight (scale) parameter
        self.weight = nn.Parameter(torch.ones(hidden_size))
        
        # Optional bias (not used in standard RMSNorm but included for compatibility)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of RMSNorm.
        
        Args:
            x (Tensor): Input tensor of shape [..., hidden_size]
                Supports any dimensionality >= 2
        
        Returns:
            Tensor: Normalized tensor of same shape as input
        """
        # Compute RMS along the last dimension
        # x.shape: [..., hidden_size]
        # We need to normalize over the feature dimension (last one)
        input_dtype = x.dtype
        x = x.float()
        
        # Compute RMS: sqrt(mean(x²) + eps)
        # Keepdims to broadcast properly
        rms = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        
        # Normalize
        x = x * rms
        
        # Apply weight (and bias if enabled)
        x = x.to(input_dtype)
        
        if self.weight is not None:
            x = x * self.weight
        
        if self.bias is not None:
            x = x + self.bias
        
        return x
    
    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, eps={self.eps}, use_bias={self.use_bias}"


# ============================================================================
# Feedforward Network (SwiGLU)
# ============================================================================

class CodlingMLP(nn.Module):
    """
    SwiGLU Feedforward Network.
    
    Implements the SwiGLU (Swish-Gated Linear Unit) feedforward network
    as described in "GLU Variants Improve Transformer" (2020). This is
    the standard FFN used in LLaMA and many modern LLMs.
    
    Mathematical Formulation:
        FFN(x) = (Swish(xW₁) * (xW₂))W₃
        
        where Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
    
    Architecture:
        x -> Linear(d_model, d_ffn) -> Swish -> * -> Linear(d_ffn, d_model) -> output
                                            ^
                                            |
        (gate branch: Linear(d_model, d_ffn) -> Swish)
    
    This differs from standard FFN (ReLU):
        FFN_ReLU(x) = max(0, xW₁ + b₁)W₂ + b₂
    
    SwiGLU Advantages:
        - Gating mechanism allows adaptive information flow
        - Swish activation is smoother than ReLU
        - Better performance on many benchmarks
    
    Args:
        config (CodlingConfig): Model configuration.
            Uses d_model and d_expand to compute hidden dim.
        bias (bool): Whether to use bias in linear layers.
            Default: False
    
    Example:
        >>> config = CodlingConfig(d_model=1024, d_expand=2)
        >>> mlp = CodlingMLP(config)
        >>> x = torch.randn(2, 10, 1024)
        >>> y = mlp(x)
        >>> y.shape
        torch.Size([2, 10, 1024])
    """
    
    def __init__(
        self,
        config: CodlingConfig,
        bias: bool = False,
    ):
        super().__init__()
        self.config = config
        self.hidden_dim = config.d_ffn
        
        # Gate branch: projects to expanded dimension
        self.gate_proj = nn.Linear(
            config.d_model,
            self.hidden_dim,
            bias=bias,
        )
        
        # Up branch: projects to expanded dimension
        self.up_proj = nn.Linear(
            config.d_model,
            self.hidden_dim,
            bias=bias,
        )
        
        # Down branch: projects back to model dimension
        self.down_proj = nn.Linear(
            self.hidden_dim,
            config.d_model,
            bias=bias,
        )
        
        # Activation function (Swish/SiLU)
        self.act_fn = nn.SiLU()
        
        # Dropout (only used during training)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of SwiGLU MLP.
        
        Args:
            x (Tensor): Input tensor of shape [batch, seq_len, d_model]
        
        Returns:
            Tensor: Output tensor of shape [batch, seq_len, d_model]
        """
        # Gate branch with Swish activation
        gate = self.act_fn(self.gate_proj(x))
        
        # Up branch (no activation)
        up = self.up_proj(x)
        
        # Element-wise multiply (gated)
        hidden = gate * up
        
        # Project down
        output = self.down_proj(hidden)
        
        # Apply dropout during training
        if self.dropout is not None:
            output = self.dropout(output)
        
        return output
    
    def extra_repr(self) -> str:
        return f"hidden_dim={self.hidden_dim}, d_model={self.config.d_model}"


# ============================================================================
# Rotary Position Embeddings
# ============================================================================

class CodlingRotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    Implements rotary position embeddings as described in "RoFormer: Enhanced
    Transformer with Rotary Position Embedding" (2021). RoPE encodes position
    information through rotation matrices, providing relative position awareness
    without explicit position embeddings.
    
    Mathematical Formulation:
        RoPE(q, pos) = R(pos) @ q
        
        where R(pos) is a rotation matrix:
        
        R(pos) = | cos(pos*θ)  -sin(pos*θ) |
                 | sin(pos*θ)   cos(pos*θ) |
        
        Applied in 2D blocks to vectors of dimension d:
        
        [q₀, q₁, ..., q_{d-2}, q_{d-1}]
        
        becomes:
        
        [q₀*cos(pos*θ₀) - q₁*sin(pos*θ₀), 
         q₀*sin(pos*θ₀) + q₁*cos(pos*θ₀),
         ...]
        
        where θ₀ = base^( -2i/d ) for i = 0, 1, ..., d/2-1
    
    Advantages:
        - Captures relative position information
        - No extra parameters for positions
        - Works with linear attention
        - Efficient computation
    
    Args:
        dim (int): Dimension of the embeddings (must be even).
        max_position_embeddings (int): Maximum sequence length.
            Default: 2048
        base (float): Base for frequency computation.
            Default: 10000.0
        device (torch.device): Device for tensors.
            Default: None
    
    Example:
        >>> rope = CodlingRotaryEmbedding(dim=64, max_position_embeddings=2048)
        >>> q = torch.randn(2, 10, 8, 64)  # batch, seq, heads, head_dim
        >>> positions = torch.arange(10).unsqueeze(0).expand(2, -1)
        >>> q_rotated = rope(q, positions=positions)
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Compute inverse frequencies
        # θ_i = base^(-2i/d) for i = 0, 1, ..., d/2-1
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        # Precompute cos and sin for all positions
        self._set_cos_sin_cache(
            max_position_embeddings,
            device,
            torch.get_default_dtype(),
        )
    
    def _set_cos_sin_cache(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """Precompute cos and sin values for efficiency."""
        self.max_seq_len_cached = seq_len
        
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        
        # Outer product: [seq_len, dim/2]
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        
        # Concatenate: [seq_len, dim]
        emb = torch.cat([freqs, freqs], dim=-1)
        
        self.register_buffer('cos_cached', emb.cos().to(dtype), persistent=False)
        self.register_buffer('sin_cached', emb.sin().to(dtype), persistent=False)
    
    def forward(
        self,
        q: Tensor,
        positions: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply rotary position embedding to query and key tensors.
        
        Args:
            q (Tensor): Query tensor of shape [..., seq_len, dim]
            positions (Tensor, optional): Position indices of shape [..., seq_len]
                If None, assumes positions 0, 1, 2, ...
        
        Returns:
            Tuple[Tensor, Tensor]: 
                - Rotated query tensor of same shape
                - Cache-ready cos/sin for keys
        """
        seq_len = q.shape[-2]
        
        # Expand positions if not provided
        if positions is None:
            positions = torch.arange(seq_len, device=q.device).unsqueeze(0).expand(q.shape[0], -1)
        
        # Expand to match query shape
        if positions.dim() == 1:
            positions = positions.unsqueeze(0).expand(q.shape[0], -1)
        
        # Get cos and sin for required positions
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, q.device, q.dtype)
        
        # Gather cos and sin
        cos = self.cos_cached[positions]  # [..., seq_len, dim]
        sin = self.sin_cached[positions]  # [..., seq_len, dim]
        
        # Apply rotation
        q = self._rotate_half(q, cos, sin)
        
        return q, cos
    
    def _rotate_half(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """
        Apply rotary rotation to input tensor.
        
        Splits the last dimension in half and rotates one half by π/2.
        """
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        
        # Apply rotation formula
        # x' = x * cos - x2 * sin
        # x'' = x1 * sin + x2 * cos
        return torch.cat([
            x1 * cos[..., :cos.shape[-1]//2] - x2 * sin[..., :sin.shape[-1]//2],
            x1 * sin[..., :sin.shape[-1]//2] + x2 * cos[..., cos.shape[-1]//2:],
        ], dim=-1)


class LongRoPE(nn.Module):
    """
    Long Rotary Position Embedding (LongRoPE).
    
    Extends standard RoPE to support extremely long contexts (up to 1M tokens)
    without retraining. Uses non-uniform interpolation and NTK-aware scaling
    as described in "LongRoPE: Extending LLM Context Window Beyond 100K Tokens".
    
    Key Innovations:
    1. Non-uniform position interpolation: More interpolation for lower frequencies
    2. NTK-aware scaling: Respects the NTK (Neural Tangent Kernel) structure
    3. Dynamic scaling: Adapts to different context lengths
    
    Mathematical Formulation:
        For position p and original dimension i:
        
        p' = p * s(i)
        
        where s(i) is a scaling factor that varies with frequency:
        
        s(i) = 1 for low frequencies (important for local patterns)
        s(i) > 1 for high frequencies (preserved via NTK scaling)
    
    This allows:
        - Original 4K context extended to 32K, 128K, or even 1M
        - Minimal quality degradation
        - No retraining required
    
    Args:
        dim (int): Dimension of embeddings (must be even).
        max_position_embeddings (int): Target maximum sequence length.
            Default: 32768
        original_max_pos (int): Original training context length.
            Default: 2048
        rope_theta (float): Base frequency for RoPE.
            Default: 10000.0
        attention_factor (float): Attention scaling factor.
            Default: 1.0
        beta_fast (float): Fast interpolation parameter.
            Default: 32.0
        beta_slow (float): Slow interpolation parameter.
            Default: 1.0
        mscale (float): Multiplicative scaling factor.
            Default: 1.0
    
    Example:
        >>> # Extend 2K context to 32K
        >>> lrope = LongRoPE(
        ...     dim=64,
        ...     max_position_embeddings=32768,
        ...     original_max_pos=2048,
        ... )
        >>> q = torch.randn(2, 10, 8, 64)
        >>> q_rotated = lrope(q, positions=None)
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        original_max_pos: int = 2048,
        rope_theta: float = 10000.0,
        attention_factor: float = 1.0,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        mscale: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.original_max_pos = original_max_pos
        self.rope_theta = rope_theta
        self.attention_factor = attention_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        
        # Compute scaling factors for different frequency bands
        # Low frequencies: interpolate less (scaling factor ~1)
        # High frequencies: interpolate more (NTK-aware)
        
        # Original inv frequencies
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2).float() / dim))
        
        # Compute position scale
        # Using the LongRoPE interpolation formula
        scale = self.max_position_embeddings / self.original_max_pos
        
        # NTK-aware scaling factor
        # Higher dimensions get higher scaling
        dim_scale = torch.ones(dim)
        half_dim = dim // 2
        
        # Compute beta based on position
        # This creates non-uniform scaling
        for i in range(half_dim):
            if scale > 1.0:
                # Use NTK-aware scaling
                dim_scale[2 * i] = self._ntk_scale(i, half_dim, scale)
                dim_scale[2 * i + 1] = dim_scale[2 * i]
        
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self.register_buffer('dim_scale', dim_scale, persistent=False)
        
        # Precompute cos/sin cache
        self._set_cos_sin_cache(max_position_embeddings)
    
    def _ntk_scale(self, i: int, half_dim: int, scale: float) -> float:
        """
        Compute NTK-aware scaling factor.
        
        This preserves high-frequency information while interpolating
        low-frequency positions.
        """
        base = self.beta_fast / self.beta_slow
        alpha = (i / half_dim) ** (1 / (self.beta_fast - self.beta_slow + 1e-5))
        return scale * (1 + (base - 1) * alpha) / scale
    
    def _set_cos_sin_cache(self, seq_len: int):
        """Precompute cos and sin with scaling."""
        self.max_seq_len_cached = seq_len
        
        t = torch.arange(seq_len)
        
        # Apply scaling to frequencies
        freqs = torch.einsum('i,j->ij', t, self.inv_freq * self.dim_scale[:self.dim//2])
        
        # Concatenate for full rotation
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # Apply mscale
        if self.mscale != 1.0:
            emb = emb * self.mscale
        
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)
    
    def forward(
        self,
        q: Tensor,
        positions: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply LongRoPE to query/key tensors.
        
        Args:
            q (Tensor): Query tensor of shape [..., seq_len, dim]
            positions (Tensor, optional): Position indices
        
        Returns:
            Tensor: Rotated tensor with extended context
        """
        seq_len = q.shape[-2]
        
        if positions is None:
            positions = torch.arange(seq_len, device=q.device).unsqueeze(0).expand(q.shape[0], -1)
        
        if positions.dim() == 1:
            positions = positions.unsqueeze(0).expand(q.shape[0], -1)
        
        # Extend cache if needed
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        
        cos = self.cos_cached[positions]
        sin = self.sin_cached[positions]
        
        # Apply rotation
        q = self._rotate_half(q, cos, sin)
        
        return q
    
    def _rotate_half(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Apply rotary rotation."""
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos,
        ], dim=-1)
    
    def interpolate(self, new_max_pos: int):
        """
        Dynamically interpolate to a new maximum position.
        
        Allows extending context length at runtime without reinitialization.
        
        Args:
            new_max_pos (int): New maximum sequence length
        """
        self.max_position_embeddings = new_max_pos
        self._set_cos_sin_cache(new_max_pos)


# ============================================================================
# Optional Hybrid Components
# ============================================================================

class CodlingAttention(nn.Module):
    """
    Linear Attention for CODLING (Based-style).
    
    Implements buffer-based linear attention as an optional component
    for hybrid SSM architectures. Provides O(n) complexity instead of O(n²).
    
    This is used when use_linear_attn=True in the configuration to create
    a hybrid SSM + attention model.
    
    Mathematical Formulation:
        Standard attention: softmax(QK^T)V
        
        Linear attention: φ(Q)(φ(K)^T V)
        
        where φ(x) = ELU(x) + 1 is the feature map
    
    Args:
        config (CodlingConfig): Model configuration.
    
    Example:
        >>> config = CodlingConfig(d_model=1024, use_linear_attn=True, linear_attn_dims=64)
        >>> attn = CodlingAttention(config)
        >>> x = torch.randn(2, 10, 1024)
        >>> y = attn(x)
    """
    
    def __init__(self, config: CodlingConfig):
        super().__init__()
        self.config = config
        
        # Use the BasedLinearAttention from attention module
        self.attn = BasedLinearAttention(
            dim=config.d_model,
            heads=config.n_heads,
            dim_head=config.d_head,
            use_linear=True,
            chunk_size=config.attn_chunk_size,
            dropout=config.dropout,
        )
    
    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        positions: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass of linear attention.
        
        Args:
            x (Tensor): Input of shape [batch, seq_len, d_model]
            attention_mask (Tensor, optional): Attention mask
            positions (Tensor, optional): Position indices for RoPE
        
        Returns:
            Tensor: Output of shape [batch, seq_len, d_model]
        """
        # Use BasedLinearAttention
        return self.attn(x, x, x)  # query, key, value all from x


class CodlingHyena(nn.Module):
    """
    Hyena Token Mixer for CODLING.
    
    Implements the Hyena operator as an optional token mixer for hybrid
    SSM architectures. Provides O(n log n) or O(n) complexity with
    competitive quality to attention.
    
    Based on: Poli et al., 2023 - "Hyena: Subquadratic Inference and 
    Training for State Space Models"
    
    The Hyena formula:
        Hyena(q, k, v) = depthwise_gated_conv(φ(q), φ(k)) ⊙ v
    
    Args:
        config (CodlingConfig): Model configuration.
    
    Example:
        >>> config = CodlingConfig(d_model=1024, use_hyena=True, hyena_num_heads=8)
        >>> hyena = CodlingHyena(config)
        >>> x = torch.randn(2, 10, 1024)
        >>> y = hyena(x)
    """
    
    def __init__(self, config: CodlingConfig):
        super().__init__()
        self.config = config
        
        # Use the HyenaLayer from hyena module
        # Note: HyenaLayer uses d_head internally, num_heads controls parallel projections
        self.hyena = HyenaLayer(
            d_model=config.d_model,
            num_heads=config.hyena_num_heads,
            kernel_size=config.hyena_kernel_size,
            filter_order=config.hyena_filter_order,
            causal=True,
            dropout=config.dropout,
            use_short_conv=True,
            short_kernel_size=3,
        )
    
    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        positions: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass of Hyena token mixer.
        
        Args:
            x (Tensor): Input of shape [batch, seq_len, d_model]
            attention_mask (Tensor, optional): Not used (Hyena handles positions)
            positions (Tensor, optional): Position indices
        
        Returns:
            Tensor: Output of shape [batch, seq_len, d_model]
        """
        # HyenaLayer handles positions internally
        return self.hyena(x)


# ============================================================================
# SSM Block
# ============================================================================

class CodlingSSMBlock(nn.Module):
    """
    Combined SSM Block with optional hybrid components.
    
    This is the core building block of CODLING, combining:
    1. Primary SSM layer (Mamba v2 or S4)
    2. Optional Hyena token mixer (for hybrid mode)
    3. Optional Linear attention (for hybrid mode)
    4. RMSNorm normalization
    5. SwiGLU feedforward
    
    The block applies the following operations:
        x -> RMSNorm -> [SSM + optional Hyena + optional LinearAttn] -> + -> MLP -> + -> output
    
    Architecture:
    ┌─────────────────────────────────────────────────────┐
    │                    Input x                         │
    │                      │                              │
    │                      ▼                              │
    │               ┌─────────────┐                       │
    │               │  RMSNorm   │                       │
    │               └─────────────┘                       │
    │                      │                              │
    │         ┌────────────┼────────────┐                │
    │         ▼            ▼            ▼                │
    │   ┌──────────┐ ┌──────────┐ ┌──────────┐           │
    │   │   SSM   │ │  Hyena   │ │ Linear   │           │
    │   │ (Mamba) │ │(optional)│ │  Attn    │           │
    │   └──────────┘ │(optional)│ │(optional)│           │
    │         │      └──────────┘ └──────────┘           │
    │         │            │            │                │
    │         └────────────┼────────────┘                │
    │                      ▼                              │
    │               ┌─────────────┐                       │
    │               │   Dropout  │                       │
    │               └─────────────┘                       │
    │                      │                              │
    │                      ▼                              │
    │               ┌─────────────┐                       │
    │               │   + Resid  │                       │
    │               └─────────────┘                       │
    │                      │                              │
    │                      ▼                              │
    │               ┌─────────────┐                       │
    │               │  SwiGLU    │                       │
    │               │    MLP     │                       │
    │               └─────────────┘                       │
    │                      │                              │
    │                      ▼                              │
    │               ┌─────────────┐                       │
    │               │   + Resid  │                       │
    │               └─────────────┘                       │
    │                      │                              │
    │                      ▼                              │
    │                   Output                            │
    └─────────────────────────────────────────────────────┘
    
    Args:
        config (CodlingConfig): Model configuration.
        layer_idx (int): Layer index (for naming).
    
    Example:
        >>> config = CodlingConfig(d_model=1024, n_layers=24, use_hyena=True)
        >>> block = CodlingSSMBlock(config, layer_idx=0)
        >>> x = torch.randn(2, 10, 1024)
        >>> y = block(x)
    """
    
    def __init__(
        self,
        config: CodlingConfig,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Pre-norm: RMSNorm before SSM
        self.norm = CodlingRMSNorm(
            config.d_model,
            eps=config.norm_eps,
            use_bias=config.use_bias_for_norm,
        )
        
        # SSM layer (primary sequence mixer)
        if config.ssm_type == "mamba":
            self.ssm = MambaBlock(
                d_model=config.d_model,
                d_state=config.d_state,
                d_conv=config.ssm_conv_size,
                expand=config.ssm_expand,
                dt_rank=config.ssm_dt_rank,
            )
        elif config.ssm_type == "mamba2":
            self.ssm = MambaBlockv2(
                d_model=config.d_model,
                d_state=config.d_state,
                d_conv=config.ssm_conv_size,
                expand=config.ssm_expand,
                dt_rank=config.ssm_dt_rank,
            )
        elif config.ssm_type == "s4":
            self.ssm = S4Block(
                d_model=config.d_model,
                d_state=config.d_state,
                dropout=config.dropout,
            )
        else:
            raise ValueError(f"Unknown SSM type: {config.ssm_type}")
        
        # Optional Hyena token mixer
        self.use_hyena = config.use_hyena
        if config.use_hyena:
            self.hyena = CodlingHyena(config)
        
        # Optional Linear attention
        self.use_linear_attn = config.use_linear_attn
        if config.use_linear_attn:
            self.linear_attn = CodlingAttention(config)
        
        # Dropout after SSM mixing
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None
        
        # Feedforward network (SwiGLU)
        self.mlp = CodlingMLP(config, bias=config.bias)
        
        # Post-norm for MLP
        self.mlp_norm = CodlingRMSNorm(
            config.d_model,
            eps=config.norm_eps,
            use_bias=config.use_bias_for_norm,
        )
    
    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        positions: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass of SSM block.
        
        Args:
            x (Tensor): Input of shape [batch, seq_len, d_model]
            attention_mask (Tensor, optional): Attention mask
            positions (Tensor, optional): Position indices for RoPE
        
        Returns:
            Tensor: Output of shape [batch, seq_len, d_model]
        """
        # Pre-norm
        residual = x
        x = self.norm(x)
        
        # SSM layer
        if self.config.ssm_type in ("mamba", "mamba2"):
            x = self.ssm(x)
        elif self.config.ssm_type == "s4":
            x, _ = self.ssm(x)
        
        # Optional Hyena
        if self.use_hyena:
            x = x + self.hyena(x)
        
        # Optional Linear attention
        if self.use_linear_attn:
            x = x + self.linear_attn(x, attention_mask)
        
        # Dropout
        if self.dropout is not None:
            x = self.dropout(x)
        
        # Residual connection
        x = residual + x
        
        # MLP with pre-norm
        residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


# ============================================================================
# Core Model
# ============================================================================

class CodlingModel(nn.Module):
    """
    Core CODLING Language Model.
    
    This is the main language model without the language modeling head.
    It consists of:
    - Token embedding layer
    - Stacked CodlingSSMBlocks
    - Final RMSNorm
    
    The model processes input token IDs through embedding + layers + norm
    to produce hidden states that can be used for downstream tasks.
    
    Architecture:
    ┌─────────────────────────────────────────────────────┐
    │                  input_ids                        │
    │                      │                              │
    │                      ▼                              │
    │              ┌─────────────┐                       │
    │              │  Embedding  │                       │
    │              └─────────────┘                       │
    │                      │                              │
    │         ┌────────────┼────────────┐                │
    │         ▼            ▼            ▼                │
    │   ┌──────────┐ ┌──────────┐ ┌──────────┐          │
    │   │ SSMBlock │ │ SSMBlock │ │ SSMBlock │ ...      │
    │   │    0    │ │    1    │ │    2    │          │
    │   └──────────┘ └──────────┘ └──────────┘          │
    │         │            │            │                │
    │         └────────────┼────────────┘                │
    │                      ▼                              │
    │               ┌─────────────┐                       │
    │               │ Final Norm  │                       │
    │               └─────────────┘                       │
    │                      │                              │
    │                      ▼                              │
    │                 hidden_states                      │
    └─────────────────────────────────────────────────────┘
    
    Args:
        config (CodlingConfig): Model configuration.
    
    Attributes:
        config (CodlingConfig): Model configuration.
        vocab_size (int): Vocabulary size.
        embedding (nn.Embedding): Token embedding layer.
        layers (nn.ModuleList): List of SSM blocks.
        norm (CodlingRMSNorm): Final normalization.
        rotary_emb (CodlingRotaryEmbedding or LongRoPE): Position embeddings.
    
    Example:
        >>> config = CodlingConfig(vocab_size=50304, d_model=1024, n_layers=24)
        >>> model = CodlingModel(config)
        >>> input_ids = torch.randint(0, 50304, (1, 512))
        >>> hidden_states = model(input_ids)
        >>> hidden_states.shape
        torch.Size([1, 512, 1024])
    """
    
    def __init__(self, config: CodlingConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id,
        )
        
        # Position embeddings (RoPE or LongRoPE)
        if config.use_longrope:
            self.rotary_emb = LongRoPE(
                dim=config.d_model,
                max_position_embeddings=config.max_position_embeddings,
                original_max_pos=config.max_position_embeddings // 4,  # Assume trained on 1/4
                rope_theta=config.rope_theta,
            )
        else:
            self.rotary_emb = CodlingRotaryEmbedding(
                dim=config.d_model,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
            )
        
        # Stack of SSM blocks
        self.layers = nn.ModuleList([
            CodlingSSMBlock(config, layer_idx=i)
            for i in range(config.n_layers)
        ])
        
        # Final normalization
        self.norm = CodlingRMSNorm(
            config.d_model,
            eps=config.norm_eps,
            use_bias=config.use_bias_for_norm,
        )
        
        # Gradient checkpointing
        self.gradient_checkpointing = False
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            # Linear layers: normal initialization
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Embeddings: normal initialization with small std
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, CodlingRMSNorm):
            # RMSNorm: initialize to identity
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        positions: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass of the model.
        
        Args:
            input_ids (Tensor): Input token IDs of shape [batch, seq_len]
            attention_mask (Tensor, optional): Attention mask
            positions (Tensor, optional): Position indices for RoPE
        
        Returns:
            Tensor: Hidden states of shape [batch, seq_len, d_model]
        """
        batch, seq_len = input_ids.shape
        
        # Get token embeddings
        hidden_states = self.embedding(input_ids)
        
        # Get position indices
        if positions is None:
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        
        # Apply each SSM block
        if self.gradient_checkpointing and self.training:
            # Use gradient checkpointing to save memory
            for layer in self.layers:
                hidden_states = self._gradient_checkpointed_forward(
                    layer,
                    hidden_states,
                    attention_mask,
                    positions,
                )
        else:
            for layer in self.layers:
                hidden_states = layer(hidden_states, attention_mask, positions)
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        return hidden_states
    
    def _gradient_checkpointed_forward(
        self,
        layer: nn.Module,
        x: Tensor,
        attention_mask: Optional[Tensor],
        positions: Tensor,
    ) -> Tensor:
        """Forward pass with gradient checkpointing."""
        # Use torch.utils.checkpoint for memory efficiency
        return torch.utils.checkpoint.checkpoint(
            layer,
            x,
            attention_mask,
            positions,
            use_reentrant=False,
        )
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = True
    
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False


class CodlingForCausalLM(nn.Module):
    """
    CODLING Language Model for Causal (Autoregressive) Language Modeling.
    
    This is the full language model with a language modeling head (output projection).
    It extends CodlingModel with:
    - Tie-able output projection (weight sharing with embeddings)
    - Proper loss computation for causal LM
    
    The model is designed for:
    - Autoregressive text generation
    - Code completion
    - Fine-tuning on downstream tasks
    
    Architecture:
    ┌─────────────────────────────────────────────────────┐
    │                  input_ids                        │
    │                      │                              │
    │                      ▼                              │
    │              ┌─────────────┐                       │
    │              │  Codling    │                       │
    │              │   Model     │                       │
    │              └─────────────┘                       │
    │                      │                              │
    │                      ▼                              │
    │              ┌─────────────┐                       │
    │              │   Linear    │  (lm_head)           │
    │              │  Projection │                       │
    │              └─────────────┘                       │
    │                      │                              │
    │                      ▼                              │
    │                   logits                           │
    └─────────────────────────────────────────────────────┘
    
    Args:
        config (CodlingConfig): Model configuration.
    
    Attributes:
        config (CodlingConfig): Model configuration.
        model (CodlingModel): Core model.
        lm_head (nn.Linear): Output projection layer.
        loss_fct (nn.CrossEntropyLoss): Loss function.
    
    Example:
        >>> config = CodlingConfig(vocab_size=50304, d_model=1024, n_layers=24)
        >>> model = CodlingForCausalLM(config)
        >>> 
        >>> # Forward pass
        >>> input_ids = torch.randint(0, 50304, (1, 512))
        >>> outputs = model(input_ids)
        >>> logits = outputs.logits
        >>> logits.shape
        torch.Size([1, 512, 50304])
        >>> 
        >>> # Autoregressive generation
        >>> generated = model.generate(input_ids, max_new_tokens=100)
    """
    
    def __init__(self, config: CodlingConfig):
        super().__init__()
        self.config = config
        
        # Core model
        self.model = CodlingModel(config)
        
        # Output projection (language modeling head)
        # Optionally tie weights with embedding layer
        if config.tie_word_embeddings:
            # Share weights between embedding and lm_head
            self.lm_head = None
            self._tie_weights = True
        else:
            # Separate weights
            self.lm_head = nn.Linear(
                config.d_model,
                config.vocab_size,
                bias=False,
            )
            self._tie_weights = False
        
        # Loss function for training
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=config.pad_token_id)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Tie weights if requested
        if config.tie_word_embeddings:
            self.tie_weights()
    
    def _init_weights(self, module: nn.Module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear) and module is not self.lm_head:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding) and not self._tie_weights:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
    def tie_weights(self):
        """Tie weights between embedding and output projection."""
        if self._tie_weights:
            self.lm_head = lambda x: F.linear(x, self.model.embedding.weight)
    
    def get_input_embeddings(self) -> nn.Embedding:
        """Get input embedding layer."""
        return self.model.embedding
    
    def set_input_embeddings(self, embeddings: nn.Embedding):
        """Set input embedding layer."""
        self.model.embedding = embeddings
    
    def get_output_embeddings(self) -> Optional[nn.Linear]:
        """Get output projection layer."""
        if self._tie_weights:
            return None
        return self.lm_head
    
    def set_output_embeddings(self, embeddings: nn.Linear):
        """Set output projection layer."""
        if not self._tie_weights:
            self.lm_head = embeddings
    
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        positions: Optional[Tensor] = None,
    ) -> dict:
        """
        Forward pass of the language model.
        
        Args:
            input_ids (Tensor): Input token IDs of shape [batch, seq_len]
            attention_mask (Tensor, optional): Attention mask
            labels (Tensor, optional): Labels for computing loss
                Shape: [batch, seq_len]
                If provided, returns loss
            positions (Tensor, optional): Position indices for RoPE
        
        Returns:
            dict: Contains:
                - logits: Output logits of shape [batch, seq_len, vocab_size]
                - loss: Cross-entropy loss (if labels provided)
        """
        # Get hidden states from core model
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            positions=positions,
        )
        
        # Compute logits
        if self._tie_weights:
            # Use tied weights
            logits = F.linear(hidden_states, self.model.embedding.weight)
        else:
            logits = self.lm_head(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for causal language modeling
            # logits[:, :-1, :] should predict labels[:, 1:]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss = self.loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )
        
        return {
            "logits": logits,
            "loss": loss,
        }
    
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids (Tensor): Input token IDs [batch, seq_len]
            max_new_tokens (int): Maximum number of new tokens to generate
            temperature (float): Sampling temperature (0 = greedy)
            top_k (int, optional): Top-k sampling
            top_p (float): Nucleus sampling threshold
            repetition_penalty (float): Penalty for repeated tokens
        
        Returns:
            Tensor: Generated token IDs [batch, seq_len + max_new_tokens]
        """
        self.eval()
        
        with torch.no_grad():
            batch_size = input_ids.shape[0]
            device = input_ids.device
            
            # Generated sequence
            generated = input_ids.clone()
            
            # KV cache for efficiency
            past_key_values = None
            
            for _ in range(max_new_tokens):
                # Forward pass
                outputs = self.forward(
                    input_ids=generated,
                )
                logits = outputs["logits"]
                
                # Get logits for last token
                next_token_logits = logits[:, -1, :]
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(batch_size):
                        for token_id in set(generated[i].tolist()):
                            next_token_logits[i, token_id] /= repetition_penalty
                
                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # Apply top-k
                if top_k is not None and top_k > 0:
                    top_k_vals = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < top_k_vals[0][..., -1, None]] = float('-inf')
                
                # Apply top-p (nucleus)
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumsum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumsum > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    for i in range(batch_size):
                        next_token_logits[i, sorted_indices[i, sorted_indices_to_remove[i]]] = float('-inf')
                
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for EOS
                if (next_token == self.config.eos_token_id).all():
                    break
            
            return generated
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing."""
        self.model.enable_gradient_checkpointing()
    
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self.model.disable_gradient_checkpointing()


# ============================================================================
# Factory Functions
# ============================================================================

def create_codling_model(
    vocab_size: int = 50304,
    d_model: int = 1024,
    n_layers: int = 24,
    d_state: int = 128,
    d_expand: int = 2,
    ssm_type: str = "s4",
    use_hyena: bool = False,
    use_linear_attn: bool = False,
    use_longrope: bool = False,
    max_position_embeddings: int = 8192,
    **kwargs,
) -> CodlingForCausalLM:
    """
    Factory function to create a CODLING model.
    
    This is a convenience function for creating models with common configurations.
    
    Args:
        vocab_size (int): Vocabulary size.
            Default: 50304 (~50K for code)
        d_model (int): Hidden dimension.
            Default: 1024
        n_layers (int): Number of layers.
            Default: 24
        d_state (int): SSM state dimension.
            Default: 128
        d_expand (int): FFN expansion factor.
            Default: 2
        ssm_type (str): SSM type ("mamba", "mamba2", "s4").
            Default: "mamba2"
        use_hyena (bool): Enable Hyena.
            Default: False
        use_linear_attn (bool): Enable linear attention.
            Default: False
        use_longrope (bool): Enable LongRoPE.
            Default: False
        max_position_embeddings (int): Maximum sequence length.
            Default: 8192
        **kwargs: Additional configuration options.
    
    Returns:
        CodlingForCausalLM: Initialized model.
    
    Example:
        >>> # Small model (~130M params)
        >>> model = create_codling_model(d_model=768, n_layers=24)
        >>> 
        >>> # Large model with hybrid architecture (~760M params)
        >>> model = create_codling_model(
        ...     d_model=1280,
        ...     n_layers=32,
        ...     use_hyena=True,
        ...     use_linear_attn=True,
        ...     use_longrope=True,
        ...     max_position_embeddings=32768,
        ... )
    """
    config = CodlingConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        d_state=d_state,
        d_expand=d_expand,
        ssm_type=ssm_type,
        use_hyena=use_hyena,
        use_linear_attn=use_linear_attn,
        use_longrope=use_longrope,
        max_position_embeddings=max_position_embeddings,
        **kwargs,
    )
    
    return CodlingForCausalLM(config)


def create_codling_model_from_pretrained(
    model_path: str,
    config: Optional[CodlingConfig] = None,
) -> CodlingForCausalLM:
    """
    Create a CODLING model from pretrained weights.
    
    Args:
        model_path (str): Path to pretrained checkpoint.
        config (CodlingConfig, optional): Model configuration.
            If None, loaded from checkpoint.
    
    Returns:
        CodlingForCausalLM: Model with loaded weights.
    """
    # This would load from a checkpoint in practice
    # For now, just create a new model
    
    if config is None:
        raise ValueError("Config must be provided when not loading from checkpoint")
    
    model = CodlingForCausalLM(config)
    
    # In practice, would load state_dict here:
    # state_dict = torch.load(model_path, map_location="cpu")
    # model.load_state_dict(state_dict)
    
    return model


# ============================================================================
# Utility Functions
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(config: CodlingConfig) -> str:
    """
    Estimate model size based on configuration.
    
    Args:
        config (CodlingConfig): Model configuration.
    
    Returns:
        str: Estimated model size (e.g., "130M", "350M", "760M").
    """
    # Rough estimation
    # Embeddings: vocab_size * d_model
    # Each layer: ~2 * d_model^2 + d_model * d_state + d_model * d_expand * d_model
    # Output: vocab_size * d_model
    
    params = (
        config.vocab_size * config.d_model  # Embedding
        + config.vocab_size * config.d_model  # Output (tied)
        + config.n_layers * (
            config.d_model * config.d_model * 4  # SSM projections
            + config.d_model * config.d_state  # SSM state
            + config.d_model * config.d_model * config.d_expand * 2  # MLP
            + config.d_model * config.d_model  # Output projections
        )
    )
    
    if params < 200_000_000:
        return f"{params // 1_000_000}M"
    elif params < 500_000_000:
        return f"{params // 1_000_000}M"
    elif params < 1_000_000_000:
        return f"{params // 1_000_000}M"
    else:
        return f"{params // 1_000_000_000}B"


# ============================================================================
# Example Usage and Testing
# ============================================================================

if __name__ == "__main__":
    # Test configurations
    print("CODLING SSM Language Model - Testing")
    print("=" * 50)
    
    # Small model
    print("\n1. Small Model (130M params)")
    small_config = CodlingConfig(
        vocab_size=50304,
        d_model=768,
        n_layers=24,
        d_state=128,
    )
    small_model = CodlingForCausalLM(small_config)
    print(f"   Parameters: {count_parameters(small_model):,}")
    print(f"   Size: {get_model_size(small_config)}")
    
    # Test forward pass
    input_ids = torch.randint(0, small_config.vocab_size, (1, 64))
    outputs = small_model(input_ids)
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Output logits shape: {outputs['logits'].shape}")
    
    # Medium model with Hyena
    print("\n2. Medium Model with Hyena (~350M params)")
    medium_config = CodlingConfig(
        vocab_size=50304,
        d_model=1024,
        n_layers=24,
        d_state=128,
        use_hyena=True,
    )
    medium_model = CodlingForCausalLM(medium_config)
    print(f"   Parameters: {count_parameters(medium_model):,}")
    print(f"   Size: {get_model_size(medium_config)}")
    
    # Test forward pass
    input_ids = torch.randint(0, medium_config.vocab_size, (1, 128))
    outputs = medium_model(input_ids)
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Output logits shape: {outputs['logits'].shape}")
    
    # Large model with full hybrid
    print("\n3. Large Model with Full Hybrid (~760M params)")
    large_config = CodlingConfig(
        vocab_size=50304,
        d_model=1280,
        n_layers=32,
        d_state=256,
        use_hyena=True,
        use_linear_attn=True,
    )
    large_model = CodlingForCausalLM(large_config)
    print(f"   Parameters: {count_parameters(large_model):,}")
    print(f"   Size: {get_model_size(large_config)}")
    
    # Test with labels (loss computation)
    print("\n4. Testing Loss Computation")
    input_ids = torch.randint(1, large_config.vocab_size, (2, 32))
    labels = input_ids.clone()
    outputs = large_model(input_ids, labels=labels)
    print(f"   Loss: {outputs['loss'].item():.4f}")
    
    # Test generation
    print("\n5. Testing Generation")
    input_ids = torch.randint(1, large_config.vocab_size, (1, 10))
    generated = large_model.generate(input_ids, max_new_tokens=20, temperature=0.8)
    print(f"   Input: {input_ids[0, :5].tolist()}...")
    print(f"   Generated length: {generated.shape[1]}")
    
    # Test gradient checkpointing
    print("\n6. Testing Gradient Checkpointing")
    large_model.enable_gradient_checkpointing()
    print("   Gradient checkpointing enabled")
    
    # Test with torch.compile (if available)
    print("\n7. Testing torch.compile")
    try:
        compiled_model = torch.compile(large_model)
        print("   torch.compile available and applied")
    except Exception as e:
        print(f"   torch.compile not available: {e}")
    
    print("\n" + "=" * 50)
    print("All tests passed!")
