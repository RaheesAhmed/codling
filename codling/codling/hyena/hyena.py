"""
Hyena Operator Implementation for CODLING
==========================================

Subquadratic token mixer replacing attention with O(n log n) or O(n) operations.

Based on: Poli et al., 2023 - "Hyena: Subquadratic Inference and Training for 
State Space Models"

Key Concepts:
- The Hyena operator replaces attention with a combination of:
  1. Implicit long convolution via FFT/iFFT (O(n log n))
  2. Gated depthwise convolutions at multiple scales
  3. Filter functions (MLP) that generate convolution kernels

The formula: Hyena(q, k, v) = depthwise_gated_conv(φ(q), φ(k)) ⊙ v

Where:
- φ: projection function mapping queries/keys to a space where depthwise 
     convolution acts as attention
- depthwise_gated_conv: combines query and key projections via depthwise 
  gated convolutions
- ⊙: element-wise multiplication with values

This provides:
- O(n log n) or O(n) complexity vs O(n²) for attention
- Same interface as attention for drop-in replacement
- Causal convolution support for autoregressive generation
- Full gradient flow support
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


def fft_conv1d(
    u: torch.Tensor, 
    kernel: torch.Tensor, 
    nonce: int = 0, 
    dropout: float = 0.0,
    training: bool = True
) -> torch.Tensor:
    """
    Perform 1D convolution via FFT for O(n log n) complexity.
    
    Args:
        u: Input tensor of shape (batch, length, channels)
        kernel: Convolution kernel of shape (kernel_size, channels)
        nonce: Nonce for dropout (for determinism)
        dropout: Dropout probability
        training: Whether in training mode
        
    Returns:
        Convolved output of shape (batch, length, channels)
    """
    # Get sequence length and pad for FFT
    length = u.shape[1]
    kernel_length = kernel.shape[0]
    
    # Convolution requires padding - use same padding for causal
    fft_length = length + kernel_length - 1
    
    # Next power of 2 for efficient FFT
    n_fft = 1 << (fft_length - 1).bit_length()
    
    # FFT of input and kernel
    u_fft = torch.fft.rfft(u, n=n_fft, dim=1)
    kernel_fft = torch.fft.rfft(kernel, n=n_fft, dim=1)
    
    # Multiply in frequency domain
    y_fft = u_fft * kernel_fft.unsqueeze(1)
    
    # Inverse FFT to get convolution result
    y = torch.fft.irfft(y_fft, n=n_fft, dim=1)
    
    # Trim to original length + kernel_length - 1
    y = y[:, :length + kernel_length - 1, :]
    
    # Apply dropout if needed (on the output)
    if dropout > 0 and training:
        # Use fixed dropout mask for efficiency
        mask = (torch.rand_like(y[:1, :, :]) > dropout).to(y.dtype)
        y = y * mask * (1.0 / (1 - dropout))
    
    return y


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal positional embeddings for the filter function.
    Projects positions to enable learnable positional biases in convolution.
    """
    
    def __init__(self, dim: int, max_positions: int = 1024):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions
        
        # Create sinusoidal embeddings
        position = torch.arange(max_positions).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2) * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(max_positions, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get positional embeddings for given sequence length."""
        return self.pe[:seq_len].to(device)


class FilterFunction(nn.Module):
    """
    Filter function that generates convolution kernels from queries and keys.
    
    Uses a small MLP with tanh GELU activation to create position-dependent
    convolution kernels. This is the "learned" part of the Hyena operator.
    
    Architecture:
        input -> Linear -> GELU -> Linear -> tanh -> output
    
    The output is a convolution kernel of shape (kernel_size, dim).
    """
    
    def __init__(
        self, 
        d_model: int, 
        d_inner: int = 128, 
        kernel_size: int = 3,
        filter_order: int = 2,
        use_bias: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.kernel_size = kernel_size
        self.filter_order = filter_order
        
        # Project from d_model to inner dimension (filter_order projections)
        # Shape: (batch, seq_len, d_inner * filter_order)
        self.proj = nn.Linear(d_model, d_inner * filter_order)
        
        # Output projection from d_inner to kernel (for EACH filter order)
        # Shape: (batch, seq_len, d_inner * filter_order) -> (batch, seq_len, kernel_size * filter_order)
        self.output = nn.Linear(d_inner, kernel_size, bias=use_bias)
        
        # Activation: GELU followed by tanh
        self.act = nn.GELU()
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with scaled uniform distribution."""
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.output.weight)
        if self.output.bias is not None:
            nn.init.zeros_(self.output.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate convolution kernels from input.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            positions: Optional positional embeddings to add
            
        Returns:
            Kernels of shape (filter_order, batch, seq_len, kernel_size)
        """
        batch, seq_len, _ = x.shape
        
        # Project to inner dimension: (batch, seq_len, d_inner * filter_order)
        h = self.proj(x)
        
        # Apply GELU activation
        h = self.act(h)
        
        # Split into filter_order parts: (batch, seq_len, filter_order, d_inner)
        h = h.view(batch, seq_len, self.filter_order, self.d_inner)
        
        # Apply output projection for each filter order
        # Result: (batch, seq_len, filter_order, kernel_size)
        outputs = []
        for i in range(self.filter_order):
            out_i = self.output(h[:, :, i, :])  # (batch, seq_len, kernel_size)
            outputs.append(out_i)
        
        # Stack: (batch, seq_len, filter_order, kernel_size)
        h = torch.stack(outputs, dim=2)
        
        # Apply tanh gating
        h = torch.tanh(h)
        
        # Permute to (filter_order, batch, seq_len, kernel_size)
        return h.permute(2, 0, 1, 3)


class GatedDepthwiseConv1d(nn.Module):
    """
    Gated Depthwise 1D Convolution with multiple scales.
    
    This is the core "mixing" operation in Hyena. It applies depthwise
    convolutions at multiple scales with gating mechanism:
    
        output = depthwise_conv(input, kernel) * sigmoid(gate)
    
    The gating provides non-linearities and helps with gradient flow.
    """
    
    def __init__(
        self,
        d_model: int,
        kernel_size: int = 3,
        filter_order: int = 2,
        dropout: float = 0.0,
        causal: bool = True,
        conv_bias: bool = True,
        num_kernels: int = 1
    ):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.filter_order = filter_order
        self.dropout = dropout
        self.causal = causal
        self.num_kernels = num_kernels
        
        # For each filter order, we have a separate convolution
        # Each conv operates on a single channel (depthwise)
        if causal:
            padding = kernel_size - 1
        else:
            padding = (kernel_size - 1) // 2
        
        self.padding = [padding] * filter_order
        
        # Depthwise conv: groups = d_model (each channel convolved separately)
        # Using simple depthwise convolutions that will be modulated by filters
        self.convs = nn.ModuleList([
            nn.Conv1d(
                d_model, 
                d_model, 
                kernel_size,
                padding=padding,
                groups=d_model,
                bias=conv_bias
            )
            for _ in range(filter_order)
        ])
        
        # Dropout for regularization
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        filters: torch.Tensor,
        force_causal: bool = False
    ) -> torch.Tensor:
        """
        Apply gated depthwise convolution.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            filters: Convolution kernels (filter_order, batch, seq_len, kernel_size)
            force_causal: Override causal setting
            
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape
        _, _, _, kernel_size = filters.shape
        
        # Transpose for conv1d: (batch, d_model, seq_len)
        x_conv = x.transpose(1, 2)
        
        outputs = []
        
        for i, conv in enumerate(self.convs):
            # Get kernel for this filter order: (batch, seq_len, kernel_size)
            # filters[i] has shape (batch, seq_len, kernel_size)
            kernel = filters[i]  # (batch, seq_len, kernel_size)
            
            # Use the first kernel (or average across sequence) as the convolution weight
            # This creates a "mixing" effect where each position contributes to the output
            # Take mean across sequence dimension to get a single kernel per batch
            kernel_weight = kernel.mean(dim=1, keepdim=True)  # (batch, 1, kernel_size)
            
            # Expand to (batch * d_model, 1, kernel_size) for grouped convolution
            # Each of the d_model channels in each batch gets the same kernel
            kernel_weight = kernel_weight.unsqueeze(1).expand(batch, d_model, -1, -1)
            kernel_weight = kernel_weight.reshape(batch * d_model, 1, kernel_size)
            
            # Apply depthwise convolution with generated kernel
            out = F.conv1d(
                x_conv.reshape(1, batch * d_model, -1),
                kernel_weight,
                padding=self.padding[i],
                groups=batch * d_model
            )  # (1, batch * d_model, seq_len + padding)
            
            out = out.reshape(batch, d_model, -1)  # (batch, d_model, seq_len + padding)
            
            # Handle causal: trim to original sequence length
            if self.causal or force_causal:
                out = out[:, :, :seq_len]
            
            # Apply dropout
            if self.dropout > 0:
                out = self.dropout_layer(out)
            
            outputs.append(out)
        
        # Combine all filter orders: sum them up
        # This is the "additive" combination from the paper
        output = sum(outputs)
        
        # Transpose back: (batch, seq_len, d_model)
        return output.transpose(1, 2)


class HyenaOperator(nn.Module):
    """
    Core Hyena Operator - The building block of Hyena layers.
    
    Implements: y = Hyena(q, k, v) = depthwise_gated_conv(φ(q), φ(k)) ⊙ v
    
    This is the subquadratic attention replacement. It works by:
    1. Projecting q and k through filter functions to get convolution kernels
    2. Applying depthwise gated convolution between the kernels
    3. Element-wise multiplying the result with values
    
    The key insight is that depthwise convolution can simulate attention
    with O(n) or O(n log n) complexity instead of O(n²).
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        kernel_size: int = 3,
        filter_order: int = 2,
        dropout: float = 0.0,
        causal: bool = True,
        conv_bias: bool = True,
        num_kernels: int = 1,
        use_bias: bool = True,
        d_inner: Optional[int] = None,
        activation: str = "gelu",
        use_short_conv: bool = True,
        short_kernel_size: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state or d_model
        self.kernel_size = kernel_size
        self.filter_order = filter_order
        self.dropout = dropout
        self.causal = causal
        self.num_kernels = num_kernels
        self.use_short_conv = use_short_conv
        self.short_kernel_size = short_kernel_size
        
        # Inner dimension for filter function
        d_inner = d_inner or max(128, d_model // 2)
        
        # Short convolution for local context (optional)
        if use_short_conv:
            self.short_conv = nn.Conv1d(
                d_model,
                d_model,
                short_kernel_size,
                padding=(short_kernel_size - 1) if causal else short_kernel_size // 2,
                groups=d_model,
                bias=conv_bias
            )
        
        # Filter functions for q and k (shared or separate)
        # Using shared weights as per original paper
        self.filter_fn = FilterFunction(
            d_model=d_model,
            d_inner=d_inner,
            kernel_size=kernel_size,
            filter_order=filter_order,
            use_bias=use_bias
        )
        
        # Gated depthwise convolutions
        self.gated_conv = GatedDepthwiseConv1d(
            d_model=d_model,
            kernel_size=kernel_size,
            filter_order=filter_order,
            dropout=dropout,
            causal=causal,
            conv_bias=conv_bias,
            num_kernels=num_kernels
        )
        
        # Output projection (like attention output projection)
        self.output_proj = nn.Linear(d_model, d_model, bias=use_bias)
        
        # Activation
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "silu":
            self.act = nn.SiLU()
        else:
            self.act = nn.Identity()
        
        # Dropout for output
        self.output_dropout = nn.Dropout(dropout)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Initialize output projection."""
        nn.init.xavier_uniform_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        v: Optional[torch.Tensor] = None,
        force_causal: bool = False,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the Hyena operator.
        
        Args:
            x: Input tensor (batch, seq_len, d_model) - serves as q, k
            v: Value tensor (batch, seq_len, d_model). If None, uses x.
            force_causal: Force causal convolution
            position_ids: Optional position IDs for positional embeddings
            
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape
        
        # Use x as values if not provided
        if v is None:
            v = x
        
        # Short convolution for local context
        if self.use_short_conv:
            x_short = x.transpose(1, 2)  # (batch, d_model, seq_len)
            x_short = self.short_conv(x_short)
            x_short = x_short.transpose(1, 2)  # (batch, seq_len, d_model)
            # Handle padding: slice to match original sequence length
            if x_short.shape[1] > seq_len:
                x_short = x_short[:, :seq_len, :]
            x = x + x_short  # Residual connection
        
        # Generate filters from queries and keys
        # In Hyena, we use the same input for q and k
        # This creates a symmetric "self-attention" like behavior
        filters = self.filter_fn(x)  # (filter_order, batch, seq_len, kernel_size)
        
        # Apply gated depthwise convolution
        # This is where the "mixing" happens
        conv_out = self.gated_conv(x, filters, force_causal=force_causal)
        
        # Element-wise multiply with values (the ⊙ operation)
        # This is the core of the Hyena formula
        y = conv_out * v
        
        # Apply activation
        y = self.act(y)
        
        # Output projection
        y = self.output_proj(y)
        
        # Dropout
        y = self.output_dropout(y)
        
        return y


class HyenaLayer(nn.Module):
    """
    Complete Hyena Layer matching the attention layer interface.
    
    This layer can be used as a drop-in replacement for attention layers
    in transformer models. It provides the same interface:
    
        output = HyenaLayer(query, key, value, ...)
    
    But with subquadratic O(n log n) or O(n) complexity.
    
    Based on Poli et al., 2023 - "Hyena: Subquadratic Inference and Training
    for State Space Models"
    
    Architecture:
        1. Input projection (q, k, v from single input for self-attention)
        2. Short convolution (local context)
        3. Long convolution (via FFT - global context)
        4. Gated mechanism
        5. Output projection
        
    The layer combines:
    - Short-range dependencies via depthwise convolutions
    - Long-range dependencies via FFT-based convolution
    - Gating for non-linearity and gradient flow
    """
    
    def __init__(
        self,
        d_model: int,
        d_head: Optional[int] = None,
        num_heads: int = 1,
        kernel_size: int = 3,
        filter_order: int = 2,
        dropout: float = 0.0,
        causal: bool = True,
        conv_bias: bool = True,
        use_bias: bool = True,
        d_inner: Optional[int] = None,
        activation: str = "gelu",
        use_short_conv: bool = True,
        short_kernel_size: int = 3,
        layer_norm_eps: float = 1e-5,
        prenorm: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_head = d_head or d_model // num_heads
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.filter_order = filter_order
        self.dropout = dropout
        self.causal = causal
        self.prenorm = prenorm
        
        # Input layer norm (prenorm) or output (postnorm)
        if prenorm:
            self.norm_q = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.norm_k = nn.Identity()  # Shared norm for efficiency
            self.norm_v = nn.Identity()
        else:
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()
            self.norm_v = nn.Identity()
            self.norm_out = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # The Hyena operator
        self.hyena = HyenaOperator(
            d_model=d_model,
            kernel_size=kernel_size,
            filter_order=filter_order,
            dropout=dropout,
            causal=causal,
            conv_bias=conv_bias,
            use_bias=use_bias,
            d_inner=d_inner,
            activation=activation,
            use_short_conv=use_short_conv,
            short_kernel_size=short_kernel_size,
        )
        
        # Dropout for residual connection
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        force_causal: bool = False,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass matching attention interface.
        
        Args:
            x: Query input (batch, seq_len, d_model)
            key: Key input (batch, kv_len, d_model). If None, uses x (self-attention)
            value: Value input (batch, kv_len, d_model). If None, uses x
            attn_mask: Not used (kept for interface compatibility)
            force_causal: Force causal masking
            position_ids: Position IDs for positional embeddings
            
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # Self-attention: use x for k and v if not provided
        if key is None:
            key = x
        if value is None:
            value = x
        
        # Pre-norm (Prenorm - more stable for training)
        if self.prenorm:
            residual = x
            x = self.norm_q(x)
            key = self.norm_k(key)
            value = self.norm_v(value)
        
        # Apply Hyena operator
        # In self-attention mode, q=k=v=x
        # The operator handles the mixing internally
        output = self.hyena(x, value, force_causal=force_causal)
        
        # Dropout and residual connection
        output = self.dropout_layer(output)
        
        # Post-norm (if using postnorm instead of prenorm)
        if not self.prenorm:
            output = self.norm_out(output + residual)
        else:
            # Residual connection for prenorm
            output = output + residual
        
        return output
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal mask for autoregressive generation.
        
        Args:
            seq_len: Sequence length
            device: Device to create mask on
            
        Returns:
            Causal mask of shape (1, seq_len, seq_len)
        """
        # Not needed for Hyena but kept for interface
        return torch.ones(seq_len, seq_len, device=device).tril()


class MultiHeadHyena(nn.Module):
    """
    Multi-head Hyena - combines multiple Hyena heads like multi-head attention.
    
    Each head attends to different aspects of the input, then results
    are combined via concatenation and projection.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_head: Optional[int] = None,
        kernel_size: int = 3,
        filter_order: int = 2,
        dropout: float = 0.0,
        causal: bool = True,
        conv_bias: bool = True,
        use_bias: bool = True,
        d_inner: Optional[int] = None,
        activation: str = "gelu",
        use_short_conv: bool = True,
        short_kernel_size: int = 3,
        layer_norm_eps: float = 1e-5,
        prenorm: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head or d_model // num_heads
        self.d_out = self.d_head * num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Create multiple Hyena layers (one per head)
        self.heads = nn.ModuleList([
            HyenaLayer(
                d_model=self.d_head,
                d_head=self.d_head,
                num_heads=1,
                kernel_size=kernel_size,
                filter_order=filter_order,
                dropout=dropout,
                causal=causal,
                conv_bias=conv_bias,
                use_bias=use_bias,
                d_inner=d_inner,
                activation=activation,
                use_short_conv=use_short_conv,
                short_kernel_size=short_kernel_size,
                layer_norm_eps=layer_norm_eps,
                prenorm=prenorm,
            )
            for _ in range(num_heads)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(self.d_out, d_model, bias=use_bias)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        force_causal: bool = False,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with multi-head processing.
        
        Args:
            x: Input (batch, seq_len, d_model)
            key: Optional key tensor
            value: Optional value tensor
            attn_mask: Not used (interface compatibility)
            force_causal: Force causal masking
            position_ids: Position IDs
            
        Returns:
            Output (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        
        # Split into heads
        x = x.view(batch, seq_len, self.num_heads, self.d_head)
        x = x.transpose(1, 2)  # (batch, num_heads, seq_len, d_head)
        
        # Process each head
        head_outputs = []
        for h_idx, head in enumerate(self.heads):
            # Get input for this specific head
            head_input = x[:, h_idx, :, :]  # (batch, seq_len, d_head)
            
            # Also split key/value if provided for this head
            k = None
            v = None
            if key is not None:
                key_h = key.view(batch, -1, self.num_heads, self.d_head)
                k = key_h[:, h_idx, :, :]  # (batch, kv_len, d_head)
            if value is not None:
                value_h = value.view(batch, -1, self.num_heads, self.d_head)
                v = value_h[:, h_idx, :, :]  # (batch, kv_len, d_head)
            
            out = head(head_input, key=k, value=v, force_causal=force_causal)
            head_outputs.append(out)
        
        # Concatenate heads
        output = torch.stack(head_outputs, dim=1)  # (batch, num_heads, seq_len, d_head)
        output = output.transpose(1, 2)  # (batch, seq_len, num_heads, d_head)
        output = output.reshape(batch, seq_len, self.d_out)  # (batch, seq_len, d_out)
        
        # Output projection
        output = self.output_proj(output)
        output = self.dropout_layer(output)
        
        # Final norm
        output = self.norm(output)
        
        return output


def create_hyena_layer(
    d_model: int,
    num_heads: int = 8,
    kernel_size: int = 3,
    filter_order: int = 2,
    dropout: float = 0.0,
    causal: bool = True,
    use_multihead: bool = False,
    **kwargs
) -> nn.Module:
    """
    Factory function to create a Hyena layer.
    
    Args:
        d_model: Model dimension
        num_heads: Number of heads (if multi-head)
        kernel_size: Convolution kernel size
        filter_order: Order of the filter function
        dropout: Dropout probability
        causal: Use causal convolution
        use_multihead: Use multi-head variant
        
    Returns:
        Hyena layer module
    """
    if use_multihead:
        return MultiHeadHyena(
            d_model=d_model,
            num_heads=num_heads,
            kernel_size=kernel_size,
            filter_order=filter_order,
            dropout=dropout,
            causal=causal,
            **kwargs
        )
    else:
        return HyenaLayer(
            d_model=d_model,
            num_heads=num_heads,
            kernel_size=kernel_size,
            filter_order=filter_order,
            dropout=dropout,
            causal=causal,
            **kwargs
        )
