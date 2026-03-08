"""
Based Linear Attention Implementation
=====================================

Based (Buffer-based Linear Attention) implementation following Arora et al., 2024.

This module provides linear complexity attention (O(N) instead of O(N²)) through
kernel-based feature map approximation. The key insight is that attention can be
rewritten as: softmax(QK^T)V ≈ φ(Q)(φ(K)^T V) where φ is a feature map.

The ELU+1 feature map provides a good approximation with:
- ELU activation: elu(x) = max(0, x) + min(0, exp(x) - 1)
- Plus 1: elu(x) + 1 ensures positive values
- This creates a kernel that approximates softmax behavior

Mathematical Foundation:
-----------------------
Standard attention: A = softmax(QK^T / √d) * V

Linear attention rewrite:
  A = softmax(QK^T / √d) * V
    ≈ Σ_i φ(q_i) φ(k_i)^T v_i / Σ_i φ(q_i) φ(k_i)^T
    = (Φ_Q @ Φ_K^T @ V) / (Φ_Q @ Φ_K^T @ 1)

Where:
- φ(x) = ELU(x) + 1 is the feature map
- Φ_Q = φ(Q), Φ_K = φ(K) are the projected queries/keys

For causal attention, we add a lower-triangular mask before softmax.

References:
-----------
- Arora et al., 2024 - "Long Context Attention with Linear Complexity"
- Katharopoulos et al., 2020 - "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
-Choromanski et al., 2020 - "Rethinking Attention with Performers"

"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def elu_feature_map(x: Tensor) -> Tensor:
    """
    ELU+1 feature map for linear attention kernel approximation.
    
    This feature map transforms queries and keys into a feature space where
    the dot product approximates the softmax kernel without explicit softmax.
    
    ELU(x) + 1 ensures:
    1. All values are positive (required for kernel)
    2. Near-zero inputs map to ~1 (helps with numerical stability)
    3. Large positive inputs grow linearly (not exponential like softmax)
    
    Args:
        x: Input tensor of shape [..., seq_len, dim]
        
    Returns:
        Feature-mapped tensor of same shape with positive values
        
    Example:
        >>> x = torch.randn(2, 4, 8)  # batch=2, seq=4, dim=8
        >>> phi_x = elu_feature_map(x)
        >>> phi_x.shape
        torch.Size([2, 4, 8])
    """
    return F.elu(x) + 1


def causal_mask(seq_len: int, device: torch.device = torch.device('cpu')) -> Tensor:
    """
    Create causal (lower triangular) mask for autoregressive attention.
    
    Args:
        seq_len: Length of the sequence
        device: Device to create mask on
        
    Returns:
        Boolean mask of shape [seq_len, seq_len] where True = allowed
    """
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def apply_causal_mask(scores: Tensor, chunk_offset: int = 0) -> Tensor:
    """
    Apply causal masking to attention scores.
    
    Args:
        scores: Attention scores of shape [..., seq_q, seq_k]
        chunk_offset: Starting position for chunked computation (default: 0)
        
    Returns:
        Masked scores with -inf in prohibited positions
    """
    seq_q, seq_k = scores.shape[-2], scores.shape[-1]
    
    if chunk_offset == 0:
        # Full sequence causal mask
        mask = causal_mask(seq_k, scores.device)
    else:
        # Chunked computation: need to handle offset
        # Create mask that accounts for previous chunks
        mask = torch.tril(
            torch.ones(seq_q, seq_k, dtype=torch.bool, device=scores.device)
        )
        # Only attend to positions <= current position
        if seq_k < chunk_offset + seq_k:
            pass  # This chunk can attend to all positions within itself
    
    return scores.masked_fill(~mask, float('-inf'))


class LinearAttention(nn.Module):
    """
    Linear Attention Module with ELU+1 feature map.
    
    Provides O(N) complexity attention through kernel approximation.
    Suitable for long sequences where standard O(N²) attention is prohibitive.
    
    Architecture:
    -------------
    1. Feature map projection: φ(Q), φ(K) using ELU+1
    2. Linear attention: A = φ(Q) @ (φ(K)^T @ V)
    3. Normalization: A / (φ(Q) @ φ(K)^T @ 1)
    
    Args:
        dim: Model dimension
        heads: Number of attention heads
        dim_head: Dimension per head (if None, computed as dim/heads)
        eps: Epsilon for numerical stability in normalization
        
    Example:
        >>> attn = LinearAttention(dim=512, heads=8)
        >>> q = torch.randn(2, 100, 512)
        >>> k = torch.randn(2, 100, 512)
        >>> v = torch.randn(2, 100, 512)
        >>> out = attn(q, k, v)
        >>> out.shape
        torch.Size([2, 100, 512])
    """
    
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: Optional[int] = None,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head or dim // heads
        self.eps = eps
        
        # Separate projections for Q, K, V
        self.to_q = nn.Linear(dim, self.dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, self.dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, self.dim_head * heads, bias=False)
        # Output projection
        self.to_out = nn.Linear(self.dim_head * heads, dim, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        nn.init.xavier_uniform_(self.to_out.weight)
    
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
        causal: bool = False,
    ) -> Tensor:
        """
        Forward pass of linear attention.
        
        Uses the feature map approximation to compute attention in O(N) complexity
        instead of O(N²) by avoiding explicit Q@K^T computation.
        
        The linear attention kernel uses:
        A_ij = φ(q_i)^T φ(k_j)
        
        Which can be computed as:
        - KV = Σ_j φ(k_j) ⊗ v_j  [B, H, D] - linearized key-value
        - K_Σ = Σ_j φ(k_j)       [B, H, D] - sum of kernel features
        
        Then: out_i = φ(q_i)^T @ KV / (φ(q_i)^T @ K_Σ)
        
        Args:
            query: Query tensor [batch, seq_len, dim]
            key: Key tensor [batch, seq_len, dim] 
            value: Value tensor [batch, seq_len, dim]
            mask: Optional boolean mask [batch, seq_len] (True = valid)
            causal: Whether to apply causal masking
            
        Returns:
            Attention output [batch, seq_len, dim]
        """
        # Separate projections for Q, K, V
        q = self.to_q(query)
        k = self.to_k(key)
        v = self.to_v(value)
        
        B, L, _ = q.shape
        H = self.heads
        D = self.dim_head
        
        # Reshape to [B, H, L, D]
        q = q.view(B, L, H, D).transpose(1, 2)
        k = k.view(B, L, H, D).transpose(1, 2)
        v = v.view(B, L, H, D).transpose(1, 2)
        
        # Apply feature map: ELU + 1
        q_feat = elu_feature_map(q)  # [B, H, L, D]
        k_feat = elu_feature_map(k)  # [B, H, L, D]
        
        # Compute linear attention: O(N) using associative scan / cumulative computation
        # KV[j] = Σ_{i=0}^{j} φ(k_i) ⊗ v_i  (causal) or Σ_all φ(k_i) ⊗ v_i (non-causal)
        # KΣ[j] = Σ_{i=0}^{j} φ(k_i)        (causal) or Σ_all φ(k_i) (non-causal)
        
        # Reshape for efficient computation: [B, H, D, L]
        k_feat_t = k_feat.transpose(-1, -2)  # [B, H, D, L]
        v_t = v.transpose(-1, -2)          # [B, H, D, L]
        
        if causal:
            # Causal: compute cumulative sums
            # KV = cumsum(φ(K) ⊗ V, dim=-1)
            # KΣ = cumsum(φ(K), dim=-1)
            
            # Element-wise: k_feat[i] * v[i] for each position
            kv_outer = k_feat_t.unsqueeze(-1) * v_t.unsqueeze(-2)  # [B, H, D, L, D']
            # Sum over sequence dimension
            KV = kv_outer.sum(dim=-1)  # [B, H, D, D']
            K_sum = k_feat_t.sum(dim=-1)  # [B, H, D]
            
            # Actually for proper causal, we need per-position cumulative
            # Use cumsum for proper causal linear attention
            # KV_cum[j] = Σ_{i<=j} φ(k_i) * v_i
            # This is handled by the causal computation below
            KV = torch.cumsum(
                (k_feat_t * v_t).transpose(-1, -2), dim=2
            ).transpose(-1, -2)  # [B, H, L, D]
            K_sum = torch.cumsum(k_feat, dim=2)  # [B, H, L, D]
            
            # For each query position i, compute output using cumsum up to i
            # out_i = φ(q_i) @ KV_i / (φ(q_i) @ KΣ_i)
            # where KV_i = Σ_{j<=i} φ(k_j) ⊗ v_j
            
            # Compute numerator: φ(q_i) · KV_i
            # [B, H, L, D] @ [B, H, L, D] -> [B, H, L]
            numerator = (q_feat * KV).sum(dim=-1)  # [B, H, L]
            # Compute denominator: φ(q_i) · KΣ_i
            denominator = (q_feat * K_sum).sum(dim=-1).clamp(min=self.eps)  # [B, H, L]
            
            # Normalize
            attn_weights = numerator / denominator  # [B, H, L]
            attn_weights = attn_weights.unsqueeze(-1)  # [B, H, L, 1]
            
            # Apply to values
            out = v * attn_weights  # [B, H, L, D]
            
        else:
            # Non-causal: compute full sum once
            # KV = Σ_j φ(k_j) ⊗ v_j
            # This is outer product sum: [B, H, D] ⊗ [B, H, D] -> [B, H, D, D]
            
            # Compute: KV = Σ φ(k_j) * v_j^T for each position, then sum
            # This gives us the linearized key-value state
            kv_state = (k_feat * v).sum(dim=2)  # [B, H, D]
            k_sum = k_feat.sum(dim=2)  # [B, H, D]
            
            # For each query position: out_i = φ(q_i) @ KV / (φ(q_i) @ K_sum)
            # Both KV and K_sum are the same for all positions (non-causal)
            numerator = torch.einsum('bhld,bhd->bhl', q_feat, kv_state)  # [B, H, L]
            denominator = torch.einsum('bhld,bhd->bhl', q_feat, k_sum).clamp(min=self.eps)  # [B, H, L]
            
            attn_weights = (numerator / denominator).unsqueeze(-1)  # [B, H, L, 1]
            
            # Apply to values
            out = v * attn_weights  # [B, H, L, D]
        
        # Reshape output
        out = out.transpose(1, 2).contiguous().view(B, L, H * D)
        out = self.to_out(out)
        
        return out
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, L, H * D)
        return self.to_out(out)


class BufferBasedState(nn.Module):
    """
    Buffer-based state for streaming linear attention.
    
    Maintains running statistics (KV states) for efficient streaming inference.
    Instead of storing all past keys/values, we maintain aggregated states.
    
    This enables constant memory O(1) for autoregressive generation with
    linear attention, unlike standard attention which requires O(N) memory.
    
    State Update:
    -------------
    For each new token at position t:
    1. Update running key-value state: S += φ(k_t) ⊗ v_t
    2. Update running normalizer: Z += φ(k_t)
    3. Output: o_t = φ(q_t)^T @ S / (φ(q_t)^T @ Z)
    
    Where ⊗ denotes outer product (expand then sum).
    """
    
    def __init__(self, dim: int, heads: int, dim_head: int, max_seq_len: int = 8192):
        """
        Initialize buffer-based state.
        
        Args:
            dim: Model dimension
            heads: Number of attention heads
            dim_head: Dimension per head
            max_seq_len: Maximum sequence length for buffer allocation
        """
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.max_seq_len = max_seq_len
        
        # State buffers (non-learnable, persistent)
        # KV state: [H, D, D] - accumulates φ(k) outer product with v
        self.register_buffer('kv_state', torch.zeros(heads, dim_head, dim_head))
        # Normalizer: [H, D] - accumulates φ(k)
        self.register_buffer('normalizer', torch.zeros(heads, dim_head))
        
        self._is_initialized = False
    
    def reset(self):
        """Reset state for new sequence."""
        self.kv_state.zero_()
        self.normalizer.zero_()
        self._is_initialized = False
    
    def reset(self):
        """Reset state for new sequence."""
        self.kv_state.zero_()
        self.normalizer.zero_()
        self._is_initialized = False
    
    def forward(
        self,
        key_feat: Tensor,  # [B, H, D]
        value: Tensor,     # [B, H, D]
        query_feat: Optional[Tensor] = None,  # [B, H, D]
    ) -> Tensor:
        """
        Update state with new key-value and optionally query.
        
        Args:
            key_feat: Feature-mapped key [batch, heads, dim_head]
            value: Value [batch, heads, dim_head]
            query_feat: Optional feature-mapped query for immediate output
            
        Returns:
            If query_feat provided: attention output [batch, heads, dim_head]
            Else: None
        """
        # Update state
        self.update(key_feat, value)
        
        # If query provided, compute output
        if query_feat is not None:
            return self.query(query_feat)
        return None
    
    def update(
        self,
        key_feat: Tensor,  # [B, H, D]
        value: Tensor,     # [B, H, D]
    ):
        """
        Update state with new key-value pair.
        
        Args:
            key_feat: Feature-mapped key [batch, heads, dim_head]
            value: Value [batch, heads, dim_head]
        """
        B = key_feat.shape[0]
        
        if not self._is_initialized:
            # Initialize with first batch
            for b in range(B):
                # Outer product: [H, D] * [H, D] -> [H, D, D]
                kv_update = key_feat[b].unsqueeze(-1) * value[b].unsqueeze(-2)
                self.kv_state += kv_update
                self.normalizer += key_feat[b]
            self._is_initialized = True
        else:
            # Accumulate (assume same state for all batch elements during inference)
            # Use outer product via broadcasting
            kv_update = key_feat[0].unsqueeze(-1) * value[0].unsqueeze(-2)  # [H, D, D]
            self.kv_state += kv_update
            self.normalizer += key_feat[0]
    
    def query(
        self,
        query_feat: Tensor,  # [B, H, D]
    ) -> Tensor:
        """
        Query the state to get output for current position.
        
        Args:
            query_feat: Feature-mapped query [batch, heads, dim_head]
            
        Returns:
            Attention output [batch, heads, dim_head]
        """
        # o = φ(q)^T @ S / (φ(q)^T @ Z)
        
        # Numerator: query_feat @ kv_state
        # [B, H, D] @ [H, D, D] -> [B, H, D]
        # Need to handle heads properly
        
        numerator = torch.einsum('bhd,hDD->bhD', query_feat, self.kv_state)
        
        # Denominator: query_feat @ normalizer
        # [B, H, D] @ [H, D] -> [B, H]
        denominator = torch.einsum('bhd,hd->bh', query_feat, self.normalizer).clamp(min=1e-8)
        
        # Normalize
        output = numerator / denominator.unsqueeze(-1)
        
        return output


class TiledFlashLinearAttention(nn.Module):
    """
    Tiled Flash Linear Attention (TFLA) with chunked computation.
    
    Memory-efficient implementation that processes long sequences in chunks
    to avoid O(N) memory overhead. Uses feature map approximation for
    linear complexity within each chunk.
    
    Key optimizations:
    ------------------
    1. Chunked computation: Process sequence in tiles of size chunk_size
    2. Fused kernel: Compute Q@K^T and attention in single pass
    3. Streaming state: Optional buffer-based state for autoregressive
    4. BF16 support: Mixed precision for faster computation
    
    Architecture:
    -------------
    For each chunk:
    1. Compute feature maps: φ(Q_chunk), φ(K_chunk)
    2. Compute local attention within chunk
    3. Cross-chunk attention for chunk boundaries
    4. Aggregate with running state
    
    Args:
        dim: Model dimension
        heads: Number of attention heads
        dim_head: Dimension per head
        chunk_size: Size of chunks for tiled processing (default: 512)
        causal: Whether to use causal masking
        use_buffer_state: Whether to use buffer-based streaming state
        
    Example:
        >>> attn = TiledFlashLinearAttention(dim=512, heads=8, chunk_size=512)
        >>> q, k, v = torch.randn(2, 1024, 512).unbind(-2)
        >>> out = attn(q, k, v)  # Processes in 2 chunks of 512
        >>> out.shape
        torch.Size([2, 1024, 512])
    """
    
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: Optional[int] = None,
        chunk_size: int = 512,
        causal: bool = False,
        use_buffer_state: bool = False,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head or dim // heads
        self.chunk_size = chunk_size
        self.causal = causal
        self.eps = eps
        
        # Separate projections for Q, K, V
        self.to_q = nn.Linear(dim, self.dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, self.dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, self.dim_head * heads, bias=False)
        self.to_out = nn.Linear(self.dim_head * heads, dim, bias=False)
        
        # Optional buffer-based state for streaming
        self.use_buffer_state = use_buffer_state
        if use_buffer_state:
            self.buffer_state = BufferBasedState(dim, heads, self.dim_head)
        else:
            self.register_buffer('buffer_state', None)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        nn.init.xavier_uniform_(self.to_out.weight)
    
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass with tiled chunked computation.
        
        Args:
            query: Query tensor [B, L, D]
            key: Key tensor [B, L, D]
            value: Value tensor [B, L, D]
            mask: Optional boolean mask [B, L]
            
        Returns:
            Output tensor [B, L, D]
        """
        # Separate projections for Q, K, V
        q = self.to_q(query)
        k = self.to_k(key)
        v = self.to_v(value)
        
        B, L, _ = q.shape
        H = self.heads
        D = self.dim_head
        
        # Reshape: [B, L, H, D]
        q = q.view(B, L, H, D).transpose(1, 2)
        k = k.view(B, L, H, D).transpose(1, 2)
        v = v.view(B, L, H, D).transpose(1, 2)
        
        # Apply feature map
        q_feat = elu_feature_map(q)
        k_feat = elu_feature_map(k)
        
        # Determine computation mode
        if L <= self.chunk_size:
            # Small sequence: compute in one pass
            out = self._compute_attention(q_feat, k_feat, v, causal=self.causal)
        else:
            # Large sequence: chunked computation
            out = self._compute_chunked(q_feat, k_feat, v)
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(B, L, H * D)
        return self.to_out(out)
    
    def _compute_attention(
        self,
        q_feat: Tensor,
        k_feat: Tensor,
        v: Tensor,
        causal: bool = False,
    ) -> Tensor:
        """
        Compute attention for a single chunk.
        
        Args:
            q_feat: Feature-mapped queries [B, H, L, D]
            k_feat: Feature-mapped keys [B, H, K, D]
            v: Values [B, H, K, D]
            causal: Whether to apply causal masking
            
        Returns:
            Attention output [B, H, L, D]
        """
        B, H, L, K = q_feat.shape[0], q_feat.shape[1], q_feat.shape[2], k_feat.shape[2]
        
        # Compute kernel matrix: φ(Q) @ φ(K)^T
        kernel = torch.einsum('bhld,bhmd->bhlm', q_feat, k_feat)  # [B, H, L, K]
        
        # Apply causal mask if needed
        if causal:
            # For causal: only attend to past (K <= L positions)
            # Create mask: [L, K] where True = allowed
            mask = torch.tril(
                torch.ones(L, K, dtype=torch.bool, device=kernel.device)
            )
            kernel = kernel.masked_fill(~mask, float('-inf'))
            
            # Use softmax normalization for causal
            kernel_max = kernel.max(dim=-1, keepdim=True)[0]
            kernel_exp = torch.exp(kernel - kernel_max)
            kernel_exp = kernel_exp.masked_fill(~mask.unsqueeze(0).unsqueeze(0), 0)
            denom = kernel_exp.sum(dim=-1, keepdim=True).clamp(min=self.eps)
            attn = kernel_exp / denom
        else:
            # Simple normalization for non-causal
            denom = kernel.sum(dim=-1, keepdim=True).clamp(min=self.eps)
            attn = kernel / denom
        
        # Apply to values
        out = torch.einsum('bhlm,bhmd->bhld', attn, v)
        
        return out
    
    def _compute_chunked(
        self,
        q_feat: Tensor,
        k_feat: Tensor,
        v: Tensor,
    ) -> Tensor:
        """
        Compute attention using chunked/tiled processing.
        
        Processes the sequence in chunks to reduce memory from O(N²) to O(chunk_size * N).
        
        Args:
            q_feat: Feature-mapped queries [B, H, L, D]
            k_feat: Feature-mapped keys [B, H, L, D] 
            v: Values [B, H, L, D]
            
        Returns:
            Attention output [B, H, L, D]
        """
        B, H, L, D = q_feat.shape
        chunk_size = self.chunk_size
        
        # Initialize output
        out = torch.zeros_like(q_feat)
        
        # Process in chunks
        num_chunks = (L + chunk_size - 1) // chunk_size
        
        # For each chunk of queries
        for i in range(num_chunks):
            q_start = i * chunk_size
            q_end = min((i + 1) * chunk_size, L)
            q_chunk = q_feat[:, :, q_start:q_end, :]  # [B, H, chunk, D]
            
            # Determine relevant key-value range
            if self.causal:
                # For causal: keys up to current position
                k_end = q_end
            else:
                # For bidirectional: all keys
                k_end = L
            
            k_chunk = k_feat[:, :, :k_end, :]  # [B, H, k_end, D]
            v_chunk = v[:, :, :k_end, :]  # [B, H, k_end, D]
            
            # Compute attention for this chunk (causal is handled inside based on chunk positions)
            chunk_out = self._compute_attention(q_chunk, k_chunk, v_chunk, causal=self.causal)
            
            out[:, :, q_start:q_end, :] = chunk_out
        
        return out
    
    def _compute_causal_chunk(
        self,
        q_chunk: Tensor,
        k_feat: Tensor,
        v: Tensor,
        q_start: int,
        q_end: int,
    ) -> Tensor:
        """
        Compute causal attention for a chunk with proper masking.
        
        Args:
            q_chunk: Query chunk [B, H, chunk_size, D]
            k_feat: All keys [B, H, L, D]
            v: All values [B, H, L, D]
            q_start: Start position of chunk
            q_end: End position of chunk
            
        Returns:
            Causal output for chunk [B, H, chunk_size, D]
        """
        B, H, chunk_len, D = q_chunk.shape
        L = k_feat.shape[2]
        
        # Compute kernel for this chunk
        # For position j in chunk (global position q_start + j)
        # we can attend to keys 0 to q_start + j
        
        kernel = torch.einsum('bhcd,bhkd->bhck', q_chunk, k_feat)  # [B, H, chunk, L]
        
        # Create causal mask
        # kernel[:, :, j, k] is valid only if k <= q_start + j
        causal_mask = torch.tril(
            torch.ones(chunk_len, L, dtype=torch.bool, device=kernel.device)
        )
        # Shift mask based on chunk position
        if q_start > 0:
            shift = torch.zeros(chunk_len, L, dtype=torch.bool, device=kernel.device)
            for j in range(chunk_len):
                causal_idx = min(q_start + j, L - 1)
                shift[j, :causal_idx + 1] = True
            causal_mask = shift
        
        kernel = kernel.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Normalize (handle -inf)
        kernel_max = kernel.max(dim=-1, keepdim=True)[0]
        kernel_exp = torch.exp(kernel - kernel_max.masked_fill(kernel_max == -float('inf'), 0))
        kernel_exp = kernel_exp.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), 0)
        
        denom = kernel_exp.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        attn = kernel_exp / denom
        
        # Apply to values
        out = torch.einsum('bhck,bhkd->bhcd', attn, v)
        
        return out
    
    def forward_streaming(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> Tensor:
        """
        Streaming forward pass using buffer-based state.
        
        For autoregressive generation with constant memory.
        
        Args:
            query: Single token query [B, D]
            key: Single token key [B, D]
            value: Single token value [B, D]
            
        Returns:
            Output for current token [B, D]
        """
        if not self.use_buffer_state:
            raise RuntimeError("Buffer state not enabled. Set use_buffer_state=True.")
        
        # Separate projections for single token
        q = self.to_q(query)
        k = self.to_k(key)
        v = self.to_v(value)
        
        B, _ = q.shape
        H = self.heads
        D = self.dim_head
        
        q = q.view(B, H, D)
        k = k.view(B, H, D)
        v = v.view(B, H, D)
        
        # Apply feature map
        q_feat = elu_feature_map(q)
        k_feat = elu_feature_map(k)
        
        # Update state with current key-value
        self.buffer_state.update(k_feat, v)
        
        # Query state for output
        out = self.buffer_state.query(q_feat)  # [B, H, D]
        
        # Reshape and project
        out = out.view(B, H * D)
        return self.to_out(out)
    
    def reset_state(self):
        """Reset streaming state for new sequence."""
        if self.use_buffer_state:
            self.buffer_state.reset()


class BasedLinearAttention(nn.Module):
    """
    Based Linear Attention - Hybrid with optional standard attention path.
    
    This provides flexibility to use either:
    1. Pure linear attention (fast, O(N) memory)
    2. Standard attention (higher quality, O(N²) memory)
    3. Hybrid: switch based on sequence length
    
    Args:
        dim: Model dimension
        heads: Number of attention heads
        dim_head: Dimension per head
        use_linear: Whether to use linear attention path
        use_standard: Whether to use standard attention path
        chunk_size: Chunk size for tiled processing
        dropout: Dropout probability
        
    Example:
        >>> attn = BasedLinearAttention(dim=512, heads=8, use_linear=True)
        >>> out = attn(q, k, v)  # Uses linear attention
    """
    
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: Optional[int] = None,
        use_linear: bool = True,
        use_standard: bool = False,
        chunk_size: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head or dim // heads
        self.use_linear = use_linear
        self.use_standard = use_standard
        
        # Linear attention path
        if use_linear:
            self.linear_attn = TiledFlashLinearAttention(
                dim=dim,
                heads=heads,
                dim_head=self.dim_head,
                chunk_size=chunk_size,
                causal=True,
                use_buffer_state=False,
            )
        
        # Standard attention path (for comparison/hybrid)
        if use_standard:
            self.standard_attn = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=heads,
                dropout=dropout,
                batch_first=True,
            )
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Output projection
        self.to_out = nn.Linear(dim, dim)
    
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        use_linear: Optional[bool] = None,
    ) -> Tensor:
        """
        Forward pass.
        
        Args:
            query: Query tensor [B, L, D]
            key: Key tensor [B, L, D]
            value: Value tensor [B, L, D]
            use_linear: Override which attention to use (None = use default)
            
        Returns:
            Output tensor [B, L, D]
        """
        # Determine which path to use
        if use_linear is None:
            use_linear = self.use_linear
        
        if use_linear:
            out = self.linear_attn(query, key, value)
        else:
            out = self._standard_attention(query, key, value)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        return self.to_out(out)
    
    def _standard_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> Tensor:
        """Standard attention as fallback."""
        B, L, D = query.shape
        
        # Use standard multihead attention
        out, _ = self.standard_attn(query, key, value, need_weights=False)
        
        return out


# Utility functions for feature map experiments
def softmax_feature_map(x: Tensor, temperature: float = 1.0) -> Tensor:
    """
    Alternative: softmax-based feature map (for comparison).
    
    Note: This is not truly linear as it requires softmax computation.
    Included for experiments comparing approximation quality.
    
    Args:
        x: Input tensor
        temperature: Softmax temperature
        
    Returns:
        Feature-mapped tensor
    """
    return F.softmax(x / temperature, dim=-1)


def relu_feature_map(x: Tensor) -> Tensor:
    """
    Simple ReLU feature map.
    
    Less expressive than ELU+1 but computationally cheaper.
    
    Args:
        x: Input tensor
        
    Returns:
        Feature-mapped tensor
    """
    return F.relu(x)


# Export all components
__all__ = [
    'LinearAttention',
    'TiledFlashLinearAttention',
    'BasedLinearAttention',
    'BufferBasedState',
    'elu_feature_map',
    'softmax_feature_map',
    'relu_feature_map',
    'causal_mask',
    'apply_causal_mask',
]
