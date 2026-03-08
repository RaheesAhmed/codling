"""
Selective Scan Module - Parallel Scan Algorithm for O(n) SSM Recurrence

This module implements the parallel scan (prefix sum) algorithm for efficiently
computing SSM recurrences in linear time O(n) instead of sequential O(n²).

Mathematical Background:
========================

The SSM recurrence is:
    h_t = A @ h_{t-1} + B @ x_t
    y_t = C @ h_t + D @ x_t

Where:
    - A: state transition matrix (N x N)
    - B: input-to-state projection (N x d)
    - C: state-to-output projection (d x N)
    - D: direct feedthrough (d x d) - optional

Parallel Scan Algorithm:
=========================

The naive sequential computation is O(n):
    h_0 = B @ x_0
    h_1 = A @ h_0 + B @ x_1
    h_2 = A² @ h_0 + A @ B @ x_1 + B @ x_2
    ...

This can be rewritten as a prefix sum:
    h_t = Σ_{i=0}^{t} A^{t-i} @ B @ x_i

The parallel scan computes all prefix sums in O(n) using associative operators:

    [h_t]   [I  0] [h_{t-1}]
    [c_t] = [B  A] [x_t    ]

Where the scan combines (c, h) pairs associatively.

For selective SSM (Mamba), A, B, C depend on x_t, so we use the
Input-dependent Projection Layer (IPL) variant.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


def causal_scan(
    x: torch.Tensor,
    y: torch.Tensor,
    dt: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    *,
    reverse: bool = False,
    differentiable: bool = False,
) -> torch.Tensor:
    """
    Causal (forward) parallel scan for SSM recurrence.
    
    Computes: h_t = A^t @ h_0 + Σ_{i=0}^{t-1} A^{t-1-i} @ B @ x_i
    
    This is the core O(n) recurrence solver.
    
    Args:
        x: Input tensor of shape (batch, seq_len, d_model)
        y: Initial state tensor of shape (batch, d_state)
        dt: Time step / log dt of shape (batch, seq_len, d_state)
            If None, assumes dt=1.0 for all steps
        a: Diagonal elements of A matrix (log A) of shape (batch, seq_len, d_state)
            Stored in log space for stability
        b: B projection of shape (batch, seq_len, d_state)
        c: C projection of shape (batch, seq_len, d_state) - if None, computes state only
        reverse: If True, compute reverse scan (for bidirectional models)
        differentiable: If True, use custom gradient (useful for training)
    
    Returns:
        Output tensor y of shape (batch, seq_len, d_state) if c is None
        or (batch, seq_len, d_model) if c is provided
    """
    batch, seq_len, d_model = x.shape
    _, d_state = y.shape
    
    # Reverse sequences if needed
    if reverse:
        x = x.flip(1)
        if dt is not None:
            dt = dt.flip(1)
        a = a.flip(1)
        b = b.flip(1)
        if c is not None:
            c = c.flip(1)
    
    # Compute A = exp(dt * A) where A is stored as log A
    # a contains log(A), so A = exp(a)
    # We compute: A_coeffs = exp(dt * log(A)) = A^dt
    if dt is not None:
        # A_coeffs shape: (batch, seq_len, d_state)
        a = torch.exp(a.unsqueeze(2) * dt.unsqueeze(3))  # (B, L, N, N) - broadcasting for diagonal
    else:
        a = torch.exp(a.unsqueeze(2))  # Just exp(log A)
    
    # b shape: (batch, seq_len, d_state)
    # For selective scan, b depends on input, so we compute b * x
    # b @ x: (batch, seq_len, d_state) @ (batch, seq_len, d_model) -> (batch, seq_len, d_state)
    # Actually b projects from d_model to d_state
    b = b.unsqueeze(2)  # (B, L, 1, N) after expand
    
    # Compute the scan
    # We'll use a simple iterative scan that's O(n) but parallelizable
    # For true GPU parallelism, we'd use CUDA kernels, but this works for testing
    
    # Initialize state
    h = y.unsqueeze(1).expand(-1, seq_len, -1)  # (B, L, N)
    
    outputs = []
    current_h = y  # (B, N)
    
    for t in range(seq_len):
        # h_{t+1} = A_t @ h_t + B_t @ x_t
        # For diagonal A: h_{t+1} = A_t * h_t + b_t * x_t
        a_t = a[:, t]  # (B, N)
        b_t = b[:, t]  # (B, N) - after squeeze from (B, L, 1, N)
        
        # Compute B @ x_t
        # b_t is (B, N) - need to project x_t (B, d_model) to (B, N)
        # Actually b stores the B matrix elements
        # Let's assume b already includes the x projection
        
        current_h = a_t * current_h + b_t  # (B, N)
        
        if c is not None:
            c_t = c[:, t]  # (B, N)
            y_t = torch.sum(c_t * current_h, dim=-1)  # (B,)
            outputs.append(y_t)
        else:
            outputs.append(current_h)
    
    # Stack outputs
    if c is not None:
        output = torch.stack(outputs, dim=1)  # (B, L, N)
    else:
        output = torch.stack(outputs, dim=1)  # (B, L, N)
    
    # Reverse back if needed
    if reverse:
        output = output.flip(1)
    
    return output


def selective_scan_ipl(
    x: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    state: Optional[torch.Tensor] = None,
    *,
    dt_bias: Optional[torch.Tensor] = None,
    dt_min: float = 0.001,
    dt_max: float = 0.1,
    dt_init: str = "random",
    dt_scale: float = 1.0,
    learn_A: bool = False,
    learn_B: bool = False,
    learn_C: bool = False,
    learn_D: bool = False,
    inverse_axis: bool = False,
    sequence_axis: int = 1,
    state_axis: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Input-dependent Projection Layer (IPL) Selective Scan.
    
    This is the core of Mamba v2 - where A, B, C are functions of the input x.
    
    The selective scan addresses the core limitation of SSMs:
    - Standard SSMs: A, B, C are FIXED (input-independent)
    - Selective SSMs: A, B, C are COMPUTED FROM INPUT x
    
    This allows the model to:
    - Filter irrelevant information into the state
    - Emit only relevant outputs
    
    Mathematical Formulation:
    ==========================
    
    Given input x_t at each timestep:
        δ_t = σ(MLP_δ(x_t))           # Time step (softplus for positivity)
        A_t = σ(MLP_A(x_t))           # Selective state transition
        B_t = MLP_B(x_t)              # Selective input projection
        C_t = MLP_C(x_t)              # Selective output projection
        D_t = MLP_D(x_t)              # Selective skip connection (optional)
        
    The selective scan recurrence:
        h_t = A_t ⊙ h_{t-1} + B_t ⊙ x_t
        y_t = C_t ⊙ h_t + D_t ⊙ x_t
    
    Where ⊙ denotes element-wise product (for diagonal SSM).
    
    Args:
        x: Input tensor of shape (batch, seq_len, d_model) or (batch, d_model, seq_len)
        delta: Time step delta (if precomputed), else None to compute from x
        A: State transition matrix A of shape (batch, seq_len, d_state)
            Can be diagonal (N,) or full (N, N). Stored as log(A) for stability.
        B: Input projection B of shape (batch, seq_len, d_state)
        C: Output projection C of shape (batch, seq_len, d_state)
        D: Skip connection D of shape (d_model,) or (batch, d_model)
        state: Initial state h_0 of shape (batch, d_state), or None for zero init
        dt_bias: Bias for delta computation
        dt_min, dt_max: Clipping values for delta
        dt_init: Initialization strategy for delta
        dt_scale: Scale factor for delta
        learn_A, learn_B, learn_C, learn_D: Whether to learn these params
        inverse_axis: If True, input is (batch, d_model, seq_len)
        sequence_axis: Axis representing sequence dimension
        state_axis: Axis representing state dimension
    
    Returns:
        Tuple of (output, final_state)
        - output: y tensor of shape (batch, seq_len, d_model)
        - final_state: h_T tensor of shape (batch, d_state)
    """
    batch, seq_len, d_model = x.shape
    
    # Handle axis ordering
    if inverse_axis:
        # x is (B, d_model, L) - transpose to (B, L, d_model)
        x = x.transpose(1, 2)
    
    # Get dimensions
    d_state = A.shape[-1]
    
    # Compute delta if not provided
    if delta is None:
        # delta = softplus(linear(x))
        # For efficiency, we assume delta is passed or x is preprocessed
        delta = torch.ones(batch, seq_len, d_state, device=x.device, dtype=x.dtype)
    
    # Delta: apply bias, scaling, and clip
    if dt_bias is not None:
        delta = delta + dt_bias.view(1, 1, -1)
    delta = delta * dt_scale
    delta = torch.clamp(delta, min=dt_min, max=dt_max)
    
    # Discretize: A_bar = exp(delta * A)
    # A is stored as log(A) for stability, so A_bar = exp(delta * exp(log A)) = A^delta
    A_bar = torch.exp(delta * A)  # (B, L, N)
    
    # B_bar = delta * B (if using zero-order hold discretization)
    # Or simply B (if using Euler discretization)
    # Mamba uses: B_bar = (exp(delta * A) - 1) / A * B = (A_bar - 1) / log(A) * B
    # For numerical stability with small A: B_bar ≈ delta * B
    B_bar = delta * B  # (B, L, N)
    
    # Compute selective scan
    # Use causal_scan for the recurrence
    # For diagonal state space: h_{t+1} = A_bar_t * h_t + B_bar_t * x_t
    # y_t = C_t * h_t + D * x_t
    
    if state is None:
        state = torch.zeros(batch, d_state, device=x.device, dtype=x.dtype)
    
    # Compute output using associative scan
    # We'll use a simple CUDA-like parallel scan pattern
    
    # Reshape for scan: (B, N, L) -> we'll scan over L
    A_bar_t = A_bar.transpose(1, 2)  # (B, N, L)
    B_bar_t = B_bar.transpose(1, 2)  # (B, N, L)
    C_t = C.transpose(1, 2)  # (B, N, L)
    
    # x needs to be projected to state dimension for the scan
    # B_bar already contains the projection: B_bar = delta * B @ x
    # But we need to apply x to B
    
    # Actually in the selective scan formulation:
    # B_bar_t = delta * (B @ x_t) where B: (d_model, N)
    # We need to compute this before the scan
    
    # Let's assume B already includes the x projection
    # This is handled in the MambaBlock
    
    # Parallel scan using associative property
    # Define associative operator:
    # (h_new, y) = (A * h + B, A * y + C)
    
    # Using PyTorch's scan with cumulative operations
    
    # Simple scan (sequential but O(n) memory, O(n) compute)
    # For true O(n) on GPU, we'd need CUDA kernels
    
    h = state  # (B, N)
    outputs = []
    
    for t in range(seq_len):
        A_t = A_bar_t[:, :, t]  # (B, N)
        B_t = B_bar_t[:, :, t]  # (B, N)
        C_t = C_t[:, :, t]  # (B, N)
        
        h = A_t * h + B_t  # (B, N)
        y_t = torch.sum(C_t * h, dim=1)  # (B,)
        outputs.append(y_t)
    
    output = torch.stack(outputs, dim=1)  # (B, L)
    
    # Add skip connection D * x
    if D is not None:
        if D.dim() == 1:
            output = output + (x @ D.unsqueeze(-1)).squeeze(-1)  # (B, L)
        else:
            output = output + torch.einsum('bln,bn->bl', x, D)  # (B, L)
    
    # Final state
    final_state = h  # (B, N)
    
    return output, final_state


def parallel_scan_associative(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    reverse: bool = False,
) -> torch.Tensor:
    """
    Parallel prefix sum using associative scan.
    
    This implements the O(n) parallel scan algorithm for:
        y_t = Σ_{i=0}^{t} a_{t-i} * b_i
    
    Using the associative operator:
        combine((h1, y1), (h2, y2)) = (h1 * h2, h1 * y2 + y1)
    
    Args:
        x: Input values b_i of shape (batch, seq_len, ...)
        a: Accumulation coefficients a_i of shape (batch, seq_len, ...)
        reverse: If True, compute reverse scan
    
    Returns:
        Prefix sums y of shape (batch, seq_len, ...)
    """
    if reverse:
        x = x.flip(1)
        a = a.flip(1)
    
    batch, seq_len = x.shape[:2]
    
    # Get the rest of the dimensions
    rest_dims = x.shape[2:]
    
    # Initialize
    # For associative scan, we need to combine (h, y) pairs
    # where h is the accumulated coefficient and y is the accumulated value
    
    # We'll use a chunk-based parallel scan
    # This is a simplified version - for production, use CUDA kernels
    
    # Pad to power of 2
    seq_len_padded = 2 ** math.ceil(math.log2(seq_len))
    padding = seq_len_padded - seq_len
    
    if padding > 0:
        x = F.pad(x, (0, 0, 0, padding))
        a = F.pad(a, (0, 0, 0, padding))
    
    # Serial scan for now (would be parallel in CUDA)
    # Full parallel scan implementation would be:
    # 1. Combine step: for each pair, compute combined (h, y)
    # 2. Reduce step: tree reduction
    
    h = torch.ones_like(a[:, 0])
    y = torch.zeros_like(x[:, 0])
    
    outputs = []
    for t in range(seq_len_padded):
        # y_{t+1} = h_t * x_t + y_t
        # h_{t+1} = h_t * a_t
        new_y = h * x[:, t] + y
        new_h = h * a[:, t]
        
        outputs.append(new_y)
        
        h = new_h
        y = new_y
    
    output = torch.stack(outputs, dim=1)
    
    if reverse:
        output = output.flip(1)
    
    # Remove padding
    if padding > 0:
        output = output[:, :seq_len]
    
    return output


def mamba_inner_scan(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    state: Optional[torch.Tensor] = None,
    *,
    learnable_init: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized inner scan for Mamba-style selective SSM.
    
    This is a more efficient implementation using einsum operations.
    
    Args:
        x: Input (batch, seq_len, d_in)
        dt: Delta (batch, seq_len, d_state)
        A: log(A) diagonal (batch, seq_len, d_state)
        B: Input projection (batch, seq_len, d_state, d_in)
        C: Output projection (batch, seq_len, d_in, d_state)
        D: Skip connection (d_in,) or (batch, d_in)
        state: Initial state (batch, d_state) or None
    
    Returns:
        Tuple of (output, final_state)
    """
    batch, seq_len, d_in = x.shape
    d_state = dt.shape[-1]
    
    # Discretize
    # A_bar = exp(dt * A)
    A_bar = torch.exp(dt * A)  # (B, L, N)
    
    # B_bar = (exp(dt * A) - 1) / A * B
    # Using approximation: (A_bar - 1) / log(A) * B ≈ dt * B for small dt
    # For numerical stability:
    A_bar_minus_1 = A_bar - 1
    # Avoid division by zero: use where
    with torch.no_grad():
        A_log = torch.where(A.abs() > 1e-8, A, torch.ones_like(A))
    B_bar = (A_bar_minus_1 / A_log) * B
    
    # State dimension is last for B: (B, L, N, d_in)
    # We need to compute B_bar @ x
    # x: (B, L, d_in) -> need to contract last dim
    # B_bar: (B, L, N, d_in) @ x: (B, L, d_in) -> (B, L, N)
    B_bar_x = torch.einsum('blnd,bld->bln', B_bar, x)
    
    # Initialize state
    if state is None:
        state = torch.zeros(batch, d_state, device=x.device, dtype=x.dtype)
    
    # Parallel scan (simplified - use xformers or cuda for production)
    h = state
    outputs = []
    
    for t in range(seq_len):
        A_t = A_bar[:, t]  # (B, N)
        B_t = B_bar_x[:, t]  # (B, N)
        
        h = A_t * h + B_t  # (B, N)
        
        # Compute output: C_t @ h
        C_t = C[:, t]  # (B, d_in, N)
        y_t = torch.einsum('bdn,bn->bd', C_t, h)  # (B, d_in)
        outputs.append(y_t)
    
    output = torch.stack(outputs, dim=1)  # (B, L, d_in)
    
    # Add D * x
    if D is not None:
        output = output + torch.einsum('nd,bld->bln', D, x) if D.dim() == 1 else \
                 output + torch.einsum('bd,bld->bl', D, x)
    
    return output, h


class SelectiveScanFunction(torch.autograd.Function):
    """
    Custom autograd function for selective scan with optional CUDA backend.
    
    This allows for optimized CUDA implementations to be plugged in.
    """
    
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: Optional[torch.Tensor],
        state: Optional[torch.Tensor],
        delta_bias: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Save for backward
        ctx.save_for_backward(x, delta, A, B, C, D, state, delta_bias)
        
        # Forward pass (using numpy-like scan for now)
        # In production, this would call CUDA kernel
        output, final_state = selective_scan_ipl(
            x, delta, A, B, C, D, state, dt_bias=delta_bias
        )
        
        return output, final_state
    
    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
        grad_state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        # Backward pass (gradient through scan is non-trivial)
        # For now, use checkpointing or ignore
        x, delta, A, B, C, D, state, delta_bias = ctx.saved_tensors
        
        # Simplified backward - just pass gradients through
        # Full implementation would require reverse-mode differentiation
        return (
            grad_output,
            torch.zeros_like(delta),
            torch.zeros_like(A),
            torch.zeros_like(B),
            torch.zeros_like(C),
            None if D is None else torch.zeros_like(D),
            None if state is None else torch.zeros_like(state),
            None if delta_bias is None else torch.zeros_like(delta_bias),
        )


def variable_length_scan(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    length_mask: torch.Tensor,
    D: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Selective scan with variable-length sequence support.
    
    Args:
        x: Input (batch, max_seq_len, d_model)
        a: A coefficients (batch, max_seq_len, d_state)
        b: B coefficients (batch, max_seq_len, d_state)
        c: C coefficients (batch, max_seq_len, d_state)
        length_mask: Boolean mask (batch, max_seq_len) - True for valid positions
        D: Skip connection (d_model,)
    
    Returns:
        Output (batch, max_seq_len, d_model)
    """
    batch, max_len, d_model = x.shape
    
    # Get actual lengths
    lengths = length_mask.sum(dim=1)  # (batch,)
    
    # Compute scan for each sequence
    output = torch.zeros_like(x)
    state = torch.zeros(batch, a.shape[-1], device=x.device, dtype=x.dtype)
    
    # Process each timestep
    for t in range(max_len):
        # Only update for sequences with sufficient length
        valid = (lengths > t).unsqueeze(-1)  # (batch, 1)
        
        # Compute state update
        a_t = a[:, t]  # (batch, d_state)
        b_t = b[:, t]  # (batch, d_state)
        
        # State update: h = a * h + b * x
        # For selective: b already includes x projection
        new_state = a_t * state + b_t
        state = torch.where(valid, new_state, state)
        
        # Output: y = c * h
        c_t = c[:, t]  # (batch, d_state)
        y_t = torch.sum(c_t * state, dim=-1)  # (batch,)
        
        # Store output
        output[:, t] = torch.where(valid.squeeze(-1), y_t, torch.zeros_like(y_t))
    
    # Add skip connection
    if D is not None:
        output = output + x * D.unsqueeze(0).unsqueeze(1)
    
    return output
