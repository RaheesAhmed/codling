"""
CODLING SSM (Selective State Space Model) Core Module

This module provides the core SSM implementations for the CODLING architecture:

1. Selective Scan - Parallel O(n) scan algorithm for SSM recurrence
2. S4 Layer - Structured State Space with HiPPO initialization
3. Mamba v2 - Selective SSM with input-dependent parameters

Key Classes:
-----------

Selective Scan:
    - causal_scan: Parallel prefix sum algorithm
    - selective_scan_ipl: Input-dependent projection layer
    - variable_length_scan: Support for variable-length sequences
    - SelectiveScanFunction: Custom autograd with CUDA support

S4:
    - S4Layer: Full S4 layer with NPLR representation
    - S4Block: S4 with residual and optional FFN
    - S4Kernel: Efficient convolution kernel computation
    - BidirectionalS4: Bidirectional S4

Mamba:
    - MambaBlock: Core Mamba v2 selective SSM block
    - Mamba2Block: Mamba v2 with multi-head selection
    - MambaResidualBlock: Mamba with residual and layer norm
    - create_mamba_block: Factory function

Mathematical Background:
========================

SSM (State Space Model):
    h_t = A @ h_{t-1} + B @ x_t
    y_t = C @ h_t + D @ x_t

The key innovation is computing this recurrence in O(n) using parallel scan.

S4 (Structured State Space):
    - Uses diagonal + low-rank (NPLR) representation for A
    - HiPPO initialization for long-range memory
    - FFT-based convolution for kernel computation

Mamba (Selective SSM):
    - A, B, C are functions of input x (selective mechanism)
    - Conv1d preprocessing for positional bias
    - Gated activation (SiLU)
    - Better scaling to large state dimensions

Usage:
======

    from codling.ssm import S4Layer, MambaBlock
    
    # S4 layer
    s4 = S4Layer(d_model=768, d_state=128)
    output, state = s4(x)
    
    # Mamba block
    mamba = MambaBlock(d_model=768, d_state=128)
    output = mamba(x)
"""

# Selective Scan exports
from codling.codling.ssm.selective_scan import (
    causal_scan,
    selective_scan_ipl,
    parallel_scan_associative,
    mamba_inner_scan,
    SelectiveScanFunction,
    variable_length_scan,
)

# S4 exports
from codling.codling.ssm.s4 import (
    S4Layer,
    S4Block,
    S4Kernel,
    BidirectionalS4,
    hippo_legendre,
    hippo_legt,
    hippo_fourier,
    discretize_bilinear,
    discretize_zoh,
)

# Mamba exports
from codling.codling.ssm.mamba import (
    MambaBlock,
    Mamba2Block,
    MambaBlockv2,
    MambaResidualBlock,
    create_mamba_block,
    MambaConfig,
)

# Version info
__version__ = "0.1.0"

# __all__ for clean imports
__all__ = [
    # Selective scan
    "causal_scan",
    "selective_scan_ipl",
    "parallel_scan_associative",
    "mamba_inner_scan",
    "SelectiveScanFunction",
    "variable_length_scan",
    
    # S4
    "S4Layer",
    "S4Block", 
    "S4Kernel",
    "BidirectionalS4",
    "hippo_legendre",
    "hippo_legt",
    "hippo_fourier",
    "discretize_bilinear",
    "discretize_zoh",
    
    # Mamba
    "MambaBlock",
    "Mamba2Block",
    "MambaBlockv2",
    "MambaResidualBlock",
    "create_mamba_block",
    "MambaConfig",
]
