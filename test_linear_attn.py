"""Test script for CODLING Linear Attention module."""
import torch
import sys

# Test imports
from codling.codling.attention import (
    LinearAttention,
    TiledFlashLinearAttention,
    BasedLinearAttention,
    elu_feature_map,
    causal_mask,
)

def test_feature_map():
    """Test ELU+1 feature map."""
    x = torch.randn(2, 4, 8)
    phi_x = elu_feature_map(x)
    assert phi_x.shape == x.shape
    assert (phi_x > 0).all(), "Feature map should be all positive"
    print("[PASS] Feature map test passed")

def test_causal_mask():
    """Test causal mask generation."""
    mask = causal_mask(8)
    expected = torch.tril(torch.ones(8, 8, dtype=torch.bool))
    assert mask.shape == (8, 8)
    assert mask.equal(expected), "Should be lower triangular"
    print("[PASS] Causal mask test passed")

def test_linear_attention():
    """Test LinearAttention module."""
    attn = LinearAttention(dim=64, heads=4)
    q = torch.randn(2, 16, 64)
    k = torch.randn(2, 16, 64)
    v = torch.randn(2, 16, 64)
    out = attn(q, k, v)
    assert out.shape == q.shape
    print("[PASS] LinearAttention test passed")

def test_linear_attention_causal():
    """Test LinearAttention with causal masking."""
    attn = LinearAttention(dim=64, heads=4)
    q = torch.randn(2, 16, 64)
    k = torch.randn(2, 16, 64)
    v = torch.randn(2, 16, 64)
    out = attn(q, k, v, causal=True)
    assert out.shape == q.shape
    print("[PASS] LinearAttention (causal) test passed")

def test_tfla():
    """Test TiledFlashLinearAttention."""
    tfla = TiledFlashLinearAttention(dim=64, heads=4, chunk_size=8)
    q = torch.randn(2, 16, 64)
    k = torch.randn(2, 16, 64)
    v = torch.randn(2, 16, 64)
    out = tfla(q, k, v)
    assert out.shape == q.shape
    print("[PASS] TiledFlashLinearAttention test passed")

def test_tfla_causal():
    """Test TiledFlashLinearAttention with causal masking."""
    tfla = TiledFlashLinearAttention(dim=64, heads=4, chunk_size=8, causal=True)
    q = torch.randn(2, 16, 64)
    k = torch.randn(2, 16, 64)
    v = torch.randn(2, 16, 64)
    out = tfla(q, k, v)
    assert out.shape == q.shape
    print("[PASS] TiledFlashLinearAttention (causal) test passed")

def test_based_linear_attention():
    """Test BasedLinearAttention."""
    based = BasedLinearAttention(dim=64, heads=4, use_linear=True)
    q = torch.randn(2, 16, 64)
    k = torch.randn(2, 16, 64)
    v = torch.randn(2, 16, 64)
    out = based(q, k, v)
    assert out.shape == q.shape
    print("[PASS] BasedLinearAttention test passed")

def test_bf16():
    """Test BF16 mixed precision."""
    attn = LinearAttention(dim=64, heads=4).to(torch.bfloat16)
    q = torch.randn(2, 16, 64).to(torch.bfloat16)
    k = torch.randn(2, 16, 64).to(torch.bfloat16)
    v = torch.randn(2, 16, 64).to(torch.bfloat16)
    out = attn(q, k, v)
    assert out.dtype == torch.bfloat16
    print("[PASS] BF16 mixed precision test passed")

def test_streaming():
    """Test streaming mode."""
    tfla = TiledFlashLinearAttention(dim=64, heads=4, use_buffer_state=True)
    for i in range(5):
        q_tok = torch.randn(1, 64)
        k_tok = torch.randn(1, 64)
        v_tok = torch.randn(1, 64)
        out_tok = tfla.forward_streaming(q_tok, k_tok, v_tok)
    assert out_tok.shape == (1, 64)
    print("[PASS] Streaming mode test passed")

def test_large_sequence():
    """Test with large sequence."""
    tfla = TiledFlashLinearAttention(dim=64, heads=4, chunk_size=512)
    q = torch.randn(1, 2048, 64)
    k = torch.randn(1, 2048, 64)
    v = torch.randn(1, 2048, 64)
    out = tfla(q, k, v)
    assert out.shape == q.shape
    print("[PASS] Large sequence test passed")

if __name__ == "__main__":
    print("=" * 50)
    print("CODLING Linear Attention Test Suite")
    print("=" * 50)
    
    test_feature_map()
    test_causal_mask()
    test_linear_attention()
    test_linear_attention_causal()
    test_tfla()
    test_tfla_causal()
    test_based_linear_attention()
    test_bf16()
    test_streaming()
    test_large_sequence()
    
    print("=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)
