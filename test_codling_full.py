import torch
from codling.codling.model import CodlingConfig, CodlingForCausalLM
import sys

results = []

def test_config(name, config):
    print(f"Testing {name}...")
    try:
        model = CodlingForCausalLM(config)
        x = torch.randint(1, config.vocab_size, (1, 16))
        labels = torch.randint(1, config.vocab_size, (1, 16))
        o = model(x, labels=labels)
        gen = model.generate(x, max_new_tokens=5)
        params = sum(p.numel() for p in model.parameters())
        results.append(f"{name}: OK - {params} params, loss={o['loss'].item():.4f}")
        print(f"  OK: {params} params, loss={o['loss'].item():.4f}")
    except Exception as e:
        results.append(f"{name}: FAILED - {str(e)}")
        print(f"  FAILED: {e}")

# Test 1: Base S4 model (larger for better performance)
test_config("S4 Base", CodlingConfig(
    vocab_size=32000, d_model=512, n_layers=6, d_state=128
))

# Test 2: S4 with Hyena
test_config("S4 + Hyena", CodlingConfig(
    vocab_size=32000, d_model=512, n_layers=6, d_state=128,
    use_hyena=True, hyena_num_heads=8, hyena_filter_order=64
))

# Test 3: S4 with Linear Attention
test_config("S4 + LinearAttn", CodlingConfig(
    vocab_size=32000, d_model=512, n_layers=6, d_state=128,
    use_linear_attn=True, attn_chunk_size=512
))

# Test 4: S4 with LongRoPE
test_config("S4 + LongRoPE", CodlingConfig(
    vocab_size=32000, d_model=512, n_layers=6, d_state=128,
    use_longrope=True, max_position_embeddings=4096
))

# Test 5: S4 with all features
test_config("S4 + Hyena + LinearAttn + LongRoPE", CodlingConfig(
    vocab_size=32000, d_model=512, n_layers=6, d_state=128,
    use_hyena=True, hyena_num_heads=8, hyena_filter_order=64,
    use_linear_attn=True, attn_chunk_size=512,
    use_longrope=True, max_position_embeddings=4096
))

# Test 6: Mamba v1
test_config("Mamba v1", CodlingConfig(
    vocab_size=32000, d_model=512, n_layers=6, d_state=128,
    ssm_type='mamba'
))

# Test 7: Gradient checkpointing
print("Testing gradient checkpointing...")
model = CodlingForCausalLM(CodlingConfig(vocab_size=3000, d_model=128, n_layers=2))
model.enable_gradient_checkpointing()
results.append("Gradient checkpointing: OK")
print("  OK")

# Write results
with open('test_results.txt', 'w') as f:
    f.write("CODLING Model Test Results\n")
    f.write("=" * 50 + "\n")
    for r in results:
        f.write(r + "\n")
    f.write("\nAll tests completed!")

print("\nAll tests completed!")
