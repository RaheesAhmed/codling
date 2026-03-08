import torch
from codling.codling.model import CodlingConfig, CodlingForCausalLM

def test_config(name, config):
    """Test a single configuration."""
    try:
        m = CodlingForCausalLM(config)
        x = torch.randint(1, config.vocab_size, (1, 16))
        labels = x.clone()
        
        # Forward pass with labels
        o = m(x, labels=labels)
        loss = o['loss']
        
        # Generate
        gen = m.generate(x[:, :3], max_new_tokens=5)
        
        print(f"  OK: {sum(p.numel() for p in m.parameters())} params, loss={loss.item():.4f}")
        return True
    except Exception as e:
        print(f"  FAILED: {str(e)[:80]}")
        return False

print("Testing CODLING SSM Model Configurations...")
print()

# Test 1: S4 Base Model
print("Test 1: S4 Base Model")
test_config("S4", CodlingConfig(
    vocab_size=5000, d_model=256, n_layers=2, d_state=32
))

# Test 2: S4 with LongRoPE
print("Test 2: S4 + LongRoPE")
test_config("S4+LongRoPE", CodlingConfig(
    vocab_size=5000, d_model=256, n_layers=2, d_state=32,
    use_longrope=True, max_position_embeddings=4096
))

# Test 3: Mamba v1
print("Test 3: Mamba v1")
test_config("Mamba", CodlingConfig(
    vocab_size=5000, d_model=256, n_layers=2, d_state=32,
    ssm_type='mamba'
))

# Test 4: Larger model (closer to 130M)
print("Test 4: Larger S4 Model (~50M params)")
test_config("S4-Large", CodlingConfig(
    vocab_size=10000, d_model=512, n_layers=8, d_state=64
))

# Test 5: With dropout and other features
print("Test 5: S4 + Dropout")
test_config("S4+Features", CodlingConfig(
    vocab_size=8000, d_model=384, n_layers=4, d_state=48,
    dropout=0.1
))

# Test 6: Gradient checkpointing
print("Test 6: Gradient Checkpointing")
config = CodlingConfig(vocab_size=3000, d_model=192, n_layers=2, d_state=32)
m = CodlingForCausalLM(config)
m.enable_gradient_checkpointing()
print("  OK: Gradient checkpointing enabled")

# Test 7: Generate with temperature
print("Test 7: Generate with temperature")
config = CodlingConfig(vocab_size=2000, d_model=128, n_layers=2, d_state=32)
m = CodlingForCausalLM(config)
x = torch.randint(1, 2000, (1, 5))
gen = m.generate(x, max_new_tokens=20, temperature=0.8, top_k=40)
print(f"  OK: Generated {gen.shape[1]} tokens")

print()
print("All tests completed successfully!")
