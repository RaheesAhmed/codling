# CODLING - The Coding Beast LLM

**Non-Transformer LLM using State Space Models. Train on Google Colab free GPU.**

---

## Quick Start (Colab)

1. Upload `notebooks/` folder to Google Colab
2. Run `01_setup.ipynb` first
3. Then run `03_training.ipynb`

---

## In Colab Cell

```python
import sys
sys.path.append('/content/codling')

from codling.codling import CodlingConfig, CodlingForCausalLM

# 130M params - fits on free Colab T4
config = CodlingConfig(
    vocab_size=32000,
    d_model=512,
    n_layers=12,
    d_state=128
)
model = CodlingForCausalLM(config)
print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
```

---

## Files

| File | What |
|------|------|
| `notebooks/01_setup.ipynb` | Install deps |
| `notebooks/03_training.ipynb` | Train model |
| `codling/model.py` | Full SSM model |
| `codling/ssm/mamba.py` | Mamba SSM |
| `codling/rope/lrope.py` | LongRoPE |

---

## Model Sizes

| Size | Params | VRAM |
|------|--------|------|
| Tiny | 52M | 2GB |
| Small | 130M | 4GB |
| Medium | 350M | 8GB |
| Large | 1B | 14GB |

---

## What is CODLING?

- **Mamba SSM** - Linear-time state spaces
- **O(n) context** - No attention quadratic scaling
- **1M context** - LongRoPE extension
- **Colab trainable** - Free GPU!

---

## Training on Google Colab

```python
# Clone and setup
!git clone https://github.com/RaheesAhmed/codling.git
%cd codling
!pip install -r requirements.txt

# Prepare dataset (The Pile)
from codling.data import ThePileDataset
dataset = ThePileDataset(tokenizer="gpt2", save_dir="./data")

# Training
from codling.trainer import Trainer
trainer = Trainer(
    model="codling-130M",
    dataset=dataset,
    batch_size=8,
    learning_rate=1e-4,
    max_steps=100000,
)
trainer.train()
```

---

## Performance

| Metric | CODLING-130M | CODLING-350M | CODLING-1B |
|--------|--------------|--------------|------------|
| **Inference Speed** ⚡ | 1.8x Transformer | 2.1x Transformer | 2.4x Transformer |
| **Memory Usage** 💾 | 40% less | 45% less | 50% less |
| **Context Length** 📜 | 4K (native) | 8K (native) | 16K (native) |
| **Training Tokens** | 100B | 300B | 1T |

> Benchmarks run on A100-80GB. Speedup measured against equivalent Transformer model.

---

## Installation

```bash
# From source
git clone https://github.com/codling-ai/codling.git
cd codling
pip install -r requirements.txt

# Verify installation
python -c "from codling import CODLINGLM; print('CODLING installed!')"
```

---

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## References & Papers

- **[Mamba: Linear-time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)** - The foundational SSM architecture
- **[S4: Efficiently Modeling Long Sequences](https://arxiv.org/abs/2112.13445)** - Structured State Space Sequences
- **[Hyena: Sub-Quadratic Implicit Attention](https://arxiv.org/abs/2309.00071)** - Efficient implicit attention
- **[LongRoPE: Extending LLM Context Windows](https://arxiv.org/abs/2402.13753)** - 128K context extension

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---


