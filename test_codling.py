import torch
from codling.codling.model import CodlingConfig, CodlingForCausalLM

c = CodlingConfig(d_model=128, n_layers=2, vocab_size=3000)
m = CodlingForCausalLM(c)
x = torch.randint(1, 3000, (1, 10))
labels = torch.randint(1, 3000, (1, 10))
o = m(x, labels=labels)
l = o['loss']
gen = m.generate(x, max_new_tokens=5)

with open('test_output.txt', 'w') as f:
    f.write(f'PARAMS: {sum(p.numel() for p in m.parameters())}\n')
    f.write(f'LOGITS_SHAPE: {o["logits"].shape}\n')
    f.write(f'LOSS: {l.item()}\n')
    f.write(f'GEN_SHAPE: {gen.shape}\n')
    f.write('SUCCESS!\n')
print("DONE")
