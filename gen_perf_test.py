import torch
import time
import argparse
from codling.codling.model import CodlingConfig, CodlingForCausalLM

def benchmark_inference(model: CodlingForCausalLM, prompt_len=16, gen_len=64, tries=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    print(f"Benchmarking on {device.type.upper()}")
    vocab_size = model.config.vocab_size
    
    # Warmup
    x = torch.randint(1, vocab_size, (1, prompt_len), device=device)
    _ = model.generate(x, max_new_tokens=2)
    
    total_time = 0.0
    with torch.no_grad():
        for _ in range(tries):
            x = torch.randint(1, vocab_size, (1, prompt_len), device=device)
            start = time.time()
            out = model.generate(x, max_new_tokens=gen_len)
            end = time.time()
            total_time += (end - start)
            
    avg_time = total_time / tries
    tokens_per_sec = gen_len / avg_time
    
    # Memory
    mem_mb = 0
    if device.type == 'cuda':
        mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
    params = sum(p.numel() for p in model.parameters()) / 1e6
    
    print(f"Model Params: {params:.2f} M")
    if mem_mb > 0:
        print(f"Max VRAM: {mem_mb:.2f} MB")
    print(f"Avg Time: {avg_time:.3f} s")
    print(f"Speed: {tokens_per_sec:.2f} tokens / sec")
    
    return tokens_per_sec

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=str, default="tiny", choices=["micro", "tiny", "small"])
    parser.add_argument("--type", type=str, default="s4", choices=["s4", "mamba"])
    args = parser.parse_args()
    
    if args.size == "micro":
        config = CodlingConfig(vocab_size=32000, d_model=128, n_layers=4, d_state=32, ssm_type=args.type)
    elif args.size == "tiny":
        config = CodlingConfig(vocab_size=32000, d_model=256, n_layers=6, d_state=64, ssm_type=args.type)
    else:  # small
        config = CodlingConfig(vocab_size=32000, d_model=512, n_layers=12, d_state=128, ssm_type=args.type)
        
    print(f"--- CODLING {args.size.upper()} ({args.type.upper()}) ---")
    model = CodlingForCausalLM(config)
    benchmark_inference(model)
