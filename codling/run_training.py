import os
import sys
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_training")

# Add parent directory to import training_data and fix module path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
# Remove script directory from sys.path to avoid "codling" package conflict
script_dir = os.path.abspath(os.path.dirname(__file__))
if script_dir in sys.path:
    sys.path.remove(script_dir)

from training_data import RAW_TS_CORPUS

from codling.codling.model import CodlingConfig, CodlingForCausalLM
from codling.codling.trainer import Trainer, TrainerConfig, create_dataloader
from codling.codling.inference import create_generator, GenerationConfig

class CodeDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_length=512):
        logger.info(f"Tokenizing {len(texts)} snippets...")
        
        # We process each snippet and handle padding
        self.inputs = []
        
        for text in texts:
            # Add BOS and EOS tokens as string if needed, or let tokenizer handle it
            encoded = tokenizer(
                text, 
                truncation=True, 
                max_length=seq_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            input_ids = encoded.input_ids[0]
            
            # Create autoregressive labels: shift by 1 is handled in the model/trainer (loss computation)
            self.inputs.append({
                'input_ids': input_ids,
                'labels': input_ids.clone()
            })
            
    def __len__(self):
        return len(self.inputs)
        
    def __getitem__(self, idx):
        return self.inputs[idx]

def main():
    logger.info("Initializing Tokenizer...")
    # Using GPT-2 tokenizer as a standard subword tokenizer
    # It works decently for code when a specialized one isn't available
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    logger.info(f"Loaded {len(RAW_TS_CORPUS)} snippets from training_data.py")
    
    # Configuration
    SEQ_LENGTH = 128
    BATCH_SIZE = 4
    MAX_STEPS = 100
    
    # Create dataset
    dataset = CodeDataset(RAW_TS_CORPUS, tokenizer, seq_length=SEQ_LENGTH)
    dataloader = create_dataloader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Model Configuration (Tiny size for quick demonstration)
    config = CodlingConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=384,
        n_layers=6,
        d_state=64,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        use_hyena=False,  # Keep it simple SSM for speed 
        use_linear_attn=False
    )
    
    logger.info("Initializing Codling Model...")
    model = CodlingForCausalLM(config)
    
    # Print param count
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
    trainer_config = TrainerConfig(
        model=model,
        train_dataloader=dataloader,
        output_dir="./checkpoints",
        max_steps=MAX_STEPS,
        learning_rate=2e-3,
        warmup_steps=20,
        gradient_accumulation_steps=1,
        use_gradient_checkpointing=False,
        precision="fp32", # Use standard precision for CPU/GPU portability
        logging_steps=5,
        save_steps=50,
        batch_size=BATCH_SIZE,
        seq_length=SEQ_LENGTH
    )
    
    logger.info("Starting Training...")
    trainer = Trainer(trainer_config)
    trainer.train()
    
    logger.info("Training Complete! Testing generation...")
    
    # Test generation
    model.eval()
    generator = create_generator(model, tokenizer=tokenizer)
    
    test_prompts = [
        "class AppError(Exception):",
        "async def with_retry(coro_func",
        "class LRUCache(Generic"
    ]
    
    gen_config = GenerationConfig(
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.1,
        streaming=False
    )
    
    # We move model to device in trainer, so ensure inputs are on correct device
    device = trainer.device
    
    for prompt in test_prompts:
        print(f"\n{'='*50}\nPrompt: {prompt}\n{'-'*50}")
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        try:
            # Generate output text directly (since streaming=False, return_dict=False)
            output_text = generator.generate(input_ids, config=gen_config, return_dict=False)
            print(output_text)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            
    print(f"\n{'='*50}\nDone examining model behavior!")

if __name__ == "__main__":
    main()
