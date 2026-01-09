import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from model import LoRALayer 

MODEL_ID = "mistralai/Mistral-7B-v0.1" # The paper uses Mistral-7B
DATA_FILE = "passllm_alpaca_data.jsonl"
OUTPUT_DIR = "PassLLM_Model"

def built_and_train():
    # We load in 4-bit (runs on consumer GPU)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        load_in_4bit=True 
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token # Fix for training
    
    # We iterate through the model and replace the Linear layers with OUR LoRALayer.
    
    for name, module in model.named_modules():
        # Identify the target layers (Attention blocks)
        if "q_proj" in name or "k_proj" in name or "v_proj" in name:
            # Get the parent module (e.g., the attention block)
            parent_name = name.rsplit('.', 1)[0]
            child_name = name.rsplit('.', 1)[1]
            parent = model.get_submodule(parent_name)
            
            # Create custom LoRA layer wrapping the original
            lora_layer = LoRALayer(module, rank=16, alpha=32)
            
            # Replace the old layer with your new one
            setattr(parent, child_name, lora_layer)
            print(f"   Replaced: {name}")

            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            all_params = sum(p.numel() for p in model.parameters())
            print(f"   Trainable Parameters: {trainable_params} (approx {100 * trainable_params / all_params:.2f}%)")

if __name__ == "__main__":
    build_and_train()