import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.model import LoRALayer
from src.generation_engine import dynamic_beam_search
from src.loader import build_model, inject_lora_layers 

WEIGHTS_PATH = "PassLLM_LoRA_Weights.pth"

def main():
    print("Loading System...")

    model, tokenizer = build_model()
    model = inject_lora_layers(model)
    
    model.load_state_dict(torch.load(WEIGHTS_PATH), strict=False)
    model.eval()

    # --- TARGETED PASSWORD GUESSING INFERENCE (ALPACA) ---
    print("\n--- TARGETING USER ---")
    # 1. EXACT System Prompt from data generator
    instruction = "As a targeted password guessing model, your task is to utilize the provided information to guess the corresponding password."  
    # 2. EXACT Data Format from 'user_input_str'
    # Format: "Name: {First} {Last}, Born: {Year}, User: {Username}, SisterPW: {SisterPW}"
    # Note: MUST provide a 'SisterPW' (or a placeholder) because the model was trained to expect it
    user_details = "Name: Matthew Miller, Born: 1950, User: matthew, SisterPW: 123456"    
    # 3. Combine them EXACTLY as train.py did
    # Logic: f"{sample['instruction']}\n{sample['input']}\nPassword: "
    prompt = f"{instruction}\n{user_details}\nPassword: "

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)

    candidates = dynamic_beam_search(
        model=model, tokenizer=tokenizer, auxiliary_info_ids=input_ids,
        max_depth=16, beam_width_schedule=[150, 150, 100, 100, 100, 50, 50, 50, 20, 20, 20, 10, 10, 10, 10, 10, 10]
    )

    print(f"\n--- GENERATED {len(candidates)} CANDIDATES ---")
    print(f"{'PROBABILITY':<12} | {'PASSWORD'}")
    print("-" * 30)

    for cand in candidates:
        pwd = tokenizer.decode(cand['sequence'], skip_special_tokens=True)
        prob = torch.exp(torch.tensor(cand['score'])).item() # Log_Prob -> %
        print(f"{prob:.2%}      | {pwd}")

if __name__ == "__main__":
    main()