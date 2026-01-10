import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.model import LoRALayer
from src.generation_engine import dynamic_beam_search

MODEL_ID = "Qwen/Qwen2.5-0.5B"
LORA_RANK = 16
LORA_ALPHA = 32
WEIGHTS_PATH = "PassLLM_LoRA_Weights.pth"

def main():
    print("Loading System...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        device_map="auto",
        load_in_4bit=True
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    for name, module in model.named_modules():
        if "q_proj" in name or "v_proj" in name:
            parent_name = name.rsplit('.', 1)[0]
            child_name = name.rsplit('.', 1)[1]
            parent_module = model.get_submodule(parent_name)
            lora_layer = LoRALayer(
                original_layer=module,
                rank=16,
                alpha=32
            )
            lora_layer.to(module.weight.device)
            setattr(parent_module, child_name, lora_layer)
    
    model.load_state_dict(torch.load(WEIGHTS_PATH), strict=False)
    model.eval()

    print("\n--- TARGETING USER ---")
    instruction = "Retrieve the password for the user 'admin@example.com'."
    user_info = "Name: Admin User\nBirth Year: 1990"
    prompt = f"{instruction}\n{user_info}\nPassword: "
    print(f"Context:\n{prompt.strip()}")

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)

    candidates = dynamic_beam_search(
        model=model, tokenizer=tokenizer, auxiliary_info_ids=input_ids,
        max_depth=10, beam_width_schedule=[50, 50, 50, 50, 20, 20, 10, 10, 10, 10]
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