import argparse
import torch
import sys
import json
from src.loader import build_model, inject_lora_layers 
from inference import predict_password # Import logic
from src.config import Config


def parse_arguments():
    parser = argparse.ArgumentParser(description="PassLLM Command Line Tool")
    
    parser.add_argument("--file", type=str, default=None, required = True, help="Path to target.json file with PII")

    # --- CONFIGURATION ---
    parser.add_argument("--weights", type=str, default="models/PassLLM_LoRA_Weights.pth", help="Path to model weights")
    parser.add_argument("--fast", action="store_true", help="Use fast mode (less accurate, quicker)")

    return parser.parse_args()

def load_profile(args):
    """
    Loads PII from either the JSON file OR the Command Line arguments.
    """
    profile = {}

    # Priority 1: Load from JSON file
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                profile = json.load(f)
            print(f"[+] Loaded target info from: {args.file}")
        except Exception as e:
            print(f"[!] Error loading JSON: {e}")
            sys.exit(1)

    # Validation
    if not profile:
        print("[!] Error: No target information provided.")
        print("    Usage: python app.py --file target.json")
        sys.exit(1)

    return profile

def main():
    # Handle our arguments
    args = parse_arguments()
    
    # Load our system (model + tokenizer)
    print(f"Loading PassLLM System...")
    try:
        model, tokenizer = build_model()
        model = inject_lora_layers(model)
        # Load the fine-tuned weights
        model.load_state_dict(torch.load(args.weights, map_location=model.device), strict=False)
        model.eval()
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Weights not found at {args.weights}")
        sys.exit(1)

    # Setup the target profile
    profile = load_profile(args)

    if args.fast:
        schedule = Config.SCHEDULE_FAST
        print("[+] Mode: FAST (Lower accuracy, higher speed)")
    else:
        schedule = Config.SCHEDULE_STANDARD
        print("[+] Mode: STANDARD (High accuracy)")

    print(f"\n[+] Target Locked: {profile['first_name']}")
    print("[+] Cracking...")

    # CALL THE LOGIC FILE
    candidates = predict_password(model, tokenizer, profile, beam_schedule=schedule)

    print(f"\n--- TOP GUESSES ---")
    print(f"{'CONFIDENCE':<12} | {'PASSWORD'}")
    print("-" * 30)

    for cand in candidates[:250]: # Show top 150
        pwd = tokenizer.decode(cand['sequence'], skip_special_tokens=True)
        # Prefer normalized percentage if generation attached it
        if 'probability' in cand:
            print(f"{cand['probability']:.2f}% | {pwd} ({cand['score']:.4f} log)")
        else:
            prob = torch.exp(torch.tensor(cand['score'])).item()
            if prob > 0.001:
                print(f"{prob:.2%}       | {pwd} ({cand['score']:.4f} log)")
            else:
                print(f"{cand['score']:.4f} (log) | {pwd}")

if __name__ == "__main__":
    main()