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
    parser.add_argument("--fast", action="store_true", help="Use fast mode (less accurate, quicker, for low-end GPUs)")
    parser.add_argument("--superfast", action="store_true", help="Use super fast mode (least accurate, quickest, for very low-end GPUs)")


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

    print(f"[+] Target Profile: {profile}")

    if args.fast:
        schedule = Config.SCHEDULE_FAST
        print("[+] Mode: FAST (Lower accuracy, higher speed)")
    elif args.superfast:
        schedule = Config.SCHEDULE_SUPERFAST
        print("[+] Mode: SUPERFAST (Lowest accuracy, highest speed)")
    else:
        schedule = Config.SCHEDULE_STANDARD
        print("[+] Mode: STANDARD (High accuracy)")

    print(f"\n[+] Target Locked: {profile['name']}")
    print("[+] Cracking...")

    # CALL THE LOGIC FILE
    candidates = predict_password(model, tokenizer, profile, beam_schedule=schedule)

    print(f"\n--- TOP GUESSES ---")
    print(f"{'CONFIDENCE':<12} | {'PASSWORD'}")
    print("-" * 30)

    for cand in candidates[:100]: # Show top 100
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

    # Construct filename: results/guesses_John.json
    safe_name = profile.get('name', 'target').replace(" ", "_")
    output_path = Config.RESULTS_DIR / f"guesses_{safe_name}.json"

    print(f"\n[+] Saving {len(candidates)} candidates to: {output_path}")

    output_data = []
    for cand in candidates:
        # Decode the text
        text = tokenizer.decode(cand['sequence'], skip_special_tokens=True)
        
        # Calculate probability if not already there
        if 'probability' in cand:
            prob = cand['probability']

        output_data.append({
            "password": text,
            "confidence": f"{prob:.4f}%",
            "log_score": cand['score']
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)
    

if __name__ == "__main__":
    main()