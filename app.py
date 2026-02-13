import argparse
import torch
import sys
import json
from src.loader import build_model, inject_lora_layers 
from inference import predict_password # Import logic
from src.config import Config

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def parse_arguments():
    parser = argparse.ArgumentParser(description="PassLLM Command Line Tool")
    parser.add_argument("--file", type=str, default=None, required = True, help="Path to target.json file with PII")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights (default: from config)")
    parser.add_argument("--fast", action="store_true", help="Use fast mode (less accurate, quicker, for low-end GPUs)")
    parser.add_argument("--superfast", action="store_true", help="Use super fast mode (least accurate, quickest, for very low-end GPUs)")
    parser.add_argument("--deep", action="store_true", help="Use deep mode (highest accuracy, slowest, for high-end GPUs)")
    return parser.parse_args()

def load_profile(args):
    profile = {}

    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                profile = json.load(f)
            print(f"[+] Loaded target info from: {args.file}")
        except Exception as e:
            print(f"[!] Error loading JSON: {e}")
            sys.exit(1)
            
    if not profile:
        print("[!] Error: No target information provided.")
        print("    Usage: python app.py --file target.json")
        sys.exit(1)

    return profile

def load():
    print("[+] Loading PassLLM System for Web UI...")
    try:
        model, tokenizer = build_model()
        model = inject_lora_layers(model)

        weights_path = Config.WEIGHTS_FILE
        print(f"[+] Loading Weights from: {weights_path}...")

        checkpoint = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)    

        model.eval()
        return model, tokenizer

    except FileNotFoundError:
        print(f"CRITICAL ERROR: Weights not found at {weights_path}")
        sys.exit(1)

def main():
    args = parse_arguments()
    # Load our system (model + tokenizer)
    print(f"Loading PassLLM System...")
    model, tokenizer = load()

    profile = load_profile(args)

    if args.fast:
        schedule = Config.SCHEDULE_FAST
        print("[+] Mode: FAST (Lower accuracy, higher speed)")
    elif args.superfast:
        schedule = Config.SCHEDULE_SUPERFAST
        print("[+] Mode: SUPERFAST (Lowest accuracy, highest speed)")
    elif args.deep:
        schedule = Config.SCHEDULE_DEEP
        print("[+] Mode: DEEP (Highest accuracy, slowest speed)")
    else:
        schedule = Config.SCHEDULE_STANDARD
        print("[+] Mode: STANDARD (High accuracy)")

    print(f"[+] Locked: {profile['name']}")
    print("[+] Guessing...")

    candidates = predict_password(model, tokenizer, profile, beam_schedule=schedule)

    print(f"\n--- TOP GUESSES ---")
    print(f"{'CONFIDENCE':<12} | {'PASSWORD'}")
    print("-" * 30)

    for cand in candidates[:100]: # Show top 100, the rest will be logged in /results
        pwd = cand['password']
        # Prefer normalized percentage if generation attached it
        if 'probability' in cand:
            print(f"{cand['probability']:.2f}% | {pwd} ({cand['score']:.4f} log)")
        else:
            prob = torch.exp(torch.tensor(cand['score'])).item()
            if prob > 0.001:
                print(f"{prob:.2%}       | {pwd} ({cand['score']:.4f} log)")
            else:
                print(f"{cand['score']:.4f} (log) | {pwd}")

    safe_name = profile.get('name', 'target').replace(" ", "_")
    output_path = Config.RESULTS_DIR / f"guesses_{safe_name}.json"

    output_data = []
    for cand in candidates:
        text = cand['password']
        
        if 'probability' in cand:
            prob = cand['probability']

        output_data.append({
            "password": text,
            "confidence": f"{prob:.4f}%",
            "log_score": cand['score']
        })

    counter = 1
    while output_path.exists():
        output_path = Config.RESULTS_DIR / f"guesses_{safe_name}_{counter}.json"
        counter += 1

    print(f"\n[+] Saving {len(candidates)} candidates to: {output_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)
    

if __name__ == "__main__":
    main()
