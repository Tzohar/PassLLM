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
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights (default: from config)")
    parser.add_argument("--fast", action="store_true", help="Use fast mode (less accurate, quicker, for low-end GPUs)")
    parser.add_argument("--superfast", action="store_true", help="Use super fast mode (least accurate, quickest, for very low-end GPUs)")
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

def main():
    args = parse_arguments()
    
    # Load our system (model + tokenizer)
    print(f"Loading PassLLM System...")
    try:
        model, tokenizer = build_model()
        model = inject_lora_layers(model)

        weights_path = args.weights if args.weights else Config.WEIGHTS_FILE
        print(f"Loading Weights from: {weights_path}...")

        # We're using the cpu map_location to ensure compatibility across devices
        checkpoint = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)    

        model.eval()

    except FileNotFoundError:
        print(f"CRITICAL ERROR: Weights not found at {weights_path}")
        sys.exit(1)

    profile = load_profile(args)

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
    print("[+] Guessing...")

    candidates = predict_password(model, tokenizer, profile, beam_schedule=schedule)

    print(f"\n--- TOP GUESSES ---")
    print(f"{'CONFIDENCE':<12} | {'PASSWORD'}")
    print("-" * 30)

    for cand in candidates[:100]: # Show top 100, the rest will be logged in /results
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

    safe_name = profile.get('name', 'target').replace(" ", "_")
    output_path = Config.RESULTS_DIR / f"guesses_{safe_name}.json"

    output_data = []
    for cand in candidates:
        text = tokenizer.decode(cand['sequence'], skip_special_tokens=True)
        
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
