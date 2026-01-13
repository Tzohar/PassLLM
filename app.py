import argparse
import torch
import sys
from src.loader import build_model, inject_lora_layers 
from inference import predict_password # Import logic

def parse_arguments():
    parser = argparse.ArgumentParser(description="PassLLM Command Line Tool")
    
    # --- USER INPUTS ---
    parser.add_argument("--name", type=str, required=True, help="Target's Name")
    parser.add_argument("--born", type=str, required=True, help="Target's Birth Year")
    parser.add_argument("--user", type=str, required=True, help="Target's Username")
    parser.add_argument("--sister", type=str, required=True, help="Target's Sister Password")
    
    # --- CONFIGURATION ---
    parser.add_argument("--weights", type=str, default="models/PassLLM_LoRA_Weights.pth", help="Path to model weights")
    parser.add_argument("--fast", action="store_true", help="Use fast mode (less accurate, quicker)")

    return parser.parse_args()

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
    profile = {
        "name": args.name,
        "born": args.born,
        "user": args.user,
        "sister": args.sister
    }

    # Define schedule based on flags
    schedule = [10, 20, 50] + [50]*13 if args.fast else None # Pass None to use default

    print(f"\n[+] Target Locked: {args.name}")
    print("[+] Cracking...")

    # CALL THE LOGIC FILE
    candidates = predict_password(model, tokenizer, profile, beam_schedule=schedule)

    print(f"\n--- TOP GUESSES ---")
    print(f"{'CONFIDENCE':<12} | {'PASSWORD'}")
    print("-" * 30)

    for cand in candidates[:15]: # Show top 15
        pwd = tokenizer.decode(cand['sequence'], skip_special_tokens=True)
        prob = torch.exp(torch.tensor(cand['score'])).item()
        
        if prob > 0.001:
            print(f"{prob:.2%}       | {pwd}")
        else:
            print(f"{cand['score']:.4f} (log) | {pwd}")

if __name__ == "__main__":
    main()