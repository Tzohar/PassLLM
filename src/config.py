import os
import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class Config:
    # =========================================================================
    # 1. DIRECTORY STRUCTURE
    # =========================================================================
    MODELS_DIR = BASE_DIR / "models"
    TRAINING_DIR = BASE_DIR / "training"
    LOGS_DIR = BASE_DIR / "logs"
    RESULTS_DIR = BASE_DIR / "results"
    # CACHE_DIR = BASE_DIR / "cache"

    # =========================================================================
    # 2. FILE PATHS
    # =========================================================================
    # Model Weights
    WEIGHTS_FILE = MODELS_DIR / "PassLLM_LoRA_Weights.pth"
    ADAPTER_CONFIG_FILE = MODELS_DIR / "adapter_config.json"
    
    # Data Files
    RAW_DATA_FILE = TRAINING_DIR / "passllm_raw_data.jsonl"
    PROCESSED_DATA_FILE = TRAINING_DIR / "passllm_processed.pt"
    
    # Logging
    APP_LOG_FILE = LOGS_DIR / "app.log"
    TRAIN_LOG_FILE = LOGS_DIR / "training.log"

    # =========================================================================
    # 3. MODEL ARCHITECTURE & HARDWARE
    # =========================================================================
    # qwen/Qwen2.5-0.5B is excellent for CPU/Consumer GPU, but mistralai/Mistral-7B-v0.1 is ideal and more powerful
    BASE_MODEL_ID = "mistralai/Mistral-7B-v0.1"    

    # Hardware Strategy
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_4BIT = True         # Set False for higher precision if you have >24GB VRAM
    USE_FLASH_ATTN = True  # Set True if on newer GPUs (A100, H100, RTX 3090/4090)
    TORCH_DTYPE = torch.bfloat16 # Use bfloat16 for modern GPUs, float16 for older ones   

    # Reproducibility
    SEED = 42

    # =========================================================================
    # 4. GENERATION ENGINE (INFERENCE)
    # =========================================================================
    MAX_PASSWORD_LENGTH = 16
    MIN_PASSWORD_LENGTH = 4
    EPSILON_END_PROB = 0.4   # Minimum probability for <EOS> to consider password complete
    
    # Beam Search Schedules (Dynamic Beam Widths)
    # [Start Small] -> [Ramp Up] -> [Full Width]
    SCHEDULE_STANDARD = [100, 100, 100, 100, 200, 200, 500, 500, 500, 500] + [1000] * 6
    SCHEDULE_FAST     = [20, 20, 50] + [50] * 13
    # SCHEDULE_FAST     = [1] * 16
    SCHEDULE_DEEP     = [100, 200, 500] + [2000] * 13
    
    # =========================================================================
    # 5 VOCABULARY & CHARACTER CONSTRAINTS
    # =========================================================================
    # LINEAR BIAS SETTINGS (Additive Score)
    # The model adds this number directly to the token's score.
    #  0.0   = Neutral (Normal)
    # -5.0   = Strong Penalty (Avoid unless necessary)
    # -10.0  = Extreme Penalty (Almost impossible)
    # +2.0   = Boost (Encourage this)  
    VOCAB_BIAS_UPPER = 0.0     # Neutral
    VOCAB_BIAS_LOWER = 0.0     # Neutral
    VOCAB_BIAS_DIGITS = 0.0   # Strong penalty against numbers
    VOCAB_BIAS_SYMBOLS = 0.0  # Mild penalty against symbols
    
    # Overrides (applied on TOP of any rules above) 
    # Add specific characters here to whitelist them even if their category is disabled.
    # Example: If you only want Digits + '@' + '!', set Digits=True and WHITELIST="@!"
    # Whitelist still overrides biases (tokens here are always neutral)
    VOCAB_WHITELIST = "" 
    
    # Add specific characters here to strict-block them.
    # Example: Block spaces and quotes to prevent injection attacks or format errors.
    # Recommended default: Block whitespace characters inside passwords
    VOCAB_BLACKLIST = " \t\r\n"

    # =========================================================================
    # 6. TRAINING HYPERPARAMETERS (LoRA)
    # =========================================================================
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 1
    BATCH_SIZE = 2           # Increase if VRAM allows, 6 is the max on free Google Colab 
    GRAD_ACCUMULATION = 32   # Simulates larger batch size (BATCH_SIZE * GRAD_ACCUMULATION = effective batch)
    
    # LoRA Specifics
    LORA_R = 16              # Rank
    LORA_ALPHA = 32          # Scaling
    LORA_DROPOUT = 0.2
    # Target modules for Qwen/Llama architectures
    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # =========================================================================
    # 7. SYSTEM PROMPTS & TEMPLATES
    # =========================================================================
    SYSTEM_PROMPT = (
        "As a targeted password guessing model, your task is to utilize the "
        "provided information to guess the corresponding password."
    )

    @staticmethod
    def get_formatted_input(pii_dict, target_password=None):
        """
        Formats input PII for the model.
        Crucially uses NEWLINES to separate fields to prevent 'comma hallucination'.
        """

        # We use space to hold space for missing data
        schema_defaults = {
            "name": "",
            "birth_year": "",
            "birth_month": "",
            "birth_day": "",
            #"id": "",
            "username": "",
            "email": "",
            "address": "",
            "phone": "",
            #"city": "",
            #"state": "",
            "country": "",
            #"zip": "",
            #"preferred_language": "",
            #"gender": "",
            "sister_pw": "",
        }

        # 1. Start with a copy of the defaults so we don't mutate the original
        final_data = schema_defaults.copy()

        # 2. Clean and merge the inputs
        if pii_dict:
            for k, v in pii_dict.items():
                val = str(v).strip()
                # KEY FIX: Check 'k in schema_defaults' before saving
                if k in schema_defaults and v and val:
                    final_data[k] = val
        
        # FILTER: Remove keys where the value is still an empty string
        final_data = {k: v for k, v in final_data.items() if v}

        # Join with newlines (Iterating over final_data ensures all keys exist)
        aux_str = "\n".join([f"{k}: {v}" for k, v in final_data.items()])
        
        # Construct final string with clear separator
        base_prompt = f"{Config.SYSTEM_PROMPT}\n{aux_str}\n\nPassword: "

        if target_password is not None:
            # TRAINING MODE: We want the model to learn the whole sequence
            return f"{base_prompt}{target_password}"
        else:
            # INFERENCE MODE: We stop right at the trigger so the model completes it
            return base_prompt
    


# =============================================================================
# AUTO-INITIALIZATION
# =============================================================================
# Automatically create the folder structure when this file is imported
for path in [Config.MODELS_DIR, Config.TRAINING_DIR, Config.LOGS_DIR, Config.RESULTS_DIR]:
# for path in [Config.MODELS_DIR, Config.TRAINING_DIR, Config.LOGS_DIR, Config.RESULTS_DIR, Config.CACHE_DIR]:
    path.mkdir(parents=True, exist_ok=True)