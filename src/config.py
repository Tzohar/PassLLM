import os
import torch
import torch_directml
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

    # Three options are available - "cuda" for Nvidia GPUs, "dml" for AMD GPUs, and "cpu" otherwise
    device = "cuda"

    # 4-bit quantization to save VRAM (requires compatible GPU), disable for AMD/CPU-only setups
    # Must be enabled for Nvidia GPUs, otherwise performance will be poor
    USE_4BIT = True         
    TORCH_DTYPE = "bfloat16"

    # Reproducibility
    SEED = 42

    # =========================================================================
    # 4. GENERATION ENGINE (INFERENCE)
    # =========================================================================
    MAX_PASSWORD_LENGTH = 16
    MIN_PASSWORD_LENGTH = 8

    # Minimum probability for <EOS> to consider password complete
    EPSILON_END_PROB = 0.3  

    # Batch size for inference (number of passwords to generate in parallel)
    # Lower values reduce VRAM usage but may slow generation
    # Nvidia GPUs can typically handle higher batch sizes than AMD/CPU, because 4-bit quantization is only supported on Nvidia
    INFERENCE_BATCH_SIZE = 32
    
    # Beam Search Schedules (Dynamic Beam Widths)
    # [Start Small] -> [Ramp Up] -> [Full Width]
    SCHEDULE_STANDARD = [50, 50, 50, 50, 100, 100, 200, 200, 200, 200] + [500] * 6
    SCHEDULE_FAST     = [50, 50, 50] + [50] * 13
    SCHEDULE_SUPERFAST = [20, 20, 20] + [30] * 13
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
    VOCAB_BIAS_UPPER = 0.0     
    VOCAB_BIAS_LOWER = 0.0     
    VOCAB_BIAS_DIGITS = -1.0   
    VOCAB_BIAS_SYMBOLS = -1.0  
    
    # Overrides (applied on TOP of any rules above) 
    # Add specific characters here to whitelist them even if their category is disabled.
    # Example: If you only want Digits + '@' + '!', set Digits=True and WHITELIST="@!"
    # Whitelist still overrides biases (tokens here are always neutral)
    VOCAB_WHITELIST = "" 
    
    # Add specific characters here to strict-block them.
    # Example: Block spaces and quotes to prevent injection attacks or format errors.
    # Recommended default: Block whitespace characters inside passwords
    VOCAB_BLACKLIST = ""

    # =========================================================================
    # 6. TRAINING HYPERPARAMETERS (LoRA)
    # =========================================================================
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 1
    
     # Increase if VRAM allows
    TRAIN_BATCH_SIZE = 2          

     # Simulates larger batch size (BATCH_SIZE * GRAD_ACCUMULATION = effective batch)
    GRAD_ACCUMULATION = 32  
    
    
    # Lora rank
    LORA_R = 16 
    
    # Scaling, higher = more LoRA influence
    LORA_ALPHA = 32

    # Percent of training samples that would render as blank
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
    
# ============================================================================
# DEFAULTS FOR RESETTING CONFIGURATION
# ============================================================================
# --- FACTORY DEFAULTS (DO NOT EDIT MANUALLY) ---
DEFAULT_MIN_PASSWORD_LENGTH = 8
DEFAULT_MAX_PASSWORD_LENGTH = 16
DEFAULT_EPSILON_END_PROB = 0.3
DEFAULT_INFERENCE_BATCH_SIZE = 32
DEFAULT_VOCAB_BIAS_UPPER = 0.0
DEFAULT_VOCAB_BIAS_LOWER = 0.0
DEFAULT_VOCAB_BIAS_DIGITS = -1.0
DEFAULT_VOCAB_BIAS_SYMBOLS = -1.0
DEFAULT_VOCAB_WHITELIST = ""
DEFAULT_VOCAB_BLACKLIST = ""
DEFAULT_DEVICE = "cuda"
DEFAULT_TORCH_DTYPE = "bfloat16"
DEFAULT_USE_4BIT = True
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_SCHEDULE_STANDARD = [50, 50, 50, 50, 100, 100, 200, 200, 200, 200] + [500] * 6
DEFAULT_SCHEDULE_FAST     = [50, 50, 50] + [50] * 13
DEFAULT_SCHEDULE_SUPERFAST = [20, 20, 20] + [30] * 13
DEFAULT_SCHEDULE_DEEP     = [100, 200, 500] + [2000] * 13

# =============================================================================
# AUTO-INITIALIZATION
# =============================================================================
# Automatically create the folder structure when this file is imported
for path in [Config.MODELS_DIR, Config.TRAINING_DIR, Config.LOGS_DIR, Config.RESULTS_DIR]:
# for path in [Config.MODELS_DIR, Config.TRAINING_DIR, Config.LOGS_DIR, Config.RESULTS_DIR, Config.CACHE_DIR]:
    path.mkdir(parents=True, exist_ok=True)
    
