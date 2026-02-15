import os
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
    # Qwen/Qwen3-4B-Instruct-2507
    BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"    

    # Three options are available - "cuda" for Nvidia GPUs, "dml" for AMD GPUs, and "cpu" otherwise
    DEVICE = "cuda"

    # 4-bit quantization to save VRAM (requires compatible GPU), disable for AMD/CPU-only setups
    # Must be enabled for Nvidia GPUs, otherwise performance will be poor
    USE_4BIT = False         
    TORCH_DTYPE = "bfloat16"

    # Reproducibility
    SEED = 42

    # If true, shifts scores for stability, then ensures they sum to 1 (100%)
    # Useful if you want to know "Which of THESE candidates is best?"
    # If false, strictly converts log-space to probability-space: p = e^(log_p)
    # These will likely be very small numbers and will NOT sum to 100%, 
    # But will more accurately reflect the model's thinking process
    NORMALIZE_PROBABILITIES = False

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
    INFERENCE_BATCH_SIZE = 64
    
    # Beam Search Schedules (Dynamic Beam Widths)
    # [Start Small] -> [Ramp Up] -> [Full Width]
    SCHEDULE_SUPERFAST = [50, 20, 20, 30, 30, 30, 30, 30, 30, 30, 30] + [50] * 100
    SCHEDULE_FAST     = [100, 50, 50, 50, 50, 50, 100, 100, 100, 100, 100] + [200] * 100
    SCHEDULE_STANDARD = [200, 100, 100, 100, 100, 100, 200, 200, 200, 200, 200, 200] + [500] * 100
    SCHEDULE_DEEP     = [500, 200, 200, 500, 500, 1000, 1000, 1000, 1000] + [2000] * 1000

    # Number of inference runs with random field dropout
    # Multiple runs with different random subsets of PII can help generate more diverse candidates, but will increase inference time.
    INFERENCE_NUM_RUNS = 50

    # Fraction of fields to keep per run (gradually increases randomness and diversity, but may reduce accuracy if too low)
    INFERENCE_KEEP_RATIO = 0.5
    
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
    VOCAB_BIAS_DIGITS = 0.0   
    VOCAB_BIAS_SYMBOLS = 0.0
    
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
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 3
    
    # Reduce for lower VRAM (16GB AMD GPU without 4-bit)
    TRAIN_BATCH_SIZE = 4     

    # Simulates larger batch size (BATCH_SIZE * GRAD_ACCUMULATION = effective batch)
    GRAD_ACCUMULATION = 16
    
    # Max sequence length for training (reduce to save VRAM)
    # Recommended values:
    #   512  = Full context, best quality (24GB+ VRAM)
    #   256  = Good balance for 16GB GPUs (default)
    #   128  = Minimum for password data, may truncate long PII inputs (12GB or less)
    # Note: Must be >= MAX_PASSWORD_LENGTH + typical PII prompt length (~100 tokens, depending on formatting and fields)
    MAX_SEQ_LENGTH = 256  # Passwords are short; 256 is plenty for PII + password
    
    # Enable gradient checkpointing to save VRAM (trades compute for memory)
    # True = Saves ~3-5GB VRAM, but ~20-30% slower training (use for <=16GB GPUs)
    # False = Faster training, but requires more VRAM (use for 24GB+ GPUs)
    USE_GRADIENT_CHECKPOINTING = True  
    
    # Lora rank
    LORA_R = 16
    
    # Scaling, higher = more LoRA influence
    LORA_ALPHA = 32

    # Percent of training samples that would render as blank
    LORA_DROPOUT = 0.2
    
    # Target modules for Qwen/Llama architectures
    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # Checkpoint frequency during training
    # 0 = Only save at end of each epoch (default, recommended for small datasets)
    # N = Save every N optimizer steps (useful for long training runs)
    # Checkpoints are saved to models/checkpoints/ and only the last 3 are kept
    CHECKPOINT_EVERY_STEPS = 100

    # =========================================================================
    # 7. SYSTEM PROMPTS & TEMPLATES
    # =========================================================================
    SYSTEM_PROMPT = (
        "As a targeted password guessing model, your task is to utilize the "
        "provided information to guess the corresponding password."
    )

    # Table of field variations and their substitution probabilities
    # Format: (main_field, variant_field, probability for main_field to be replaced by variant_field) 
    field_variations = [
        ("name", "pet_name", 0.05),
        ("name", "partner_name", 0.1),
        ("address", "work_address", 0.4),
        ("email", "work_email", 0.4),
        ("phone", "id", 0.3),
    ]

    schema_defaults = {
        "name": "",
        "birth_year": "",
        "birth_month": "",
        "birth_day": "",
        "username": "",
        "email": "",
        "address": "",
        "phone": "",
        "city": "",
        #"state": "",
        "country": "",
        #"zip": "",
        #"preferred_language": "",
        #"gender": "",
        "sister_pw": ""
    }

    @staticmethod
    def get_formatted_input(pii_dict, target_password=None, tokenizer=None):
        """
        Formats input PII for Qwen3 using chat template.
        
        Args:
            pii_dict: Dictionary of PII fields
            target_password: Password for training, None for inference
            tokenizer: Required for specific models to apply chat template
            
        Returns: Formatted text string ready for tokenization
        """
        final_data = Config.schema_defaults.copy()

        if pii_dict:
            for k, v in pii_dict.items():
                if k in Config.schema_defaults and v:
                    if isinstance(v, (list, tuple)):
                        val = ", ".join(str(item).strip() for item in v if item and str(item).strip())
                    else:
                        val = str(v).strip()
                    if val:
                        final_data[k] = val
        
        final_data = {k: v for k, v in final_data.items() if v}

        aux_str = "\n".join([f"{k}: {v}" for k, v in final_data.items()])
        user_content = f"{aux_str}\n\nPassword:"
        
        if target_password is not None:
            messages = [
                {"role": "system", "content": Config.SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": target_password}
            ]
        else:
            messages = [
                {"role": "system", "content": Config.SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]
        
        result = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=(target_password is None)
        )
        return result
    
# ============================================================================
# DEFAULTS FOR RESETTING CONFIGURATION
# ============================================================================
# --- FACTORY DEFAULTS (DO NOT EDIT MANUALLY) ---
DEFAULT_MIN_PASSWORD_LENGTH = 8
DEFAULT_MAX_PASSWORD_LENGTH = 16
DEFAULT_EPSILON_END_PROB = 0.3
DEFAULT_INFERENCE_BATCH_SIZE = 64
DEFAULT_VOCAB_BIAS_UPPER = 0.0
DEFAULT_VOCAB_BIAS_LOWER = 0.0
DEFAULT_VOCAB_BIAS_DIGITS = 0.0
DEFAULT_VOCAB_BIAS_SYMBOLS = 0.0
DEFAULT_VOCAB_WHITELIST = ""
DEFAULT_VOCAB_BLACKLIST = ""
DEFAULT_DEVICE = "cuda"
DEFAULT_TORCH_DTYPE = "bfloat16"
DEFAULT_USE_4BIT = False
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_CHECKPOINT_EVERY_STEPS = 100
DEFAULT_USE_GRADIENT_CHECKPOINTING = True
DEFAULT_INFERENCE_KEEP_RATIO = 0.5
DEFAULT_NORMALIZE_PROBABILITIES = False
DEFAULT_INFERENCE_NUM_RUNS = 50
# =============================================================================
# AUTO-INITIALIZATION
# =============================================================================
# Automatically create the folder structure when this file is imported
for path in [Config.MODELS_DIR, Config.TRAINING_DIR, Config.LOGS_DIR, Config.RESULTS_DIR]:
# for path in [Config.MODELS_DIR, Config.TRAINING_DIR, Config.LOGS_DIR, Config.RESULTS_DIR, Config.CACHE_DIR]:
    path.mkdir(parents=True, exist_ok=True)
    
