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

    # =========================================================================
    # 4. TRAINING HYPERPARAMETERS (LoRA)
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
    # Note: Must be >= typical password length + typical PII prompt length (~100 tokens, depending on formatting and fields)
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
    # 5. SYSTEM PROMPTS & TEMPLATES
    # =========================================================================
    SYSTEM_PROMPT = (
        "As a targeted password guessing model, your task is to utilize the "
        "provided information to guess the corresponding password."
    )

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
            target_password: Target password for full sequence, None for prompt-only masking
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

# =============================================================================
# AUTO-INITIALIZATION
# =============================================================================
# Automatically create the folder structure when this file is imported
for path in [Config.MODELS_DIR, Config.TRAINING_DIR, Config.LOGS_DIR]:
    path.mkdir(parents=True, exist_ok=True)
    
