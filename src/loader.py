import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.model import LoRALayer  
from src.config import Config

def build_model():
    print(f"Loading Base Model: {Config.BASE_MODEL_ID}...")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    dtype_str = getattr(Config, "TORCH_DTYPE", "float16")
    target_dtype = dtype_map.get(dtype_str, torch.float16)

    # Configure Quantization based on Config
    # If Config.USE_4BIT is True, we use the modern BitsAndBytes config object
    # We MUST ensure that 4-bit quantization is only used on compatible devices (e.g., NVIDIA GPUs)
    if Config.USE_4BIT and Config.DEVICE == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=target_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        quantization_config = None

    # We load the model from HuggingFace with the specified device and dtype
    # Use device_map="auto" to load directly to GPU (avoids CPU RAM OOM on large models)
    # For DirectML (Windows AMD), we must load to CPU first then move manually
    if Config.DEVICE == "dml":
        device_map = None
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        Config.BASE_MODEL_ID,
        device_map=device_map,
        torch_dtype=target_dtype,
        quantization_config=quantization_config
    )

    tokenizer = AutoTokenizer.from_pretrained(Config.BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Move model to the correct device (only needed for DirectML, others use device_map="auto")
    if Config.DEVICE == "dml":
        import torch_directml
        device = torch_directml.device()
        print(f"Moving model to DirectML (AMD GPU)")
        model.to(device)
    
    return model, tokenizer

def inject_lora_layers(model):
    count = 0
    target_modules = set(Config.LORA_TARGET_MODULES)

    for name, module in model.named_modules():
        module_suffix = name.split('.')[-1]

        if module_suffix in target_modules:
            parent_name = name.rsplit('.', 1)[0]
            child_name = name.rsplit('.', 1)[1]

            parent_module = model.get_submodule(parent_name)

            lora_layer = LoRALayer(
                original_layer=module, 
                rank=Config.LORA_R,
                alpha=Config.LORA_ALPHA
            )

            lora_layer.to(module.weight.device)

            setattr(parent_module, child_name, lora_layer)
            count += 1
            
    return model
