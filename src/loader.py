import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.model import LoRALayer  # We import the LoRA class logic
from src.config import Config

# First, we load the base pre-trained model
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
    model = AutoModelForCausalLM.from_pretrained(
        Config.BASE_MODEL_ID,
        device_map=None, # We will move it to the correct device later
        dtype=target_dtype,
        quantization_config=quantization_config,
    )

    # This loads the dictionary that turns "password" into numbers
    tokenizer = AutoTokenizer.from_pretrained(Config.BASE_MODEL_ID)

    # Technical Fix: Mistral/Llama don't have a "padding" token by default
    # We set it to EOS (End of Sequence) so training doesn't crash
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

# Now that we have the base model, we inject LoRA layers into it
# LoRA layers are tiny modules that allow us to fine-tune massive models efficiently

def inject_lora_layers(model):
    count = 0
    print(f"Injecting LoRA Layers (Rank={Config.LORA_R}, Alpha={Config.LORA_ALPHA})...")

    # Get the list of modules we want to target from Config
    # e.g. ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    target_modules = set(Config.LORA_TARGET_MODULES)

    # We shall go through each layer in the model
    # Neural networks are like trees - this function walks through every branch
    for name, module in model.named_modules():

        # Check if the module name ends with one of our targets (e.g. "...self_attn.q_proj")
        # name.split('.')[-1] gets the last part: "q_proj"
        module_suffix = name.split('.')[-1]

        if module_suffix in target_modules:

            # Again, neural networks are like trees
            # A layer is named according to its position in the tree, in the format:
            # parent_module.child_module
            parent_name = name.rsplit('.', 1)[0]
            child_name = name.rsplit('.', 1)[1]

            # get_submodule is a PyTorch function that retrieves a layer given its name
            parent_module = model.get_submodule(parent_name)

            # We replace the original layer with a LoRA-wrapped layer
            # We use rank=16 and alpha=32 as per the LoRA paper's recommendation
            lora_layer = LoRALayer(
                original_layer=module, 
                rank=Config.LORA_R,
                alpha=Config.LORA_ALPHA,
                dropout=Config.LORA_DROPOUT
            )

            # We move the new LoRA layer to the same device as the original layer   
            lora_layer.to(module.weight.device)

            # This line does the actual replacement in the model
            setattr(parent_module, child_name, lora_layer)
            count += 1

    print(f"Success! Injected LoRA into {count} layers matching: {target_modules}")       
            
    return model
