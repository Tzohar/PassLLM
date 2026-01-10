import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.model import LoRALayer  # We import the LoRA class logic

# --- CONFIGURATION ---
# We centralize the config here so both Train/Inference always match
MODEL_ID = "mistralai/Mistral-7B-v0.1"
LORA_RANK = 16
LORA_ALPHA = 32

# First, we load the base pre-trained model
def build_model():
    print("Loading Base Model...")

    # We load the model with "load_in_4bit=True" to save memory
    # It compresses the 14GB model to around 4GB so it can fit in limited GPU memory
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        load_in_4bit=True
    )

    # This loads the dictionary that turns "password" into numbers
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Technical Fix: Mistral/Llama don't have a "padding" token by default
    # We set it to EOS (End of Sequence) so training doesn't crash
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

# Now that we have the base model, we inject LoRA layers into it
# LoRA layers are tiny modules that allow us to fine-tune massive models efficiently
def inject_lora_layers(model):
    print("Injecting LoRA Layers...")

    # We shall go through each layer in the model
    # Neural networks are like trees - this function walks through every branch
    for name, module in model.named_modules():

        # Attention is calculated as Attention(Q, K, V)
        # We only want to modify the Query and Value projection layers in the attention mechanism
        if "q_proj" in name or "v_proj" in name:

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
                original_layer=module, # module = parent_module.__getattr__(child_name)
                rank=LORA_RANK,
                alpha=LORA_ALPHA
            )

            # We move the new LoRA layer to the same device as the original layer   
            lora_layer.to(module.weight.device)

            # This line does the actual replacement in the model
            setattr(parent_module, child_name, lora_layer)

            print(f"Replaced: {name} with LoRA Layer.")
            
    return model