import torch
import torch.nn as nn
import math
from transformers import AutoModelForCasualLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from model import LoRALayer 
from tqdm import tqdm

# --- CONFIGURATION ---
MODEL_ID = "mistralai/Mistral-7B-v0.1"
DATA_FILE = "passllm_aplaca_data.jsonl"
LORA_RANK = 16
LORA_ALPHA = 32
LEARNING_RATE = 1e-4

# First, we load the base pre-trained model
def build_model():
    print("Loading Base Model...")

    # We load the model with "load_in_4bit=True" to save memory
    # It compresses the 14GB model to around 4GB so it can fit in limited GPU memory
    model = AutoModelForCasualLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        load_in_4bit=True
    )

    # This loads the dictionary that turns "password" into numbers
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Technical Fix: Mistral/Llama don't have a "padding" token by default
    # We set it to EOS (End of Sequence) so training doesn't crash
    tokenizer.pad_token = tokenizer.eos_token

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

            # This line does the actual replacement in the model
            setattr(parent_module, child_name, lora_layer)

            print(f"Replaced: {name} with LoRA Layer.")

# Now that we have injected the new LoRA layers, 
# we must tell PyTorch exactly what to update
# We want to freeze the massive 7B parameters and only train the tiny LoRA parameters
# Which are currently just two small matrices per modified layer
def freeze_parameters(model):
    print("Freezing Base Model Parameters...")

    # We look at every single variable in the model (billions)
    for name, param in model.named_parameters():
        # If the parameter belongs to a LoRA layer, we want to train it
        if "lora_a" in name or "lora_b" in name:
            param.requires_grad = True
        else:
            # Otherwise, we freeze it to save memory and computation
            param.requires_grad = False
        # We are skipping the math for 99.9% of parameters!
   
    return model


# We need to confirm that our "Freezing Logic" worked correctly
def print_trainable_parameters(model):

    # We can sum the size (numel) of all parameters that require gradients (about 0.1% of total)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # We can also sum the total number of parameters (about 7 Billion for Mistral-7B)
    all_params = sum(p.numel() for p in model.parameters())

    # We print out the results
    print(
        f"Trainable Parameters: {trainable_params} || " 
        f"All Parameters: {all_params} || "
        f"Percentage: {100 * trainable_params / all_params:.2f}%"
    )

# Optimizing the loss over the entire sequence could expose the model to
# potential vulnerabilities, so we restrict the loss calculation to the 
# password portion of the sequence only
def format_and_mask(sample, tokenizer):

    # We combine the instruction, the user info and the real password together
    full_text = f"{sample['instruction']}\n{sample['input']}\nPassword: {sample['output']}
    
    # Convert the full text into token IDs
    encodings = tokenizer(full_text, truncation=True, padding='max_length', max_length=512)

    # Initially, the labels are identical to the input
    input_ids = torch.tensor(encodings['input_ids'])
    labels = input_ids.clone()

    # We re-tokenize JUST the prompt (Instruction + Input) to find its length
    prompt_text = f"{sample['instruction']}\n{sample['input']}\n"
    prompt_len = len(tokenizer(prompt_text, truncation=True, max_length=512)["input_ids"])

    # In PyTorch, setting a label to -100 means "Ignore this"
    # We set all tokens BEFORE the password to -100 so they don't contribute to the loss
    labels[:prompt_len] = -100

def prepare_data(tokenizer):
    print("Processing Data with Masking...")

    # This read the JSONL file full of existing PII and passwords
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")

    # We run the 'format_and_mask' function on every sample in the dataset
    tokenized_dataset = dataset.map(lambda x: format_and_mask(x, tokenizer))

    # The data initially holds standard Python lists
    # We convert it into PyTorch Tensors 
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # bath_size=1: We process one sample at a time to save memory
    # Shuffle=True: We randomize the order of samples to improve training, SGD style
    return DataLoader(tokenized_dataset, batch_size=1, shuffle=True)

# Training Loop
def train_loop(model, tokenizer, dataloader):
    print("Starting Training Loop...")

    # We use AdamW optimizer, which is standard for training transformers
    # We ignore the frozen parameters by filtering with 'requires_grad'
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr = LEARNING_RATE
    )

    # Set model to training mode to enable Dropout layers
    model.train()  

    # The main training loop
    # The paper suggests several epochs, but for demo purposes we do just 1
    for epoch in range(1): 
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        for batch in progress_bar:

            # Moving data to the GPU
            input_ids = batch['input_ids'].to(model.device)
            mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)

            # The model reads the inputs and predicts the next token.
            outputs = model(input_ids=input_ids, attention_mask=mask, labels=labels)

            # Because we passed 'labels' with -100 masking, the model
            # automatically computes the loss only on the password tokens
            loss = outputs.loss

            # The learning
            loss.backward()  # Backpropagation
            optimizer.step() # Update LoRA parameters
            optimizer.zero_grad() # Clear gradients for next step

            # Updating the stats
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        print(f"Epoch {epoch+1} Complete. Average Loss: {total_loss / len(dataloader)}")
    
    return model

# Now we finally save the fine-tuned model
def save_model(model):
    print("Saving LoRA Weights...")

    # We don't want to save the whole 14GB model again
    # We only want to save the parameters named "lora_..."
    lora_state_dict = {k: v for k, v in model.state_dict().items() if "lora_" in k}

    torch.save(lora_state_dict, "PassLLM_LoRA_Weights.pth")
    print("Success! Saved to PassLLM_LoRA_Weights.pth")

# Our main function that ties everything together, the logical flow

if __name__ == "__main__":
    # 1. Build
    model, tokenizer = build_model()
    # 2. Inject
    model = inject_lora_layers(model)
    # 3. Freeze
    model = freeze_parameters(model)
    # 4. Verify
    print_trainable_parameters(model)
    # 5. Prepare Data
    dataloader = prepare_data(tokenizer)
    # 6. Train
    model = train_loop(model, tokenizer, dataloader)
    # 7. Save
    save_model(model)