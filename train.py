import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from src.model import LoRALayer
from src.loader import build_model, inject_lora_layers 

# --- CONFIGURATION ---
DATA_FILE = "passllm_alpaca_data.jsonl"
LEARNING_RATE = 1e-4

# Now that we have injected the new LoRA layers into the model in loader.py, 
# we must tell PyTorch exactly what to update
# We want to freeze the massive 7B parameters and only train the tiny LoRA parameters
# Which are currently just two small matrices per modified layer
def freeze_parameters(model):
    print("Freezing Base Model Parameters...")

    frozen_count = 0
    lora_count = 0
    
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
    full_text = f"{sample['instruction']}\n{sample['input']}\nPassword: {sample['output']}"
    
    # Convert the full text into token IDs
    encodings = tokenizer(full_text, truncation=True, padding='max_length', max_length=512)

    # Initially, the labels are identical to the input
    input_ids = encodings['input_ids']
    labels = list(input_ids)

    # We re-tokenize JUST the prompt (Instruction + Input) to find its length
    prompt_text = f"{sample['instruction']}\n{sample['input']}\n"
    prompt_len = len(tokenizer(prompt_text, truncation=True, max_length=512)["input_ids"])

    # In PyTorch, setting a label to -100 means "Ignore this"
    # We set all tokens BEFORE the password to -100 so they don't contribute to the loss
    if prompt_len < len(labels):
        labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": input_ids,
        "attention_mask": encodings["attention_mask"],
        "labels": labels
    }

def prepare_data(tokenizer):
    print("Processing Data with Masking...")

    # This read the JSONL file full of existing PII and passwords
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")

    # We run the 'format_and_mask' function on every sample in the dataset
    tokenized_dataset = dataset.map(
        lambda x: format_and_mask(x, tokenizer),
        remove_columns=dataset.column_names  
    )

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