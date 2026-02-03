import torch
import torch.nn as nn
import math
import os
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_cosine_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm
from src.model import LoRALayer
from src.loader import build_model, inject_lora_layers 
from src.config import Config 

'''
Now that we have injected the new LoRA layers into the model in loader.py, we must tell PyTorch exactly what to update. 
We want to freeze the massive 7B parameters and only train the tiny LoRA parameters, according to conventions described in the paper, which are currently just two small matrices per modified layer.
'''

def freeze_parameters(model):
    '''
    We look through every single variable in the model. Billions.
    If the parameter belongs to a LoRA layer (which we determine according to naming protocols), we want to train it.
    Otherwise, if it does not belong to any LoRA layer - we will skip it.
    It will be frozen for the training process, allowing us to save MASSIVE amounts of resources rather than training parameters for every single layer...
    '''
    print("Freezing Base Model Parameters...")
    frozen_count = 0
    lora_count = 0
    
    for name, param in model.named_parameters():
        if "lora_a" in name or "lora_b" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
   
    return model


def print_trainable_parameters(model):
    ''' 
    A minor "debug" function, giving us an idea of just how much parameters we're going to be training.
    We'll sum the size of all parameters that require gradients, usually between 0.1% and 5% for certain models.
    And we'll also sum the total numbers of parameters available by the model (7 billion for Mistral, 0.5 billion for Qwen).
    '''
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())

    print(
        f"Trainable Parameters: {trainable_params} || " 
        f"All Parameters: {all_params} || "
        f"Percentage: {100 * trainable_params / all_params:.2f}%"
    )

def format_and_mask(sample, tokenizer):
    '''
    Optimizing the loss over the ENTIRE sequence would inevitably expose the model to many potential vulnerabilities.
    We'll restrict the loss calculation only to the PASSWORD portion of the sequence. 
    '''
    full_text = Config.get_formatted_input(
        pii_dict=sample['pii'], 
        target_password=sample['output']
    )
    full_text += tokenizer.eos_token

    encodings = tokenizer(full_text, truncation=True, padding='max_length', max_length=512)

    # Initially, the labels are identical to the input
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    labels = list(input_ids)

    # We re-tokenize JUST the prompt (Instruction + Input) to find its length
    prompt_text = Config.get_formatted_input(
        pii_dict=sample['pii'], 
        target_password=None  
    )
    
    prompt_ids = tokenizer(prompt_text, truncation=True, max_length=512)["input_ids"]
    prompt_len = len(prompt_ids)

    # Setting a label to -100 means "Ignore this"
    # We set all tokens BEFORE the password to -100 so they don't contribute to the loss
    if prompt_len < len(labels):
        labels[:prompt_len] = [-100] * prompt_len

    # attention_mask is 0 by default for padding tokens, so we must set their labels to -100
    for i, mask_val in enumerate(attention_mask):
        if mask_val == 0:
            labels[i] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": encodings["attention_mask"],
        "labels": labels
    }

def prepare_data(tokenizer):
    print("Processing Data with Masking...")

    dataset = load_dataset("json", data_files=str(Config.RAW_DATA_FILE), split="train")

    tokenized_dataset = dataset.map(
        lambda x: format_and_mask(x, tokenizer),
        remove_columns=dataset.column_names  
    )

    # The data initially holds standard Python lists
    # We convert it into PyTorch Tensors 
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Shuffle is set to true because we want to randomize the order of samples to improve training
    return DataLoader(tokenized_dataset, Config.TRAIN_BATCH_SIZE, shuffle=True)

def train_loop(model, tokenizer, dataloader):
    '''
    The training loop established by the paper is pretty standard. 
    We'll implement a warmup gradient for pure stabilization during the early processing steps, as per the paper.
    And we're using an AdamW optimizer, although not specified by the paper.
    '''
    print("Starting Training Loop...")
    global_step = 0
    num_training_steps = (len(dataloader) // Config.GRAD_ACCUMULATION) * Config.NUM_EPOCHS

    # We ignore the frozen parameters by filtering with 'requires_grad'
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr = Config.LEARNING_RATE
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.2 * num_training_steps), 
        num_training_steps=num_training_steps
    )

    model.train()  

    for epoch in range(Config.NUM_EPOCHS): 
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        optimizer.zero_grad()

        for step, batch in enumerate(progress_bar):

            # Moving data back to the model's device
            input_ids = batch['input_ids'].to(model.device)
            mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=mask, labels=labels)

            # Because we passed 'labels' with -100 masking, the model
            # automatically computes the loss only on the password tokens
            loss = outputs.loss / Config.GRAD_ACCUMULATION
            loss.backward()

            if (step + 1) % Config.GRAD_ACCUMULATION == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            current_loss = loss.item() * Config.GRAD_ACCUMULATION
            total_loss += current_loss
            progress_bar.set_postfix(loss=current_loss)
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Complete. Average Loss: {avg_loss}")

    return model

def save_model(model):
    '''
    We don't want to save the whole massive model with billions of parameters again
    We only want to save the parameters named "lora_..."
    '''
    print(f"Saving LoRA Weights to {Config.WEIGHTS_FILE}...")
    lora_state_dict = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    torch.save(lora_state_dict, Config.WEIGHTS_FILE)
    print("Success!")

if __name__ == "__main__":
    # 1. Build
    model, tokenizer = build_model()
    # 2. Inject
    model = inject_lora_layers(model)
    # 3. Freeze
    model = freeze_parameters(model)
    # 4. Verify
    print_trainable_parameters(model)
    # 6. Prepare Data
    dataloader = prepare_data(tokenizer)
    # 7. Train
    model = train_loop(model, tokenizer, dataloader)
    # 8. Save
    save_model(model)
