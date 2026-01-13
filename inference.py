import torch
from src.generation_engine import dynamic_beam_search

def predict_password(model, tokenizer, target_profile, max_depth=16, beam_schedule=None):
    """
    Pure logic function. 
    Args:
        model: Loaded LLM
        tokenizer: Loaded Tokenizer
        target_profile: Dict containing {'name', 'born', 'user', 'sister'}
    Returns:
        List of candidate dictionaries
    """
    
    instruction = "As a targeted password guessing model, your task is to utilize the provided information to guess the corresponding password."
    user_details = f"Name: {target_profile['name']}, Born: {target_profile['born']}, User: {target_profile['user']}, SisterPW: {target_profile['sister']}"
    prompt = f"{instruction}\n{user_details}\nPassword: "

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)

    # 3. Default Schedule
    if beam_schedule is None:
        beam_schedule = [10, 50, 100, 100, 100, 200, 500, 500, 500, 500, 1000, 1000, 1000, 1000, 1000, 1000]

    candidates = dynamic_beam_search(
        model=model, 
        tokenizer=tokenizer, 
        auxiliary_info_ids=input_ids, 
        max_depth=max_depth, 
        beam_width_schedule=beam_schedule
    )
    
    return candidates