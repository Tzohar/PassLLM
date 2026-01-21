import torch
from src.generation_engine import dynamic_beam_search
from src.config import Config 

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

    if max_depth is None:
        max_depth = Config.MAX_PASSWORD_LENGTH
        
    if beam_schedule is None:
        beam_schedule = Config.SCHEDULE_STANDARD
    
    prompt_text = Config.get_formatted_input(target_profile)
    print(f"[+] Generated Prompt:\n{prompt_text}\n")

    input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(model.device)
    
    candidates = dynamic_beam_search(
        model=model, 
        tokenizer=tokenizer, 
        auxiliary_info_ids=input_ids, 
        max_depth=max_depth, 
        beam_width_schedule=beam_schedule
    )
    
    return candidates