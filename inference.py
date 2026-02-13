import torch
from src.generation_engine import dynamic_beam_search
from src.config import Config  
import copy
import random
from tqdm import tqdm


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

    all_candidates = []
    keys = list(target_profile.keys())
    num_runs = getattr(Config, 'INFERENCE_NUM_RUNS', 5)
    target_keep_ratio = getattr(Config, 'INFERENCE_KEEP_RATIO', 0.8)
    start_ratio = (target_keep_ratio * 0.2) + 0.8

    print(f"\n[+] Running {num_runs} inference passes with {target_keep_ratio*100:.0f}% of fields randomly masked each time...")

    pbar = tqdm(range(num_runs), desc="Inference runs", ncols=80)
    pbar.update(0)  # Show bar immediately

    for run_idx in pbar:
        progress = run_idx / max(1, num_runs - 1)
        current_keep_ratio = start_ratio + (target_keep_ratio - start_ratio) * progress

        new_profile = copy.deepcopy(target_profile)

        for main_field, variant_field, prob in Config.field_variations:
            if random.random() < prob:
                if main_field in new_profile and variant_field in new_profile:
                    new_profile[main_field] = new_profile[variant_field]

        for existing_field in list(new_profile.keys()):
            if existing_field not in Config.schema_defaults:
                del new_profile[existing_field]

        total_keys = len(new_profile)
        profile_partial = {}

        for i, (key, value) in enumerate(new_profile.items()):

            if random.random() < current_keep_ratio:
                if isinstance(value, list):
                    filtered_items = []
                    for item in value:
                        if random.random() < current_keep_ratio:
                            filtered_items.append(item)
                    profile_partial[key] = filtered_items
                else:
                    profile_partial[key] = value


        prompt_text = Config.get_formatted_input(profile_partial, tokenizer=tokenizer)
        input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(model.device)
        print(prompt_text)  
        candidates = dynamic_beam_search(
            model=model,
            tokenizer=tokenizer,
            auxiliary_info_ids=input_ids,
            max_depth=max_depth,
            beam_width_schedule=beam_schedule
        )
        all_candidates.extend(candidates)

    seen = set()
    deduped_candidates = []
    for cand in all_candidates:
        pw = cand['password'] 
        if isinstance(pw, str) and pw.strip() and pw not in seen:
            deduped_candidates.append(cand)
            seen.add(pw)
    deduped_candidates.sort(key=lambda x: x.get('score', 0), reverse=True)
    return deduped_candidates