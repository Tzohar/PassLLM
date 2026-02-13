import torch
import torch.nn.functional as F
import string
from typing import List, Tuple, Dict
from transformers.cache_utils import DynamicCache
from src.config import Config
import math
import copy
import random

# Key: (vocab_size, device_str) -> Value: mask_tensor
_MASK_CACHE = {}

def get_alphanumeric_mask(tokenizer, true_vocab_size, device):
    '''
    The alphanumeric mask erases thousands of tokens from our vocabulary,
    leaving only those that can be used in 99% of password systems.
    Additionally, it is able to boost or reduce the bias of specific character sets,
    adapting to frameworks where passwords must contain different types of characters.
    '''
    bias_key = (Config.VOCAB_BIAS_UPPER, Config.VOCAB_BIAS_LOWER, Config.VOCAB_BIAS_DIGITS, Config.VOCAB_BIAS_SYMBOLS)
    cache_key = (true_vocab_size, str(device), bias_key)
    if cache_key in _MASK_CACHE: return _MASK_CACHE[cache_key]

    s_upper = Config.VOCAB_BIAS_UPPER
    s_lower = Config.VOCAB_BIAS_LOWER
    s_digits = Config.VOCAB_BIAS_DIGITS
    s_symbol = Config.VOCAB_BIAS_SYMBOLS
    
    u_set, l_set = set(string.ascii_uppercase), set(string.ascii_lowercase)
    d_set, s_set = set(string.digits), set(string.punctuation)
    
    whitelist = set(getattr(Config, "VOCAB_WHITELIST", ""))
    blacklist = set(getattr(Config, "VOCAB_BLACKLIST", " \t\r\n"))

    mask = torch.full((true_vocab_size,), float('-inf'), device=device)

    # Loop through EVERY token
    for token_id in range(true_vocab_size):
        text = tokenizer.decode([token_id]).strip()

        # Skip empty or blacklisted tokens
        if not text or any(c in blacklist for c in text): continue

        if text in whitelist:
            mask[token_id] = 0.0
            continue

        if   all(c in u_set for c in text): mask[token_id] = s_upper
        elif all(c in l_set for c in text): mask[token_id] = s_lower
        elif all(c in d_set for c in text): mask[token_id] = s_digits
        elif all(c in s_set for c in text): mask[token_id] = s_symbol

    if tokenizer.eos_token_id is not None:
        mask[tokenizer.eos_token_id] = 0.0

    _MASK_CACHE[cache_key] = mask
    return mask

def dynamic_beam_search(
    model, 
    tokenizer, 
    auxiliary_info_ids: torch.Tensor, 
    max_depth: int = None, 
    beam_width_schedule: List[int] = None, 
    batch_size: int = None, 
    epsilon: float = None,
    score_penalty: float = 0.0
):
    """
    Implements 'Algorithm 2: Dynamic Beam Search' from the paper
    
    This function generates passwords based on user PII (auxiliary_info) 
    while managing GPU memory and filtering irrelevant passwords.

    ARGS:
    -------------------------------
    1. model (M): 
       The trained LLM (PassLLM). We need it to predict the next character.
       
    2. tokenizer (V): 
       Represents the 'Vocabulary'. Limited to 95 ASCII characters 
       plus the EOS (End of Sequence) token. 
       
    3. auxiliary_info_ids (INFO): 
       The encoded PII (Name, Birth Year, Sister Password). 

    4. max_depth (m):
       How long can the password be? (e.g., 16 chars), 'maximum depth of generation'. 
       
    5. beam_width_schedule (K[i]):
       The 'Dynamic Beam Width'. instead of a fixed number (like 1,000), 
       this is a list from config.py: [50, 200, 1000, 1000...].
       It tells us how many guesses to keep at each step 'i'. 
       
    6. batch_size (B):
       The memory safety limit. Candidates are divided into 
       smaller batches processed in parallel to prevent VRAM crashes. 
       
    7. epsilon (ε):
       The 'EOS Threshold'. A sequence is forcibly ended 
       only if the model's predicted Pr(EOS) > ε'. 

    8. score_penalty:
       A small penalty added to the score of candidates at each step,
       to slightly favor earlier candidates in the beam search.
    """ 

    if max_depth is None: max_depth = Config.MAX_PASSWORD_LENGTH
    if beam_width_schedule is None: beam_width_schedule = Config.SCHEDULE_STANDARD
    if batch_size is None: batch_size = getattr(Config, "INFERENCE_BATCH_SIZE", 10)
    if epsilon is None: epsilon = getattr(Config, "EPSILON_END_PROB", 0.01)

    # Tells PyTorch we are in inference mode (saves memory due to no gradients)
    with torch.no_grad():
        outputs = model(
            input_ids=auxiliary_info_ids, 
            use_cache=True # We use use_cache=True to get the 'past_key_values' (The Memory)
        )
        
        # This object contains the mathematical representation  of "Name: John, Born: 1990"
        # We ran it once. We have 1 copy
        # Later, when we have 1,000 beams, they will all point to this single copy in memory
        shared_kv_cache = outputs.past_key_values
        # record how many tokens are in the auxiliary info (PII)
        info_len = auxiliary_info_ids.shape[-1]

        # The model has read the PII, now it predicts the very first letter of the password
        # We only care about the last token's prediction (index -1)
        # But since we have only provided PII, the model's next token is the FIRST char of the password
        next_token_logits = outputs.logits[:, -1, :]

    model_vocab_size = next_token_logits.shape[-1]
    alphanumeric_mask = get_alphanumeric_mask(tokenizer, model_vocab_size, model.device)

    # Move logits to CPU for processing (saves GPU memory and ensures compatibility for AMD users)
    logits_cpu = next_token_logits.cpu()
    logits_cpu = logits_cpu + alphanumeric_mask.cpu()

    # ON CPU
    next_token_probs = F.softmax(logits_cpu, dim=-1)
    current_k = beam_width_schedule[0] if beam_width_schedule else 100
    top_k_probs, top_k_ids = torch.topk(next_token_probs, k=current_k, dim=-1)
        
    # MOVE BACK TO DEVICE: So the next model loop can use them
    top_k_ids = top_k_ids.to(model.device)

    # Each "Beam" is a dictionary tracking:
    # - 'sequence': The password characters generated so far
    # - 'score': The cumulative probability (log probability)
    # - 'finished': Has it hit EOS?
    beams = []
    
    # For starters, we create K beams based on the top-K first characters
    # Each beam is basically a guess starting with one of these characters we found
    # The score is log(probability) to make multiplication easier (sum of logs)
    for i in range(current_k):
        beams.append({
            'sequence': [top_k_ids[0, i].item()],
            'score': torch.log(top_k_probs.cpu()[0, i]).item(), #
            'finished': False
        })
        
    #print(f"Initialized {len(beams)} beams based on PII.")

    final_candidates = []

    for depth in range(max_depth):
        # At depth 0, we might keep 50. At depth 5, we might keep 1000
        # If no schedule is provided, we default to keeping 100 candidates
        current_beam_width = beam_width_schedule[depth] if beam_width_schedule else 100

        # We take our list of 'beams' and only keep the top K best ones
        # Sort by score (highest probability first) and slice
        beams.sort(key=lambda x: x['score'], reverse=True)
        active_beams = beams[:current_beam_width]     

        new_beams = []       

        for i in range(0, len(active_beams), batch_size):
            batch_candidates = active_beams[i : i + batch_size]
            current_batch_size = len(batch_candidates)

            if depth == 0:
                expanded_cache = DynamicCache()
                for layer_idx in range(len(shared_kv_cache)):
                    # shared_kv_cache[layer_idx] is a tuple (Key, Value)
                    k, v = shared_kv_cache[layer_idx] 
                    
                    k_expanded = k.expand(current_batch_size, -1, -1, -1)
                    v_expanded = v.expand(current_batch_size, -1, -1, -1)
                    
                    expanded_cache.update(k_expanded, v_expanded, layer_idx)

            else:
                expanded_cache = DynamicCache()
                num_layers = len(shared_kv_cache) 

                for layer_idx in range(num_layers):
                    # 1. Get Shared PII (Expand 1 -> Batch Size)
                    k_shared, v_shared = shared_kv_cache[layer_idx]
                    k_pii = k_shared.expand(current_batch_size, -1, -1, -1)
                    v_pii = v_shared.expand(current_batch_size, -1, -1, -1)

                    # 2. Get Candidate Passwords (Stack Beam -> Batch Size)
                    k_pwd = torch.cat([b['cache'][layer_idx][0] for b in batch_candidates], dim=0)
                    v_pwd = torch.cat([b['cache'][layer_idx][1] for b in batch_candidates], dim=0)

                    # 3. Concatenate along Sequence Dimension (dim=2)
                    k_combined = torch.cat([k_pii, k_pwd], dim=2)
                    v_combined = torch.cat([v_pii, v_pwd], dim=2)

                    expanded_cache.update(k_combined, v_combined, layer_idx)

            # We feed the LAST character of each candidate
            batch_input_ids = torch.tensor(
                [[b['sequence'][-1]] for b in batch_candidates], 
                device=model.device
            )

            with torch.no_grad():
                outputs = model(
                    input_ids=batch_input_ids,
                    past_key_values=expanded_cache, 
                    use_cache=True
                )

            next_token_logits = outputs.logits[:, -1, :]
            logits_cpu = next_token_logits.cpu()
            logits_cpu = logits_cpu + alphanumeric_mask.cpu()
            next_token_probs_cpu = F.softmax(logits_cpu, dim=-1)

            vocab_probs, vocab_ids = torch.topk(next_token_probs_cpu, k=current_beam_width, dim=-1)

            # MOVE BACK TO DEVICE: So the next model loop can use them
            vocab_probs = vocab_probs.to(model.device)
            vocab_ids = vocab_ids.to(model.device)


            # Check EOS Probability for each candidate in this batch
            eos_id = tokenizer.eos_token_id
            eos_probs = next_token_probs_cpu[:, eos_id] # Shape: (batch_size,)

            # Condition A: The model predicts EOS is likely (Implicit in beam search)
            # Condition B: The probability is explicitly > epsilon (The Paper's Fix)
            has_high_eos_prob = eos_probs > epsilon
            vocab_probs, vocab_ids = torch.topk(next_token_probs_cpu, k=current_beam_width, dim=-1)
            new_kv_cache = outputs.past_key_values

            for batch_idx, candidate in enumerate(batch_candidates):

                # We subtract 1 from len() if the tokenizer adds a start token, 
                # but usually just checking len() >= 8 is safe
                is_long_enough = len(candidate['sequence']) >= getattr(Config, "MIN_PASSWORD_LENGTH", 4)

                # If this specific candidate passes the Epsilon test, we save it as a "Finished Password"
                if has_high_eos_prob[batch_idx] and is_long_enough:
                    finished_candidate = candidate.copy()
                    finished_candidate['finished'] = True
                    finished_candidate['cache'] = None
                    finished_candidate['score'] += torch.log(eos_probs[batch_idx]).item()
                    final_candidates.append(finished_candidate)

                # We create a lightweight tuple structure for the next iteration to read
                # We assume the cache has L layers. We slice the batch_idx for each layer.
                candidate_next_cache = []
                for layer_idx in range(len(new_kv_cache)):
                    k_layer, v_layer = new_kv_cache[layer_idx]
                    
                    # Slice out ONLY this candidate's row (keep dim 0 as size 1)
                    # k_layer shape is (Batch, Heads, Seq, Dim). We want (1, Heads, Seq, Dim)
                    k_slice = k_layer[batch_idx : batch_idx+1, :, info_len:, :].detach().clone()
                    v_slice = v_layer[batch_idx : batch_idx+1, :, info_len:, :].detach().clone()
                    
                    candidate_next_cache.append((k_slice, v_slice))

                # Even if we stopped, we ALSO continue expanding (in case the password is longer)
                # We loop through the top-K likely next characters
                for k in range(current_beam_width):
                    next_char_id = vocab_ids[batch_idx, k].item()
                    next_char_prob = vocab_probs[batch_idx, k].item()
                    
                    if next_char_id == eos_id:
                        continue
                        
                    new_beam = {
                        'sequence': candidate['sequence'] + [next_char_id],
                        'score': candidate['score'] + torch.log(torch.tensor(next_char_prob)).item(),
                        'finished': False,
                        'cache': candidate_next_cache 
                    }
                    new_beams.append(new_beam)

        valid_beams = []
        for b in new_beams:
            decoded_text = tokenizer.decode(b['sequence'], skip_special_tokens=True)
            if len(decoded_text) <= max_depth:
                valid_beams.append(b)
        new_beams = valid_beams

        new_beams.sort(key=lambda x: x['score'], reverse=True)
        next_width = beam_width_schedule[depth+1] if (beam_width_schedule and depth+1 < len(beam_width_schedule)) else 100
        beams = new_beams[:next_width]
        

    if len(final_candidates) == 0 and len(beams) > 0:
        final_candidates = [b for b in beams]

    if len(final_candidates) > 0:
        scores = torch.tensor([c['score'] for c in final_candidates], device=model.device)
    
        scores = scores - score_penalty 
        if Config.NORMALIZE_PROBABILITIES:
            scores = scores - torch.max(scores)
            probs = torch.exp(scores)
            probs = probs / torch.sum(probs)
        else:
            scores = scores / 1.5
            probs = torch.exp(scores)

    for i, c in enumerate(final_candidates):
        new_c = {
            'password': tokenizer.decode(c['sequence'], skip_special_tokens=True),
            'probability': max(probs[i].item() * 100.0, 0.0001), 
            'score': scores[i].item()
        }
        final_candidates[i] = new_c

    final_candidates.sort(key=lambda x: x['score'], reverse=True)
    return final_candidates
