import torch
import torch.nn.functional as F
import string
from typing import List, Tuple, Dict
from transformers.cache_utils import DynamicCache
from src.config import Config

# Key: (vocab_size, device_str) -> Value: mask_tensor
_MASK_CACHE = {}

def get_alphanumeric_mask(tokenizer, true_vocab_size, device):
    cache_key = (true_vocab_size, str(device))
    if cache_key in _MASK_CACHE: return _MASK_CACHE[cache_key]

    print(f"[System] Scanning vocabulary ({true_vocab_size} tokens) for valid characters...")

    allowed = set()
    if getattr(Config, "VOCAB_CUSTOM_ALLOW_UPPER", True): allowed.update(string.ascii_uppercase)
    if getattr(Config, "VOCAB_CUSTOM_ALLOW_LOWER", True): allowed.update(string.ascii_lowercase)
    if getattr(Config, "VOCAB_CUSTOM_ALLOW_DIGITS", True): allowed.update(string.digits)
    if getattr(Config, "VOCAB_CUSTOM_ALLOW_SYMBOLS", True): allowed.update(string.punctuation)

    if getattr(Config, "VOCAB_WHITELIST", ""):
        allowed.update(Config.VOCAB_WHITELIST)

    blacklist = getattr(Config, "VOCAB_BLACKLIST", " \t\r\n")
    if blacklist:
        allowed.difference_update(blacklist)
    
    mask = torch.full((true_vocab_size,), float('-inf'), device=device)

    # Loop through EVERY token in the vocabulary
    for token_id in range(true_vocab_size):
        # Decode the token to see what text it represents
        # .strip() removes the invisible 'start of word' spaces common in Llama/Qwen
        text = tokenizer.decode([token_id]).strip()

        # If text is not empty AND contains ONLY allowed characters -> Allow it
        if text and all(char in allowed for char in text):
            mask[token_id] = 0.0

    # Always allow End-of-Sequence (so it can stop)
    if tokenizer.eos_token_id is not None:
        mask[tokenizer.eos_token_id] = 0.0

    print(mask)
    _MASK_CACHE[cache_key] = mask
    return mask



def dynamic_beam_search(
    model, 
    tokenizer, 
    auxiliary_info_ids: torch.Tensor, 
    max_depth: int = None, 
    beam_width_schedule: List[int] = None, 
    batch_size: int = None, 
    epsilon: float = None
):
    """
    Implements 'Algorithm 2: Dynamic Beam Search' from the paper
    
    This function generates passwords based on user PII (auxiliary_info) 
    while managing GPU memory and filtering low-quality 'junk' passwords.

    ARGS):
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
       this is a list: [50, 200, 1000, 1000...].
       It tells us how many guesses to keep at each step 'i'. 
       
    6. batch_size (B):
       The memory safety limit. Candidates are divided into 
       smaller batches processed in parallel to prevent VRAM crashes. 
       
    7. epsilon (ε):
       The 'EOS Threshold'. A sequence is forcibly ended 
       only if the model's predicted Pr(EOS) > ε'. 
    """
    # We use None defaults to ensure we always read the latest Config values
    if max_depth is None: max_depth = Config.MAX_PASSWORD_LENGTH
    if beam_width_schedule is None: beam_width_schedule = Config.SCHEDULE_STANDARD
    if batch_size is None: batch_size = getattr(Config, "BATCH_SIZE", 10)
    if epsilon is None: epsilon = getattr(Config, "EPSILON_END_PROB", 0.01)

    # --- STEP 1: THE KV CACHE OPTIMIZATION ---
    # Tells PyTorch we are in inference mode (saves memory due to no gradients)
    with torch.no_grad():
        
        # Feed the auxiliary info (PII) into the model ONCE
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

    # --- STEP 2: INITIALIZE CANDIDATES ---
    # We started with a single empty candidate (the beginning of the password)
    # We will limit the model to generating only 95 ASCII characters
    # We force the probability of all other characters (emojis, chinese, etc.) to -Infinity


    # We generate the ASCII mask and apply it
    model_vocab_size = next_token_logits.shape[-1]
    alphanumeric_mask = get_alphanumeric_mask(tokenizer, model_vocab_size, model.device)
    next_token_logits = next_token_logits + alphanumeric_mask

    # Turn raw scores (logits) into probabilities (0.0 to 1.0)
    next_token_probs = F.softmax(next_token_logits, dim=-1)

    # For the first step (depth=0), we use K[0].
    current_k = beam_width_schedule[0] if beam_width_schedule else 100

    # Get the top-K tokens and their probabilities.
    top_k_probs, top_k_ids = torch.topk(next_token_probs, k=current_k, dim=-1)
        
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
            'sequence': [top_k_ids[0, i].item()], # The first character ID
            'score': torch.log(top_k_probs[0, i]).item(), # Log probability
            'finished': False
        })
        
    print(f"Initialized {len(beams)} beams based on PII.")

    # --- STEP 3: THE MAIN SEARCH LOOP ---
    ## This is where the code loops from depth=1 to max_depth (e.g., 16 characters),
    ## generating one character at a time

    final_candidates = []

    for depth in range(max_depth):
        print(f"Depth {depth+1}/{max_depth} with {len(beams)} beams.")

        # A. DYNAMIC BEAM WIDTH 
        # At depth 0, we might keep 50. At depth 5, we might keep 1000
        # If no schedule is provided, we default to keeping 100 candidates
        current_beam_width = beam_width_schedule[depth] if beam_width_schedule else 100

        # B. PREPARE NEXT STEP CANDIDATES
        # We take our list of 'beams' and only keep the top K best ones
        # Sort by score (highest probability first) and slice
        beams.sort(key=lambda x: x['score'], reverse=True)
        active_beams = beams[:current_beam_width]     

        new_beams = []       

        # --- STEP 4: THE BATCHING MECHANISM (Solving VRAM Crashes) ---
        # This is the "Safety Valve". If active_beams has 1,000 candidates, 
        # and batch_size is 10, we process 100 loops of 10
        # This prevents the GPU memory from spiking

        for i in range(0, len(active_beams), batch_size):
            batch_candidates = active_beams[i : i + batch_size]
            current_batch_size = len(batch_candidates)

            # 1. Create a new cache object for this batch
            # --- FIX: PREPARE CACHE BASED ON DEPTH ---
            if depth == 0:
                # CASE A: DEPTH 0 (First Generation)
                # We only have PII. Every candidate shares the exact same PII memory.
                # We simply expand the 'shared_kv_cache' N times (N = current_batch_size).
                
                expanded_cache = DynamicCache()
                for layer_idx in range(len(shared_kv_cache)):
                    # shared_kv_cache[layer_idx] is a tuple (Key, Value)
                    k, v = shared_kv_cache[layer_idx] 
                    
                    # Expand from Batch=1 to Batch=Current_Batch_Size
                    k_expanded = k.expand(current_batch_size, -1, -1, -1)
                    v_expanded = v.expand(current_batch_size, -1, -1, -1)
                    
                    expanded_cache.update(k_expanded, v_expanded, layer_idx)

            else:
                # CASE B: DEPTH > 0 (Subsequent Generations)
                # The candidates have diverged. Candidate A has memory "Joh", Candidate B has "199".
                # We must GATHER their individual caches and STACK them into a single batch.
                
                expanded_cache = DynamicCache()
                
                # We assume all candidates have the same number of layers
                # We look at the first candidate to know how many layers to loop over
                num_layers = len(batch_candidates[0]['cache']) 
                
                for layer_idx in range(num_layers):
                    # 1. Collect the Keys (K) for this layer from all candidates in this batch
                    # candidate['cache'][layer_idx][0] is the Key tensor for that specific candidate
                    # We use torch.cat to stack them along dimension 0 (Batch Dimension)
                    k_stacked = torch.cat(
                        [b['cache'][layer_idx][0] for b in batch_candidates], 
                        dim=0
                    )
                    
                    # 2. Collect the Values (V) similarly
                    v_stacked = torch.cat(
                        [b['cache'][layer_idx][1] for b in batch_candidates], 
                        dim=0
                    )
                    
                    # 3. Add this combined layer to our batch cache
                    expanded_cache.update(k_stacked, v_stacked, layer_idx)


            # --- PREPARE INPUT ---
            # We feed the LAST character of each candidate
            batch_input_ids = torch.tensor(
                [[b['sequence'][-1]] for b in batch_candidates], 
                device=model.device
            )

            # --- RUN MODEL ---
            with torch.no_grad():
                outputs = model(
                    input_ids=batch_input_ids,
                    past_key_values=expanded_cache, # Passing the expanded memory
                    use_cache=True
                )

            # Get the probabilities for the next token, we apply our mask again
            next_token_logits = outputs.logits[:, -1, :]
            next_token_logits = next_token_logits + alphanumeric_mask
            next_token_probs = F.softmax(next_token_logits, dim=-1)

            # --- STEP 5: THE EPSILON THRESHOLD ---

            # Check EOS Probability for each candidate in this batch
            # We assume tokenizer.eos_token_id is the ID for "End of Sequence"
            # eos_probs is basically Pr(EOS) for each candidate, forming a vector of size (batch_size,)
            eos_id = tokenizer.eos_token_id
            eos_probs = next_token_probs[:, eos_id] # Shape: (batch_size,)

            # Identify which candidates WANT to stop and ARE ALLOWED to stop
            # Condition A: The model predicts EOS is likely (Implicit in beam search)
            # Condition B: The probability is explicitly > epsilon (The Paper's Fix)
            # We create a boolean mask for candidates that meet the threshold
            has_high_eos_prob = eos_probs > epsilon
            # Process the Top-K Predictions
            # We take the top beam_width characters to expand our search
            # Note: We expand *more* than we need so we can filter later
            vocab_probs, vocab_ids = torch.topk(next_token_probs, k=current_beam_width, dim=-1)

            new_kv_cache = outputs.past_key_values

            for batch_idx, candidate in enumerate(batch_candidates):
                # --- CHECK FOR STOPPING (The "Junk Password" Fix) ---
                # We subtract 1 from len() if the tokenizer adds a start token, 
                # but usually just checking len() >= 8 is safe.
                is_long_enough = len(candidate['sequence']) >= getattr(Config, "MIN_PASSWORD_LENGTH", 4)

                # If this specific candidate passes the Epsilon test, we save it as a "Finished Password"
                if has_high_eos_prob[batch_idx] and is_long_enough:
                    # Create a finished entry
                    finished_candidate = candidate.copy()
                    finished_candidate['finished'] = True
                    # Update score: Add log(P(EOS))
                    finished_candidate['score'] += torch.log(eos_probs[batch_idx]).item()
                    final_candidates.append(finished_candidate)


                # --- EXTRACT CACHE FOR THIS SPECIFIC CANDIDATE ---
                # We create a lightweight tuple structure for the next iteration to read
                # We assume the cache has L layers. We slice the batch_idx for each layer.
                candidate_next_cache = []
                for layer_idx in range(len(new_kv_cache)):
                    # Get Key and Value for this layer
                    k_layer, v_layer = new_kv_cache[layer_idx]
                    
                    # Slice out ONLY this candidate's row (keep dim 0 as size 1)
                    # k_layer shape is (Batch, Heads, Seq, Dim). We want (1, Heads, Seq, Dim)
                    k_slice = k_layer[batch_idx : batch_idx+1]
                    v_slice = v_layer[batch_idx : batch_idx+1]
                    
                    candidate_next_cache.append((k_slice, v_slice))



                # --- EXPANDING (Creating New Candidates) ---
                # Even if we stopped, we ALSO continue expanding (in case the password is longer)
                # We loop through the top-K likely next characters
                for k in range(current_beam_width):
                    next_char_id = vocab_ids[batch_idx, k].item()
                    next_char_prob = vocab_probs[batch_idx, k].item()
                    
                    # Skip EOS here (we handled it above)
                    if next_char_id == eos_id:
                        continue
                        
                    # Create the new extended candidate
                    new_beam = {
                        'sequence': candidate['sequence'] + [next_char_id],
                        'score': candidate['score'] + torch.log(torch.tensor(next_char_prob)).item(),
                        'finished': False,
                        'cache': candidate_next_cache 
                    }
                    new_beams.append(new_beam)

        # --- STEP 6: MERGE & PRUNE ---
        # We have processed all batches. 'new_beams' now has thousands of candidates.
        # We must prune it down to K[i+1] for the next depth.
        
        # Sort by score (highest probability first)
        new_beams.sort(key=lambda x: x['score'], reverse=True)
        
        # Keep only the top K for the next round
        # Algorithm 2, Line 9: "SelectTopK"
        next_width = beam_width_schedule[depth+1] if (beam_width_schedule and depth+1 < len(beam_width_schedule)) else 100
        beams = new_beams[:next_width]
        
        print(f"Depth {depth}: Kept {len(beams)} candidates. Found {len(final_candidates)} finished passwords.")

    # --- STEP 7: FINAL RETURN ---
    # If we didn't find any 'finished' candidates, fall back to the current beam set
    if len(final_candidates) == 0 and len(beams) > 0:
        # Convert current beams into final candidates (they already have 'sequence' and 'score')
        final_candidates = [b for b in beams]

    # Compute normalized probabilities (from log-scores -> probabilities)
    if len(final_candidates) > 0:
        # Collect scores as tensor for numerical stability
        scores = torch.tensor([c['score'] for c in final_candidates], device=model.device)
        # Stabilize with max before exponentiating
        scores = scores - torch.max(scores)
        probs = torch.exp(scores)
        probs = probs / torch.sum(probs)
        # Attach probability (%) to each candidate
        for i, c in enumerate(final_candidates):
            c['probability'] = probs[i].item() * 100.0  # percentage

    # Sort again by score and return
    final_candidates.sort(key=lambda x: x['score'], reverse=True)
    return final_candidates