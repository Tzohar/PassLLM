import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict
from transformers.cache_utils import DynamicCache

def dynamic_beam_search(
    model, 
    tokenizer, 
    auxiliary_info_ids: torch.Tensor, 
    max_depth: int = 16, 
    beam_width_schedule: List[int] = None, 
    batch_size: int = 10, 
    epsilon: float = 0.1
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

        
        # The model has read the PII, now it predicts the very first letter of the password
        # We only care about the last token's prediction (index -1)
        # But since we have only provided PII, the model's next token is the FIRST char of the password
        next_token_logits = outputs.logits[:, -1, :]

    # --- STEP 2: INITIALIZE CANDIDATES ---
    # We started with a single empty candidate (the beginning of the password)
    # We will limit the model to generating only 95 ASCII characters
    # We force the probability of all other characters (emojis, chinese, etc.) to -Infinity

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

            # We take the PII memory (shared_kv_cache) and virtually 
            # copy it 'current_batch_size' times to match the input batch

            # Create a new cache object for this batch
            expanded_cache = DynamicCache()
            
            # Loop through every layer of the memory
            for layer_idx in range(len(shared_kv_cache)):
                # Get the original (Batch=1) keys and values
                k, v = shared_kv_cache[layer_idx]
                
                # COMMAND: "Put it N times"
                # .expand() creates virtual copies instantly
                k_expanded = k.expand(current_batch_size, -1, -1, -1)
                v_expanded = v.expand(current_batch_size, -1, -1, -1)
                
                # Add to our new cache object
                expanded_cache.update(k_expanded, v_expanded, layer_idx)

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

            # Get the probabilities for the next token
            next_token_logits = outputs.logits[:, -1, :]
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

            for batch_idx, candidate in enumerate(batch_candidates):
                
                # --- CHECK FOR STOPPING (The "Junk Password" Fix) ---
                # If this specific candidate passes the Epsilon test, we save it as a "Finished Password"
                if has_high_eos_prob[batch_idx]:
                    # Create a finished entry
                    finished_candidate = candidate.copy()
                    finished_candidate['finished'] = True
                    # Update score: Add log(P(EOS))
                    finished_candidate['score'] += torch.log(eos_probs[batch_idx]).item()
                    final_candidates.append(finished_candidate)

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
                        'finished': False
                    }
                    new_beams.append(new_beam)

            # --- STEP 6: MERGE & PRUNE (End of Batch Loop) ---
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
        # Return sorted results
        final_candidates.sort(key=lambda x: x['score'], reverse=True)
        return final_candidates