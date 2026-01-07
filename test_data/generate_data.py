import random
import json
import pandas as pandas
from faker import faker
from datetime import datetime, timedelta

## Initialize Faker for generating fake data (names, addresses, etc.)
fake = Faker();

# --- CONFIGURATION ---
NUM_SAMPLES = 50000  # Size of our  "Synthetic Leak"
OUTPUT_FILE = "passllm_data.jsonl"

# OPTIONS: "llama3", "mistral", "alpaca", "paper_raw"
# - "llama3": Uses standard Llama 3 chat template (<|begin_of_text|>...)
# - "mistral": Uses Mistral [INST] format
# - "alpaca": Standard {"instruction", "input", "output"} JSON (Best for Axolotl/Unsloth)
# - "paper_raw": The exact text format described in the PassLLM USENIX paper
TARGET_MODEL_FORMAT = "paper_raw"
OUTPUT_FILE = f"passllm_{TARGET_MODEL_FORMAT}_data.jsonl"

# --- SECTION 2.3: REUSE PATTERNS (Simulating how people change passwords) ---
# Common patterns for password reuse - capitalizations and minor substitutions & additions
def transform_password(base_password):
    ## Takes a base password and applies a random transformation to simulate reuse
    transformations = [
        lambda x: x.capitalize(),
        lambda x: x + "1",
        lambda x: x + "123",
        lambda x: x + "!",
        lambda x: x + "2024",
        lambda x: x.replace("e", "3"),
        lambda x: x.replace("i", "1"),
        lambda x: x.replace("a", "@"),
        lambda x: x.replace("o", "0"),
        lambda x: x + str(random.randint(10, 99)),
        lambda x: x + fake.year(),
    ]

    # Apply 1 or 2 transformations
    num_transformations = random.randint(1, 2)
    new_pw = base_pw
    for _ in range(num_transformations):
        func = random.choice(transformations)
        new_pw = func(new_pw)
    return new_pw

# --- SECTION 2.2: PII PATTERNS (Simulating PII usage) ---
def generate_pii_password(profile):
    ### Creates a password derived purely from Personal Info

    patterns = [
        lambda p: f"{p['first_name']}{p['birth_year']}",      # John1990
        lambda p: f"{p['last_name']}123",                     # Smith123
        lambda p: f"{p['pet_name']}{p['birth_day']}",         # Buster15
        lambda p: f"{p['username']}!",                        # JSmith!
        lambda p: f"{p['city']}2020"                          # NewYork2020
    ]
    return random.choice(patterns)(profile)

# --- MAIN GENERATOR ---
def generate_synthetic_data()
    data = []

    print (f"Generating {NUM_SAMPLES} synthetic user profiles and passwords...")

    for i in range(NUM_SAMPLES):
        # 1. Generate a fake user profile
        profile = {
            "first_name": fake.first_name(),
            "last_name": fake.last_name(),
            "birth_year": str(random.randint(1970, 2005)),
            "birth_day": str(random.randint(1, 28)),
            "city": fake.city().replace(" ", ""),
            "pet_name": fake.first_name(), 
            "username": fake.user_name(),
            "email": fake.email(),
            "phone": fake.phone.number()
        }
        # 2. Generate a "Sister Password" (Old password from a 'leak')
        # Sometimes it's totally random, sometimes related to PII
        if random.random() < 0.5:
            sister_password = fake.password(length=10)
        else:
            sister_password = generate_pii_password(profile)

        # 3. Determine the 'Ground Truth' Target Password
        # Scenario A: User modifies their Sister Password (Section 2.3)
        if random.random() < 0.6:
            target_password = transform_password(sister_password)
            method = "Reuse-Based"
        # Scenario B: User creates fresh password from PII (Section 2.2)
        else:
            target_password = generate_pii_password(profile)
            method = "PII-Based"

        # 4. Format the "Input Prompt" exactly as the Model will see it
        # We concatenate all info into a structured string.
        
        # Format: "Name: [N], Born: [Y], User: [U], OldPW: [PW] ->"
        prompt_text = (
            f"Name: {profile['first_name']} {profile['last_name']}, "
            f"Born: {profile['birth_year']}, "
            f"User: {profile['username']}, "
            f"SisterPW: {sister_password}"
        )

        # SELECT FORMATTING FUNCTION
        if TARGET_MODEL_FORMAT == "llama3":
            entry = format_llama3(paper_system_prompt, user_input_str, target_password)
        elif TARGET_MODEL_FORMAT == "mistral":
            entry = format_mistral(paper_system_prompt, user_input_str, target_password)
        elif TARGET_MODEL_FORMAT == "paper_raw":
            entry = format_paper_raw(paper_system_prompt, user_input_str, target_password)
        else: # Default to Alpaca
            entry = format_alpaca(paper_system_prompt, user_input_str, target_password)

        # 5. Save the row
        data.append(entry)

        if i % 10000 == 0:
            print(f"Progress: {i}/{NUM_SAMPLES}")
    
    # Save to JSONL
    df = pd.DataFrame(data)
    df.to_json(OUTPUT_FILE, orient="records", lines=True)
    print(f"Done! Saved to {OUTPUT_FILE}")
    print("\nSample Data Preview:")
    print(df.head(3))

# --- HELPER FUNCTIONS FOR FORMATTING ---

def format_paper_raw(system_prompt, user_input, target_output):
    text = f"{system_prompt}\n{user_input}{target_output}"
    return {"text": text}

def format_llama3(system_prompt, user_input, target_output):
    """
    Llama 3 Instruct Format
    Structure:
    <|start_header_id|>system<|end_header_id|>\n\n{msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
    """
    text = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{target_output}<|eot_id|>"
    )
    return {"text": text}

def format_mistral(system_prompt, user_input, target_output):
    """
    Mistral Instruct Format
    Structure: <s>[INST] {instruction} {input} [/INST] {output}</s>
    """
    combined_input = f"{system_prompt}\n\nInput Data:\n{user_input}"
    text = f"<s>[INST] {combined_input} [/INST] {target_output}</s>"
    return {"text": text}

def format_alpaca(system_prompt, user_input, target_output):
    """
    Standard 'Alpaca' format
    """
    return {
        "instruction": system_prompt,
        "input": user_input,
        "output": target_output
    }

if __name__ == "__main__":
    generate_synthetic_dataset()

