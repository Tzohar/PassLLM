import random
import json
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta

## Initialize Faker for generating fake data (names, addresses, etc.)
fake = Faker();

# --- CONFIGURATION ---
NUM_SAMPLES = 50000  # Size of our  "Synthetic Leak"
OUTPUT_FILE = f"passllm_raw_data.jsonl"

# --- SECTION 2.3: REUSE PATTERNS (Simulating how people change passwords) ---
# Common patterns for password reuse - capitalizations and minor substitutions & additions
def transform_password(base_pw):
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
def generate_synthetic_data():
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
        user_input_str = (
            f"Name: {profile['first_name']} {profile['last_name']}, "
            f"Born: {profile['birth_year']}, "
            f"User: {profile['username']}, "
            f"SisterPW: {sister_password}"
        )
        
        # 1. Define the System Prompt 
        paper_system_prompt = (
            "As a targeted password guessing model, your task is to utilize the "
            "provided information to guess the corresponding password."
        )
        entry = format_paper_raw(paper_system_prompt, user_input_str, target_password)

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
    """
    The exact format used in the PassLLM Paper
    
    Target Model: Mistral-7B-v0.1 (Base)
    Strategy: Raw Text Concatenation
    
    Structure:
    [Instruction]
    [Auxiliary Information]
    Password: [Target Password]
    ([Instruction] \n [Auxiliary Information] \n Password: [Target Password])
    """
    
    # The paper found that a direct instruction worked best
    # We add newlines (\n) to separate the sections clearly
    # The "Password: " string acts as the trigger for the model to start guessing
    
    # text = f"{system_prompt}\n{user_input}\nPassword: {target_output}"
    
    # We return a dictionary because HuggingFace datasets expect this format
    return {
        "instruction": system_prompt,
        "input": user_input,
        "output": target_output
    }
if __name__ == "__main__":
    generate_synthetic_data()

