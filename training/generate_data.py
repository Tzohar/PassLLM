import random
import json
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config 

fake = Faker();

NUM_SAMPLES = 5000  # Size of our  "Synthetic Leak"
OUTPUT_FILE = Config.RAW_DATA_FILE # Output path for the generated data

# Common patterns for password reuse - capitalizations and minor substitutions & additions
def transform_password(base_pw):
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
            "name": fake.name(),
            "first_name": fake.first_name(),
            "last_name": fake.last_name(),
            "birth_year": str(random.randint(1970, 2005)),
            "birth_day": str(random.randint(1, 28)),
            "city": fake.city().replace(" ", ""),
            "pet_name": fake.first_name(), 
            "username": fake.user_name(),
            "email": fake.email(),
        }

        # Sometimes it's totally random, sometimes related to PII
        if random.random() < 0.5:
            sister_password = fake.password(length=10)
        else:
            sister_password = generate_pii_password(profile)

        # Scenario A: User modifies their Sister Password (Section 2.3)
        if random.random() < 0.6:
            target_password = transform_password(sister_password)
            method = "Reuse-Based"
        # Scenario B: User creates fresh password from PII (Section 2.2)
        else:
            target_password = generate_pii_password(profile)
            method = "PII-Based"
        
        # Format: "Name: [N], Born: [Y], User: [U], OldPW: [PW] ->"
        pii_dict = {
            "name": f"{profile['name']}",
            "birth_year": profile['birth_year'],
            "username": profile['username'],
            "sister_pw": sister_password
        }
        
        entry = {
            "pii": pii_dict,         
            "output": target_password
        }

        # Save the row
        data.append(entry)

        if i % 1000 == 0:
            print(f"Progress: {i}/{NUM_SAMPLES}")
    
    # Save to JSONL
    df = pd.DataFrame(data)
    df.to_json(OUTPUT_FILE, orient="records", lines=True)
    print(f"Done! Saved to {OUTPUT_FILE}")
    print("\nSample Data Preview:")
    print(df.head(3))


if __name__ == "__main__":
    generate_synthetic_data()

