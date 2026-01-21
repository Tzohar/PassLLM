# PassLLM: AI-Based Targeted Password Guessing
[![Paper](https://img.shields.io/badge/USENIX%20Security-2025-blue)](https://www.usenix.org/conference/usenixsecurity25/presentation/zou-yunkai)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tzohar/PassLLM/blob/main/PassLLM_Demo.ipynb)
## About The Project

**PassLLM is the world's most accurate targeted password guessing framework**, [outperforming other models by 15% to 45%](https://www.usenix.org/conference/usenixsecurity25/presentation/zou-yunkai) in most scenarios. It uses Personally Identifiable Information (PII) - such as _names, birthdays, phone numbers, emails and previous passwords_ - to predict the specific passwords a target is most likely to use. 
The model fine-tunes 7B+ parameter LLMs on millions of leaked PII records using LoRA, enabling a private, high-accuracy framework that runs entirely on consumer PCs.

## Capabilities

* **State-of-the-Art Accuracy:** Achieves **+45% higher success rates** than leading benchmarks (RankGuess, TarGuess) in most scenarios.
* **PII Inference:** With sufficient information, it successfully guesses **12.5% - 31.6%** of typical users within just **100 guesses**.
* **Efficient Fine-Tuning:** Custom training loop utilizing *LoRA* to lower VRAM usage without sacrificing model reasoning capabilities, runnable on consumer GPUs.
* **Advanced Inference:** Implements the paper's algorithm to maximize probability, prioritizing the most likely candidates over random sampling.
* **Data-Driven:** Can be trained on millions of real-world credentials to learn the deep statistical patterns of human passwords creation.
* **Pre-trained Weights:** Includes robust models pre-trained on millions of real-world records from major PII breaches (e.g., Post Millennial, ClixSense) combined with the COMB dataset.

## Use Guide
**Tip:** You can run this tool instantly without any local installation by opening our [Google Colab Demo](https://colab.research.google.com/github/Tzohar/PassLLM/blob/main/PassLLM_Demo.ipynb), providing your target's PII, and simply executing each cell in order (make sure to use T4 GPU).

### Installation
* **Python:** 3.10+
* **Password Guessing:** Runs on **Any Hardware**. A standard CPU or Mac (M1/M2) is sufficient to run the pre-trained model.
* **Training:** NVIDIA GPU with CUDA (RTX 3090/4090 recommended, Google Colab's free tier is sufficient).

```bash
# 1. Clone the repository
git clone https://github.com/tzohar/PassLLM.git
cd PassLLM

# 2. Install dependencies
pip install torch transformers peft datasets bitsandbytes accelerate
```

### Password Guessing (Pre-Trained)

Use the pre-trained LoRA weights to guess passwords for a specific target based on their PII.

1. Download the [trained weights](https://github.com/Tzohar/PassLLM/releases/download/v1.0.0/PassLLM_LoRA_Weights.pth) (~160 MB) and place them in the `models/` directory.
   Alternatively, run this command in your terminal:
   ```bash
   mkdir -p models && curl -L https://github.com/Tzohar/PassLLM/releases/download/v1.0.0/PassLLM_LoRA_Weights.pth -o models/PassLLM_LoRA_Weights.pth
   
2.  Create a `target.jsonl` file in the main library. You can include any field defined in `schema_defaults` within `src/config.py` (e.g., middle names, cities, usernames).
    ```json
    {
      "name": "Johan P.", 
      "birth_year": "1966",
      "email": "johan66@gmail.com",
      "sister_pw": "Johan123"
    }
    ```
    
3.  **Run the inference engine:**
    ```bash
    python app.py --file target.jsonl --fast
    ```
    * `--file`: Path to your target PII file.
    * `--fast`: Uses optimized, shallow beam search (omit for full deep search).
    * `--superfast`: Very quick but inaccurate, mainly for testing purposes.

An installation of ~15 GBs will commence. The model will generate a ranked list of candidates (sorted by probability) and save them to `/results`.

### Training From Databases

To reproduce the paper's results or train on a new breach, you must provide a dataset of **PII-to-Password pairs**.

1.  **Prepare Your Dataset:**
    Create a file at `training/passllm_raw_data.jsonl`. Each line must be a valid JSON object containing a `pii` dictionary and the target `output` password.
    
    *Example `passllm_raw_data.jsonl`:*
    ```json
    {"pii": {"name": "Alice", "birth_year": "1990"}, "output": "Alice1990!"}
    {"pii": {"email": "bob@test.com", "sister_pw": "iloveyou"}, "output": "iloveyou2"}
    ```
    *Note: Ensure your keys (e.g., `first_name`, `email`) match the schema defined in `src/config.py`.*

2.  **Configure Parameters:**
    Edit `src/config.py` to match your hardware and dataset specifics:
    ```python
    # Hardware Settings
    TRAIN_BATCH_SIZE = 4           # Lower to 1 or 2 if hitting OOM on consumer GPUs
    GRAD_ACCUMULATION = 16   # Simulates larger batches (Effective Batch = 4 * 16 = 64)
    
    # Model Settings
    LORA_R = 16              # Rank dimension (Keep at 16 for standard reproduction)
    VOCAB_BIAS_DIGITS = -4.0 # Penalty strength for non-password patterns
    ```

3.  **Start Training:**
    ```bash
    python train.py
    ```
    This script automates the full pipeline:
    * **Freezes** the base model (Mistral/Qwen).
    * **Injects** Trainable LoRA adapters into Attention layers.
    * **Masks** the loss function so the model only learns to predict the *password*, not the PII.
    * **Saves** the lightweight adapter weights to `models/PassLLM_LoRA_Weights.pth`.

## Results & Demo

`{"name": "Marcus Thorne", "birth_year": "1976", "username": "mthorne88", "country": "Canada"}`:

```text
$ python app.py --file target.jsonl --superfast

--- TOP CANDIDATES ---
CONFIDENCE | PASSWORD
------------------------------
42.25%    | 123456       
11.16%    | 888888           
6.59%     | 1976mthorne     
5.32%     | 88Marcus88
5.28%     | 1234ABC
3.78%     | 88Marcus!
2.61%     | 1976Marcus
... (85 passwords generated)
```


`{"name": "Elena Rodriguez", "birth_year": "1995", "birth_month": "12", "birth_day": "04", "email": "elena1.rod51@gmail.com"}`:

```text
$ python app.py --file target.jsonl --fast

--- TOP CANDIDATES ---
CONFIDENCE | PASSWORD
------------------------------
11.62%    | 123456       
10.98%    | 19950404           
10.03%    | 1qaz2wsx     
5.29%     | 19951204
4.50%     | 1995elena
4.40%     | 111111
4.19%     | 1995Rod
... (428 passwords generated)
```


`{"name": "Sophia M. Turner", "birth_year": "2001", "username": "soph_t", "email": "sturner99@yahoo.com", "country": "England", "sister_pw": ["soph12345", "13rockm4n", "01mamamia"]}`:

```text
$ python app.py --file target.jsonl --fast

--- TOP CANDIDATES ---
CONFIDENCE | PASSWORD
------------------------------
47.79%    | 01mamamia       
28.30%    | 13rockm4n            
3.74%     | 01Mamamia     
2.36%     | 13mamamia
1.87%     | 13rockm4n!
1.73%     | 01mamamia!
1.60%     | 123mamamia
... (435 passwords generated)
```

 
`{"name": "Omar Al-Fayed", "birth_year": "1992", "birth_month": "05", "birth_day": "18", "username": "omar.fayed92", "email": "o.alfayed@business.ae", "address": "Villa 14, Palm Jumeirah", "phone": "+971-50-123-4567", "country": "UAE", "sister_pw": "Amira1235"}`:

```text
$ python app.py --file target.jsonl --fast

--- TOP CANDIDATES ---
CONFIDENCE | PASSWORD
------------------------------
20.28%    | 123456 
5.30%     | 1qaz2wsx             
4.56%     | 123Fayed      
3.40%     | 1OmarFayed 
2.86%     | 1992Omar
2.36%     | 1234ABC
1.86%     | 1992amira
... (3091 passwords generated)
```
## Disclaimer

**Please read this section carefully before using.**

* **Unofficial Implementation:** This repository is an independent reproduction and implementation of the research paper *"Password Guessing Using Large Language Models"* (USENIX Security 2025). I am **not** the author of the original paper, nor was I involved in its research or publication. Full credit for the concept and methodology belongs to **Yunkai Zou, Maoxiang An, and Ding Wang** (Nankai University).
* **Educational Purpose Only:** This tool is intended **solely for educational purposes and security research**. It is designed to help security professionals, companies, institutions and casual users understand the risks of LLM-based password attacks and improve defense mechanisms.
* **No Liability:** The author of this repository is not responsible for any misuse of this software. You may not use this tool to attack targets without explicit, authorized consent. **Any illegal use of this software is strictly prohibited.**
