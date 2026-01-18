# PassLLM: LLM-Based Targeted Password Guessing
[![Paper](https://img.shields.io/badge/USENIX%20Security-2025-blue)](https://www.usenix.org/conference/usenixsecurity25/presentation/zou-yunkai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
## About The Project

**PassLLM is the world's most accurate targeted password guessing framework**, outperforming other models by 15% to 45% in most scenarios. It uses Personally Identifiable Information (PII) - such as _names, birthdays, phone numbers, emails and previous passwords_ - to predict the specific passwords a target is most likely to use. 
The model fine-tunes 7B+ parameter LLMs on millions of leaked PII records using LoRA, enabling a private, high-accuracy framework that runs entirely on consumer PCs.

## Capabilities

* **State-of-the-Art Accuracy:** Achieves **+45% higher success rates** than leading benchmarks (RankGuess, TarGuess) in most scenarios.
* **PII Inference:** With sufficient information, it successfully guesses **12.5% - 31.6%** of typical users within just **100 attempts**.
* **Efficient Fine-Tuning:** Custom training loop utilizing *LoRA* to lower VRAM usage without sacrificing model reasoning capabilities.
* **Advanced Inference:** Implements the paper's algorithm to maximize probability, prioritizing the most likely candidates over random sampling.
* **Data-Driven:** Can be trained on millions of real-world credentials to learn the deep statistical patterns of human passwords creation.

## Use Guide
### Installation
* **Python:** 3.10+
* **Inference:** Runs on **Any Hardware**. A standard CPU or Mac (M1/M2) is sufficient to run the pre-trained model.
* **Training:** NVIDIA GPU with CUDA (RTX 3090/4090 recommended, Google Colab's free tier is sufficient).

```bash
# 1. Clone the repository
git clone [https://github.com/tzohar/PassLLM.git](https://github.com/tzohar/PassLLM.git)
cd PassLLM

# 2. Install dependencies
pip install torch transformers peft datasets bitsandbytes accelerate
```

### Targeted Guessing (Pre-Trained)

Use the pre-trained LoRA weights to guess passwords for a specific target based on their PII.

1.  Create a `target.json` file. You can include any field defined in `schema_defaults` within `src/config.py` (e.g., middle names, cities, usernames).
    ```json
    {
      "first_name": "Johan",
      "birth_year": "1966",
      "email": "johan66@gmail.com",
      "sister_pw": "Johan123"
    }
    ```
    
2.  **Run the inference engine:**
    ```bash
    python app.py --file target.json --fast
    ```
    * `--file`: Path to your target PII file.
    * `--fast`: Uses optimized beam search (omit for full deep search).

The model will generate a ranked list of candidates (sorted by probability) and save them to `results/guesses_Johan.json`.

### Training From Databases

To reproduce the paper's results or train on a new breach, you must provide a dataset of **PII-to-Password pairs**.

1.  **Prepare Your Dataset:**
    Create a file at `training/passllm_raw_data.jsonl`. Each line must be a valid JSON object containing a `pii` dictionary and the target `output` password.
    
    *Example `passllm_raw_data.jsonl`:*
    ```json
    {"pii": {"first_name": "Alice", "birth_year": "1990"}, "output": "Alice1990!"}
    {"pii": {"email": "bob@test.com", "sister_pw": "iloveyou"}, "output": "iloveyou2"}
    ```
    *Note: Ensure your keys (e.g., `first_name`, `email`) match the schema defined in `src/config.py`.*

2.  **Configure Parameters:**
    Edit `src/config.py` to match your hardware and dataset specifics:
    ```python
    # Hardware Settings
    BATCH_SIZE = 4           # Lower to 1 or 2 if hitting OOM on consumer GPUs
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
    * **Saves** the lightweight adapter weights (~20MB) to `models/PassLLM_LoRA_Weights.pth`.

Here are the final sections to complete your README. These are crucial for a portfolio project: Results (proof it works), Architecture (proof you understand the engineering), Disclaimer (legal protection), and Citation (academic credit).

Section 1: Results & Demo
Place this after the "Training" section. It visualizes the output for the reader.

Markdown
## 📊 Results & Demo

Here is a sample run targeting a user with the PII profile: `Name: Johan, Year: 1966`.
