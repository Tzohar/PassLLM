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

 <img src="https://github.com/user-attachments/assets/00cafb1e-1c28-4c50-9e12-9e00ad33a32f" alt="PassLLM Demo" width="42%">

## Capabilities

* **State-of-the-Art Accuracy:** Achieves **+45% higher success rates** than leading benchmarks (RankGuess, TarGuess) in most scenarios.
* **PII Inference:** With sufficient information, it successfully guesses **12.5% - 31.6%** of typical users within just **100 guesses**.
* **Efficient Fine-Tuning:** Custom training loop utilizing *LoRA* to lower VRAM usage without sacrificing model reasoning capabilities.
* **Advanced Inference:** Implements the paper's algorithm to maximize probability, prioritizing the most likely candidates over random sampling.
* **Data-Driven:** Can be trained on millions of real-world credentials to learn the deep statistical patterns of human passwords creation.
* **Pre-trained Weights:** Includes robust models pre-trained on millions of real-world records from major PII breaches (e.g., Post Millennial, ClixSense) combined with the COMB dataset.

## Use Guide
> **Tip:** You can run this tool instantly without any local installation by opening our [Google Colab Demo](https://colab.research.google.com/github/Tzohar/PassLLM/blob/main/PassLLM_Demo.ipynb), providing your target's PII, and simply executing each cell in order.

### Installation 
* **Python:** 3.10+
* **Password Guessing:** Runs on **Any GPU**, Nvidia or AMD. A standard CPU or Mac (M1/M2) is also sufficient to run the pre-trained model.
* **Training:** NVIDIA GPU with CUDA (RTX 3090/4090 recommended, Google Colab's free tier is often enough).

```bash
# 1. Clone the repository
   git clone https://github.com/tzohar/PassLLM.git
   cd PassLLM

# 2. Install dependencies (Choose one)
   # Option A: Install from requirements (Recommended)
   pip install -r requirements.txt
   
   # Option B: Manual install
   pip install torch torch-directml "transformers<5.0.0" peft datasets bitsandbytes accelerate gradio

```

### Configuration
   
Download the [trained weights](https://github.com/Tzohar/PassLLM/releases/download/v1.0.0/PassLLM_LoRA_Weights.pth) (~160 MB) and place them in the `models/` directory.
*Alternatively, via terminal:*
```bash
curl -L https://github.com/Tzohar/PassLLM/releases/download/v1.0.0/PassLLM_LoRA_Weights.pth -o models/PassLLM_LoRA_Weights.pth
```

   
Once installed and downloaded, adjust the settings in the WebUI or `src/config.py` to match your hardware.
| Hardware | Device | 4-Bit Quantization | Torch DType | Batch Size |
| --- | --- | --- | --- | --- |
| **NVIDIA** | `cuda` | ✅ **On** (Recommended) | `bfloat16` | High (32+) |
| **AMD** | `dml` | ❌ **Off** | `float16` | Low (4-8) |
| **CPU** | `cpu` | ❌ **Off** | `float32` | Low (1-4) |

> **Tip:** Don't forget to customize the **Min/Max Password Length**, **Character Bias**, and **Epsilon** (search strictness) according to your specific target's needs!

### Password Guessing (Pre-Trained)

You can use the graphical interface (WebUI) or the command line to generate candidates.

#### Option A: WebUI (Recommended)

1. **Launch the Interface:**
```bash
python webui.py

```
2. **Generate:**
* Open the local URL (e.g., `http://127.0.0.1:7860`).
* **Select Model:** Choose `PassLLM_LoRA_Weights.pth` from the dropdown.
* **Enter PII:** Fill in the target's Name, Email, Birth Year, etc., into the form.
* **Click Generate:** The engine will stream ranked candidates in real-time.


#### Option B: Command Line (CLI)

Best for automation or headless servers.

1. **Create a Target File:**
Create a `target.jsonl` file (or use the existing one) in the main folder. You can include any field defined in `src/config.py`.
```json
{
  "name": "Johan P.", 
  "birth_year": "1966",
  "email": "johan66@gmail.com",
  "sister_pw": "Johan123"
}

```

2. **Run the Engine:**
```bash
python app.py --file target.jsonl --fast 

```

* `--file`: Path to your target PII file.
* `--fast`: Uses optimized, shallow beam search (omit for full deep search).
* `--superfast`: Very quick but less accurate, mainly for testing.
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
0.42%     | 88888888       
0.32%     | 12345678            
0.16%     | 1976mthorne     
0.15%     | 88marcus88
0.15%     | 1234ABC
0.15%     | 88Marcus!
0.14%     | 1976Marcus
... (227 passwords generated)
```


`{"name": "Elena Rodriguez", "birth_year": "1995", "birth_month": "12", "birth_day": "04", "email": "elena1.rod51@gmail.com"}`:

```text
$ python app.py --file target.jsonl --fast

--- TOP CANDIDATES ---
CONFIDENCE | PASSWORD
------------------------------
1.82%     | 19950404       
1.27%     | 19951204            
0.88%     | 1995rodriguez      
0.55%     | 19951204
0.50%     | 11111111
0.48%     | 1995Rodriguez
0.45%     | 19951995
... (338 passwords generated)
```


`{"name": "Sophia M. Turner", "birth_year": "2001", "username": "soph_t", "email": "sturner99@yahoo.com", "country": "England", "sister_pw": ["soph12345", "13rockm4n", "01mamamia"]}`:

```text
$ python app.py --file target.jsonl --fast

--- TOP CANDIDATES ---
CONFIDENCE | PASSWORD
------------------------------
1.69%     | 01mamamia01       
1.23%     | 13Rockm4n!            
1.14%     | 01mamamia13     
1.02%     | 13rockm4n01
0.96%     | 01mamamia123
0.93%     | 01mama1234
0.77%     | 01mama12345
... (288 passwords generated)
```

 
`{"name": "Omar Al-Fayed", "birth_year": "1992", "birth_month": "05", "birth_day": "18", "username": "omar.fayed92", "email": "o.alfayed@business.ae", "address": "Villa 14, Palm Jumeirah", "phone": "+971-50-123-4567", "country": "UAE", "sister_pw": "Amira1235"}`:

```text
$ python app.py --file target.jsonl 

--- TOP CANDIDATES ---
CONFIDENCE | PASSWORD
------------------------------
1.88%     | 1q2w3e4r
1.59%     | 05181992        
0.95%     | 12345678     
0.66%     | 12345Fayed 
0.50%     | 1OmarFayed92
0.48%     | 1992OmarFayed
0.43%     | 123456amira
... (2865 passwords generated)
```
## Disclaimer

**Please read this section carefully before using.**

* **Unofficial Implementation:** This repository is an independent reproduction and implementation of the research paper *"Password Guessing Using Large Language Models"* (USENIX Security 2025). I am **not** the author of the original paper, nor was I involved in its research or publication. Full credit for the concept and methodology belongs to **Yunkai Zou, Maoxiang An, and Ding Wang** (Nankai University).
* **Educational Purpose Only:** This tool is intended **solely for educational purposes and security research**. It is designed to help security professionals, companies, institutions and casual users understand the risks of LLM-based password attacks and improve defense mechanisms.
* **No Liability:** The author of this repository is not responsible for any misuse of this software. You may not use this tool to attack targets without explicit, authorized consent. **Any illegal use of this software is strictly prohibited.**
