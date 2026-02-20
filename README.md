# PassLLM: AI-Based Targeted Password Guessing
[![Paper](https://img.shields.io/badge/USENIX%20Security-2025-blue)](https://www.usenix.org/conference/usenixsecurity25/presentation/zou-yunkai)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tzohar/PassLLM/blob/main/PassLLM_Demo.ipynb)
[![Email](https://img.shields.io/badge/Email-PassLLM%40proton.me-blue?style=flat&logo=protonmail&logoColor=white)](mailto:passllm@proton.me)

## About The Project
**PassLLM is the world's most accurate targeted password guessing framework**, [outperforming other models by 15% to 45%](https://www.usenix.org/conference/usenixsecurity25/presentation/zou-yunkai) in most scenarios. It uses Personally Identifiable Information (PII) - such as _names, birthdays, phone numbers, emails and previous passwords_ - to predict the specific passwords a target is most likely to use. 
The model fine-tunes 7B/4B parameter LLMs on millions of leaked PII records using LoRA, enabling a private, high-accuracy framework that runs entirely on consumer PCs.

 <img src="https://github.com/user-attachments/assets/00cafb1e-1c28-4c50-9e12-9e00ad33a32f" alt="PassLLM Demo" width="52%">

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
   
Download the [trained weights](https://github.com/Tzohar/PassLLM/releases/download/v1.3.0/PassLLM-Qwen3-4B-v1.0.pth) (~126 MB) and place them in the `models/` directory.
*Alternatively, via terminal:*
```bash
curl -L https://github.com/Tzohar/PassLLM/releases/download/v1.3.0/PassLLM-Qwen3-4B-v1.0.pth -o models/PassLLM_LoRA_Weights.pth
```

   
Once installed and downloaded, adjust the settings in the WebUI or `src/config.py` to match your hardware.
| Hardware | OS | Device | 4-Bit Quantization | Torch DType | Inference Batch Size |
| --- | --- | --- | --- | --- | --- |
| **NVIDIA** | Any | `cuda` | ✅ **On** (Recommended) | `bfloat16` | High (64+) |
| **AMD** | Windows | `dml` | ❌ **Off** | `float16` | Low (8-16) |
| **AMD (RDNA 3+)** | Linux/WSL | `cuda` | ❌ **Off** | `bfloat16` | Medium (64+) |
| **AMD (Older)** | Linux/WSL | `cuda` | ❌ **Off** | `float16` | Low (8-16) |
| **CPU** | Any | `cpu` | ❌ **Off** | `float32` | Low (1-4) |

> **Note (AMD on Linux/WSL):** DirectML (`dml`) is Windows-only. For AMD GPUs on Linux or WSL, you must install [ROCm](https://rocm.docs.amd.com/) and [PyTorch for ROCm](https://pytorch.org/get-started/locally/). Once installed, set `DEVICE = "cuda"` as ROCm uses the CUDA API. 4-bit quantization (bitsandbytes) is not officially supported on ROCm. Newer AMD GPUs (RDNA 3 / RX 7000 series, MI200/MI300) have native `bfloat16` support, use it for significant speed improvements.

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
* **Select Model:** Choose the most recent model from the dropdown.
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
1.96%     | marcus1976   
1.91%     | thorne1976 
1.20%     | mthorne1976 
1.19%     | marc1976 (marc is a common diminutive of Marcus, used in many passwords) 
1.18%     | a123456 (a high-probability global baseline across users with similar PII) 
1.16%     | marci1976 (another common variation of Marcus)
1.01%     | winniethepooh (our training dataset demonstrated Winnie-related passwords to be common in Canada)
... (907 passwords generated)
```


`{"name": "Elena Rodriguez", "birth_year": "1995", "birth_month": "12", "birth_day": "04", "email": "elena1.rod51@gmail.com", "id":"489298321"}`:

```text
$ python app.py --file target.jsonl --fast

--- TOP CANDIDATES ---
CONFIDENCE | PASSWORD
------------------------------
8.55%     | elena1204 (all variations of name + birth date are naturally given very high probability)
8.16%     | elena1995
7.77%     | elena951204     
6.29%     | elena9512
5.37%     | Elena1995
5.32%     | elena1.rod51 
5.00%     | 120495
... (5,895 passwords generated)
```


`{"name": "Sophia M. Turner", "birth_year": "2001", "pet_name": "Fluffy", "username": "soph_t", "email": "sturner99@yahoo.com", "country": "England", "sister_pw": ["soph12345", "13rockm4n", "01mamamia"]}`:

```text
$ python app.py --file target.jsonl --fast

--- TOP CANDIDATES ---
CONFIDENCE | PASSWORD
------------------------------
2.93%     | sophia123 (this is a mix of the target's first name and the sister password "soph12345")       
2.53%     | mamamia01 (a simple variation of another sister password)       
1.96%     | sophia2001     
1.78%     | sophie123 (UK passwords often interchange between "sophie" and "sophia")
1.45%     | 123456a (a very commmon password, ranked high due to the "12345" pattern) 
1.39%     | sophiesophie1
1.24%     | sturner999 
... (10,169 passwords generated)
```

 
`{"name": "Omar Al-Fayed", "birth_year": "1992", "birth_month": "05", "birth_day": "18", "username": "omar.fayed92", "email": "o.alfayed@business.ae", "address": "Villa 14, Palm Jumeirah", "phone": "+971-50-123-4567", "country": "UAE", "sister_pw": "Amira1235"}`:

```text
$ python app.py --file target.jsonl 

--- TOP CANDIDATES ---
CONFIDENCE | PASSWORD
------------------------------
79.75%     | amira1235 (sister password, with lower case a) 
43.77%     | Ammira1235 (common pattern in reusing passwords)     
19.14%     | Omar1235 (drawing on the sister password pattern)    
11.03%     | Omar1234
8.52%      | omarr.alfayid
8.20%      | omar1235
7.52%      | 051892
... (24,559 passwords generated)
```
## Disclaimer

**Please read this section carefully before using.**

* **Unofficial Implementation:** This repository is an independent reproduction and implementation of the research paper *"Password Guessing Using Large Language Models"* (USENIX Security 2025). I am **not** the author of the original paper, nor was I involved in its research or publication. Full credit for the concept and methodology belongs to **Yunkai Zou, Maoxiang An, and Ding Wang** (Nankai University).
* **Educational Purpose Only:** This tool is intended **solely for educational purposes and security research**. It is designed to help security professionals, companies, institutions and casual users understand the risks of LLM-based password attacks and improve defense mechanisms.
* **No Liability:** The author of this repository is not responsible for any misuse of this software. You may not use this tool to attack targets without explicit, authorized consent. **Any illegal use of this software is strictly prohibited.**
