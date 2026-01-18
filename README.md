# PassLLM: LLM-Based Targeted Password Guessing
[![Paper](https://img.shields.io/badge/USENIX%20Security-2025-blue)](https://www.usenix.org/conference/usenixsecurity25/presentation/zou-yunkai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
## About The Project

**PassLLM is the world's most accurate targeted password guessing framework**, outperforming other models by 15% to 45% in most scenarios. It uses Personally Identifiable Information (PII) - such as _names, birthdays, phone numbers, emails and previous passwords_ - to predict the specific passwords a target is most likely to use. 
The model fine-tunes 7B+ parameter LLMs on millions of leaked PII records using LoRA, enabling a private, high-accuracy framework that runs entirely on consumer PCs.

## Capabilities

* **State-of-the-Art Accuracy:** Achieves **+45% higher success rates** than leading benchmarks in targeted scenarios.
* **PII Inference:** With sufficient information, it successfully guesses **12.5% - 31.6%** of typical users within just **100 attempts**.
* **Efficient Fine-Tuning:** Custom training loop utilizing *LoRA* to lower VRAM usage without sacrificing model capabilities.
* **Advanced Inference:** Implements the paper's algorithm to maximize probability, prioritizing the most likely candidates.
* **Data-Driven:** Can be trained on millions of real-world credentials to learn deep statistical patterns of human passwords.



