# SmolLM2-360M Medical Q&A — Full SFT Fine-Tuning

[![Model](https://img.shields.io/badge/🤗%20Model-PrajwalS21%2Fsmollm2--360m--medical--sft-blue)](https://huggingface.co/PrajwalS21/smollm2-360m-medical-sft)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-PrajwalS21%2Fmedical--qa--sft--dataset-green)](https://huggingface.co/datasets/PrajwalS21/medical-qa-sft-dataset)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Problem Statement

General-purpose LLMs lack the domain specificity required for reliable medical Q&A, while large domain-specific models are computationally expensive to deploy. This project fine-tunes SmolLM2-360M using full Supervised Fine-Tuning (SFT) on a mixed medical Q&A dataset, benchmarking it against larger base and instruction-tuned models to quantify the efficiency-performance tradeoff on intent-typed medical question answering.

> **Core Research Question:** How much of a 7B instruction-tuned model's medical Q&A performance can a fine-tuned 360M model recover — at a fraction of the inference cost?

---

## Benchmark Results

> *Populated after training*

| Model                              | ROUGE-1 | ROUGE-2 | ROUGE-L |
|------------------------------------|---------|---------|---------|
| SmolLM2-360M (base)                |    -    |    -    |    -    |
| SmolLM2-360M (fine-tuned) ⬅ ours  |    -    |    -    |    -    |
| Mistral-7B (base)                  |    -    |    -    |    -    |
| Mistral-7B Instruct                |    -    |    -    |    -    |

---

## Dataset

A mixed medical Q&A dataset combining five sources to ensure diversity across question styles, medical subdomains, and registers.

| Dataset              | Examples Used | Source                          | What It Contributes              |
|----------------------|---------------|---------------------------------|----------------------------------|
| MedQUAD              | 47K           | NIH (12 websites)               | Structured, intent-typed Q&A     |
| MedMCQA              | 50K           | Medical entrance exams          | Volume and subdomain diversity   |
| ChatDoctor           | 50K           | HealthCareMagic platform        | Conversational medical register  |
| PubMedQA             | 30K           | PubMed research abstracts       | Research-level medical language  |
| MedQA (USMLE)        | 10K           | US Medical Licensing Exam       | Clinical reasoning questions     |
| **Total**            | **~187K**     |                                 |                                  |

> Dataset cleaned, formatted, and published at [PrajwalS21/medical-qa-sft-dataset](https://huggingface.co/datasets/PrajwalS21/medical-qa-sft-dataset)

---

## Approach

### Why Full SFT Over LoRA/QLoRA?

SmolLM2-360M in fp16 occupies ~720MB of VRAM — well within the T4's 15GB limit. With no memory constraint justifying adapter-based methods, full SFT was chosen to maximize domain knowledge instillation by updating all 360M parameters directly on medical Q&A data.

### Training Pipeline

```
SmolLM2-360M (base pretrained)
        ↓
Full SFT on ~187K mixed medical Q&A pairs
fp16 · gradient checkpointing · gradient accumulation
        ↓
SmolLM2-360M (fine-tuned)
        ↓
Benchmark against base + Mistral-7B base + Mistral-7B Instruct
```

### Instruction Format

Each training example is formatted as:

```
<|system|>
You are a medical information assistant. Answer the following 
question clearly and accurately based on established medical knowledge.

<|user|>
[Question]

<|assistant|>
[Answer]
```

---

## Training Setup

| Parameter              | Value                              |
|------------------------|------------------------------------|
| Base model             | HuggingFaceTB/SmolLM2-360M         |
| Base model license     | Apache 2.0                         |
| Fine-tuning method     | Full Supervised Fine-Tuning (SFT)  |
| Hardware               | NVIDIA T4 16GB — Google Colab Pro  |
| Precision              | fp16                               |
| Epochs                 | 2                                  |
| Gradient checkpointing | Enabled                            |
| Gradient accumulation  | 8 steps                            |
| Optimizer              | AdamW                              |
| Learning rate          | 2e-5                               |
| LR scheduler           | Cosine with warmup                 |

---

## Repository Structure

```
smollm2-medical-sft/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── sample/
│       └── sample_medquad.json        # 10-row sample for reference
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      # EDA on all 5 datasets
│   ├── 02_data_preprocessing.ipynb    # Cleaning, formatting, HF push
│   ├── 03_smoke_test.ipynb            # 5K examples, verify pipeline
│   ├── 04_training.ipynb              # Full training run
│   ├── 05_evaluation.ipynb            # Benchmark all 4 models
│   └── 06_inference_demo.ipynb        # Run and compare outputs
│
├── src/
│   ├── data/
│   │   ├── loader.py                  # Load datasets from HuggingFace
│   │   ├── cleaner.py                 # Clean and normalize text
│   │   ├── formatter.py               # Format instruction-response pairs
│   │   └── splitter.py                # Train/val/test split
│   ├── training/
│   │   ├── config.py                  # Hyperparameters
│   │   └── trainer.py                 # Training loop
│   └── evaluation/
│       ├── metrics.py                 # ROUGE scoring
│       └── benchmark.py               # Multi-model evaluation
│
├── configs/
│   ├── smoke_test_config.yaml         # Small run config
│   └── full_training_config.yaml      # Full run config
│
└── results/
    └── benchmark_results.json         # Populated after training
```

---

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR-USERNAME/smollm2-medical-sft.git
cd smollm2-medical-sft
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Notebooks in Order
Open each notebook in Google Colab with **T4 GPU + High RAM** runtime:

| Notebook | Purpose | Est. Time |
|----------|---------|-----------|
| 01_data_exploration | Understand the datasets | 15 min |
| 02_data_preprocessing | Clean and format data | 30 min |
| 03_smoke_test | Verify pipeline on 5K examples | 20 min |
| 04_training | Full training run | ~9 hours |
| 05_evaluation | Benchmark all 4 models | 1-2 hours |
| 06_inference_demo | Run inference and compare | 30 min |

---

## HuggingFace

| Artifact | Link |
|----------|------|
| 🤗 Model | [PrajwalS21/smollm2-360m-medical-sft](https://huggingface.co/PrajwalS21/smollm2-360m-medical-sft) |
| 📦 Dataset | [PrajwalS21/medical-qa-sft-dataset](https://huggingface.co/datasets/PrajwalS21/medical-qa-sft-dataset) |

---

## Limitations

- This model is fine-tuned on publicly available medical Q&A data sourced from NIH, medical exams, and research abstracts. It may not reflect the most current clinical guidelines.
- SmolLM2-360M is a small model. Performance on complex clinical reasoning is limited compared to larger models.
- Benchmark comparisons are on a held-out test set from the same distribution as training data. Out-of-distribution medical questions may produce lower quality responses.

---

## Disclaimer

> ⚠️ This model is for **research and educational purposes only**.
> It is **not intended for clinical use, medical diagnosis, or treatment decisions**.
> Always consult a qualified healthcare professional for medical advice.

---

## License

This project is licensed under the [MIT License](LICENSE).

Base model [HuggingFaceTB/SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) is licensed under Apache 2.0.

---

## Acknowledgements

- [HuggingFace](https://huggingface.co) for SmolLM2 and the datasets ecosystem
- [MedQUAD](https://github.com/abachaa/MedQuAD) — Ben Abacha & Demner-Fushman (2019)
- [MedQA](https://github.com/jind11/MedQA) — Jin et al. (2021)
- [MedMCQA](https://medmcqa.github.io) — Pal et al. (2022)
- [PubMedQA](https://pubmedqa.github.io) — Jin et al. (2019)
- [ChatDoctor](https://github.com/Kent0n-Li/ChatDoctor) — Li et al. (2023)
