# Domain-Adaptive Fine-Tuning of SmolLM2-360M for Medical Q&A

Fine-tuning a 360M-parameter language model on a mixed medical Q&A corpus using full Supervised Fine-Tuning (SFT), then benchmarking it against larger models to study the cost-performance tradeoff of small domain-adapted models.

## Research Question

> How much of a 7B instruction-tuned model's medical Q&A performance can a fine-tuned 360M model recover — at roughly 20x lower inference cost?

## Approach

- **Base model:** `HuggingFaceTB/SmolLM2-360M` (base pretrained checkpoint, not instruction-tuned)
- **Method:** Full Supervised Fine-Tuning — all weights updated (no LoRA/QLoRA; the model fits comfortably in T4 VRAM at ~0.72GB in fp16)
- **Training format:** Alpaca-style schema (`instruction` / `input` / `output`)
- **Evaluation:** ROUGE (1, 2, L) against the base model and Mistral-7B (base and instruct)

## Training Corpus

Five medical Q&A datasets are unified into a single Alpaca-style format:

| Dataset | Source | Usable rows | Description |
|---|---|---|---|
| MedQUAD | Parsed from GitHub XML (`abachaa/MedQuAD`) | 16,407 | Intent-typed NIH disease/condition Q&A |
| MedMCQA | `openlifescienceai/medmcqa` | 182,822 | Medical entrance-exam multiple-choice questions |
| ChatDoctor | `lavita/ChatDoctor-HealthCareMagic-100k` | ~112,156 | Real patient-doctor conversations |
| PubMedQA | `qiaojin/PubMedQA` (`pqa_labeled`) | 1,000 | Biomedical research abstract Q&A (human-labeled subset) |
| MedQA USMLE | `GBaker/MedQA-USMLE-4-options` | 10,178 | US licensing-exam clinical vignettes |

Notable data-quality decisions made during exploration:
- **MedQUAD** — 65% of raw answers were empty due to copyright redaction (MedlinePlus Drug info and A.D.A.M. medical encyclopedia). These rows were dropped, leaving 16,407 usable disease-focused pairs.
- **PubMedQA** — only the human-annotated `pqa_labeled` subset is used. The larger `pqa_artificial` subset (211K) was rejected for having auto-generated labels heavily skewed toward "yes" (~93%), and `pqa_unlabeled` has no decision labels.

## Repository Structure

```
smollm2-medical-sft/
├── notebooks/
│   ├── 01a_data_exploration_medquad.ipynb
│   ├── 01b_data_exploration_medmcqa.ipynb
│   ├── 01c_data_exploration_chatdoctor.ipynb
│   ├── 01d_data_exploration_pubmedqa.ipynb
│   └── 01e_data_exploration_medqa_usmle.ipynb
├── configs/
├── src/
├── results/
├── requirements.txt
├── DOCUMENTATION.md
└── README.md
```

Detailed exploration findings, decisions, and rationale are documented in [`DOCUMENTATION.md`](DOCUMENTATION.md).

## Setup

```bash
# Clone the repository
git clone https://github.com/PrajwalSathyanarayana/smollm2-medical-sft.git
cd smollm2-medical-sft

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

HuggingFace authentication (for loading datasets) is handled via `hf auth login` or a local `.env` file containing `HF_TOKEN`.

## Environment

Data exploration and preprocessing are done locally (VS Code + venv). GPU-dependent training and evaluation run on Google Colab (T4). Datasets are stored in Google Drive during development.

## Status

Data exploration is complete for all five datasets. Preprocessing (unifying the datasets into the Alpaca format and building train/validation/test splits) is in progress.