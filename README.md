# BanglaDial-NLU: A Trilingual Benchmark for Intent and Slot Filling in Bangladesh

A trilingual (**Bangla · English · Banglish**) task-oriented dialogue dataset covering everyday services in Bangladesh, plus three BERT baselines for **context-aware joint intent detection and slot filling**.

The benchmark compares a multilingual encoder (mBERT) against monolingual ones (BanglaBERT, English BERT) on the same data, split, and hyperparameters.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [License](#license)
- [Citation](#citation)

---

## Overview

Most dialogue NLU resources are English-only. Real users in Bangladesh switch between Bangla script, English, and romanised Bangla ("Banglish", e.g. *"Ami ekhoni application submit dite chai"*). This project provides:

1. **A parallel dataset**: every utterance is written in all three forms, with token-level slot annotations and turn-level dialogue states.
2. **Three baselines**: a joint intent + slot model on `bert-base-multilingual-cased`, `sagorsarker/bangla-bert-base`, and `bert-base-uncased`.
3. **Careful evaluation**: dialogue-level splits, leakage checks, multi-seed trials, a context ablation, and majority-class baselines.

---

## Dataset

Six JSON files, one per domain group:

| File | Domain(s) | Dialogues | Turns |
|---|---|---:|---:|
| `E-commerce.json` | E-commerce / shopping | 25 | 210 |
| `Education.json` | University admission, tuition, scholarships | 50 | 398 |
| `Government_Services.json` | NID, passport, and other public services | 32 | 256 |
| `Healthcare.json` | Appointments, symptoms, medicine, tests, emergency | 25 | 160 |
| `Mobile_Financial_Services.json` | bKash-style transfers, cash-out, bills, recharge | 25 | 140 |
| `Restaurant_Tourist_Travel_Hotel.json` | Restaurant, tourist, travel, hotel | 52 | 399 |
| **Total** | | **209** | **1,563** (789 user / 774 system) |

### Schema

Each file is a list of dialogues:

```jsonc
{
  "dialogue_id": "dlg_001",
  "domains": ["healthcare"],
  "intent": "appointment_booking",          // one intent per dialogue
  "turns": [
    {
      "turn_id": 0,
      "speaker": "user",                     // "user" | "system"
      "utterances": {
        "bangla":   { "tokens": ["হ্যালো", "..."], "slots": [ /* ... */ ] },
        "english":  { "tokens": ["Hello", "..."],  "slots": [ /* ... */ ] },
        "banglish": { "tokens": ["Hello", "..."],  "slots": [ /* ... */ ] }
      },
      "state_update":   { "healthcare": { "department": "internal_medicine" } },
      "dialogue_state": { "healthcare": { "department": "internal_medicine" } }
    }
  ]
}
```

Each slot annotation:

```json
{
  "slot": "department",
  "surface": "Internal Medicine",
  "normalized_value": "internal_medicine",
  "start_token": 10,
  "end_token": 11
}
```

- Tokens are pre-split per language; slot spans index into that language's token list.
- `normalized_value` is shared across the three languages, so the same concept links the parallel utterances.
- **Span convention caveat:** the notebooks convert spans to BIO tags assuming `end_token` is **exclusive**. Checking `surface` against the tokens shows the raw files do not all follow that convention: `E-commerce.json` and `Government_Services.json` match an exclusive end, while `Education.json`, `Healthcare.json` and `Mobile_Financial_Services.json` match an inclusive end (`Restaurant_Tourist_Travel_Hotel.json` is a mix). A further ~430 slot annotations match neither reading exactly (e.g. surface text normalised differently from the tokens). See [Limitations](#limitations).
- 348 distinct slot names appear in the raw annotations (e.g. `service`, `location`, `payment_method`, `university`, `budget`, `symptom`, `booking_id`). After BIO conversion this becomes **445 slot labels**.
- **Note:** 13 dialogues in `Education.json` have no `domains` field. The notebooks label them `UNKNOWN` for the derived-domain metric.

### Intents

The dataset has **64 distinct intents**, for example `appointment_booking`, `symptom_checker`, `money_transfer`, `nid_card_services`, `admission_requirement`, `scholarship_application`, `price_comparison_discount`, `restaurant_recommendation_booking`, and `travel_flight_ticket_booking`.

---

## Models

All three notebooks share the same pipeline and differ **only in the pretrained backbone** (and in tokenizer casing settings for English BERT).

| Notebook | Backbone |
|---|---|
| `model_mBert.ipynb` | `google-bert/bert-base-multilingual-cased` |
| `model_banglaBert.ipynb` | `sagorsarker/bangla-bert-base` |
| `model_englishBert.ipynb` | `bert-base-uncased` (`strip_accents=False` to preserve Bangla vowel signs) |

### Architecture

A `JointBertNLU` model: one BERT encoder with two heads.

- **Intent head**: pooled `[CLS]` → Dropout → Linear(768→256) → GELU → Dropout → Linear(256→#intents)
- **Slot head**: per-token hidden state → Dropout → Linear(768→#BIO labels)

### Dialogue context

For every **user turn** (in each language) the input is:

```
[CLS] ctx₁ [SEP] ctx₂ [SEP] ctx₃ [SEP] CURRENT USER TURN [SEP]
```

- Up to **3 previous turns** (user and system) in the same language, oldest first.
- When the sequence exceeds `max_len = 256`, the **oldest context is trimmed first**; the current utterance is protected.
- Slot labels (BIO) are applied only to the **first sub-token of each word of the current turn**; context tokens are masked with `-100`.
- Gold `dialogue_state` / `state_update` fields are **dropped at parse time**, so no state information leaks into the model.

### Training setup

| Setting | Value |
|---|---|
| Max sequence length | 256 |
| Context turns | 3 |
| Batch size (train / eval) | 16 / 64 |
| Max epochs / early-stopping patience | 100 / 8 |
| Learning rate | 1.5e-5 with **layer-wise LR decay** (0.95) |
| Scheduler | Cosine with 10% warmup |
| Loss | Label-smoothed (0.1) cross-entropy; intent weight 1.0, slot weight 1.5 |
| Intent class weights | Capped at a max/min ratio of 5 |
| Dropout / weight decay / grad clip | 0.15 / 0.01 / 1.0 |
| Mixed precision | Enabled on CUDA |
| Seeds | 42, 43, 44 (training seeds; the data split seed stays fixed at 42) |
| Checkpoint selection | Validation only: `0.5 × intent acc + 0.5 × slot token macro-F1` |

### Data split

- **Dialogue-level**, stratified by intent, 80 / 10 / 10.
- 137 / 36 / 36 dialogues → 1,560 / 402 / 405 user-turn records (each dialogue contributes records in all three languages; 2,367 records in total).
- Because splitting is by dialogue, the same conversation never appears in two splits, even in a different language.
- The test set is only built and used in the final evaluation cell.

---

## Results

Test-set results, **mean ± std over 3 training seeds** (405 test records per seed).

### Overall

| Model | Intent Acc | Intent Macro-F1 | Slot Token Macro-F1 (incl. `O`) | Slot Entity Micro-F1 | Joint Exact Match |
|---|---:|---:|---:|---:|---:|
| **mBERT** | **0.601 ± 0.012** | **0.408 ± 0.015** | **0.242 ± 0.006** | **0.330 ± 0.003** | **0.219 ± 0.010** |
| English BERT | 0.551 ± 0.034 | 0.348 ± 0.031 | 0.211 ± 0.004 | 0.285 ± 0.007 | 0.189 ± 0.001 |
| BanglaBERT | 0.380 ± 0.017 | 0.230 ± 0.016 | 0.118 ± 0.007 | 0.192 ± 0.007 | 0.123 ± 0.007 |

For reference, the majority-class intent baseline reaches only **0.030** intent accuracy.

### Intent accuracy by language

| Model | Bangla | English | Banglish |
|---|---:|---:|---:|
| mBERT | 0.605 ± 0.019 | 0.625 ± 0.007 | 0.573 ± 0.013 |
| English BERT | 0.435 ± 0.041 | 0.620 ± 0.035 | 0.600 ± 0.026 |
| BanglaBERT | **0.659 ± 0.042** | 0.232 ± 0.019 | 0.249 ± 0.004 |

### Takeaways

- **mBERT is the best all-rounder**: it is the most consistent across scripts and has the lowest variance across seeds.
- **BanglaBERT is the best on Bangla script** (0.659) but collapses on English and Banglish, as expected from a Bangla-only vocabulary.
- **English BERT handles English and Banglish** (both Latin script) well but is weakest on Bangla script among the two non-Bangla models.
- **Context matters a lot.** Context ablation on the test set (seed 42):

  | Model | Intent Acc (with → without context) | Joint EM (with → without context) |
  |---|---|---|
  | BanglaBERT | 0.385 → 0.195 | 0.114 → 0.040 |
  | English BERT | 0.578 → 0.321 | 0.188 → 0.077 |

  The context ablation was run in the BanglaBERT and English BERT notebooks only.

- Bootstrap 95% CIs for intent accuracy (seed 42): BanglaBERT 0.341–0.435, English BERT 0.531–0.622.

> Full per-seed metrics, per-language tables, confusion matrices, and classification reports are written to the output directory when you run the notebooks (see below).

---

## Repository Structure

Suggested layout:

```
.
├── README.md
├── data/
│   ├── E-commerce.json
│   ├── Education.json
│   ├── Government_Services.json
│   ├── Healthcare.json
│   ├── Mobile_Financial_Services.json
│   └── Restaurant_Tourist_Travel_Hotel.json
└── notebooks/
    ├── model_mBert.ipynb
    ├── model_banglaBert.ipynb
    └── model_englishBert.ipynb
```

---

## Getting Started

### Requirements

- Python 3.9+
- A CUDA GPU is strongly recommended (the notebooks were run on Colab / Kaggle)

```bash
pip install torch transformers scikit-learn pandas numpy matplotlib scipy
```

### Run

1. Clone the repo and place the six JSON files in `data/`.
2. **Update the data paths and file names** in Cell 1 of each notebook. They were written against earlier file names and Kaggle/Colab paths:

   | Name used in the notebooks | File in this repo |
   |---|---|
   | `dataset_new.json` | `Restaurant_Tourist_Travel_Hotel.json` |
   | `e_commerce1.json` | `E-commerce.json` |
   | `educationDomain (1).json` | `Education.json` |
   | `gov_service.json` | `Government_Services.json` |
   | `1HealthCare.json` | `Healthcare.json` |
   | `finance.json` | `Mobile_Financial_Services.json` |

   Edit `JSON_FILES` and `DATA_DIR` (or `DATA_DIR_OVERRIDE` in the English BERT notebook) accordingly.
3. Run the cells top to bottom. Outputs are saved to `outputs/<run_id>/` (or `/kaggle/working/outputs`, `/content/outputs` depending on the environment), including:
   - `config.json`, `label_maps_and_split.json`
   - `seed_42/`, `seed_43/`, `seed_44/` (per-seed checkpoints' metrics, reports, confusion matrices)
   - `all_seed_test_metrics.json`, `cross_seed_mean_std.json` / `.csv`
   - `inference_log.json`

To run a quick smoke test, set `train_subset`, `val_subset`, `test_subset`, and `epochs` to small values in `CONFIG`.

---

## Reproducibility

- Seeds are set for Python, NumPy, and PyTorch, with deterministic cuDNN.
- The dialogue split seed is fixed (42); only model init and shuffling vary across the three trials.
- The notebooks include runtime assertions verifying: no future-turn leakage, no state-field leakage, chronological context, zero train/val/test dialogue overlap, and correct BIO ↔ sub-token alignment on every example.
- Checkpoints are selected on the validation set only; the test set is touched once per seed at the end.

---

## Limitations

- **Small dataset**: 209 dialogues, with only 36 in the test split. Differences of a few points between models can fall within noise (note the seed-to-seed std and CIs above).
- **Hard label space**: 64 intents and 445 slot labels with few examples each, so macro metrics are far lower than accuracy.
- **Dialogue-level intent**: each dialogue has a single intent, so short follow-up turns (e.g. *"Yes, Dhanmondi."*) are inherently ambiguous without context. This accounts for many of the intent errors.
- **Span-convention inconsistency**: because `end_token` is interpreted as exclusive in the notebooks but is inclusive in several files, multi-token slots in those files can lose their last token in the BIO labels. This likely depresses the slot scores reported above. Normalising the spans (e.g. using `surface` to resolve each span) before training is recommended, and the numbers here should be treated as a lower bound for slot filling.
- **Parallel data**: the three language versions are parallel, not independently collected, so they may be cleaner and more regular than real user text.
- Domain coverage is limited to the six groups above.

---
