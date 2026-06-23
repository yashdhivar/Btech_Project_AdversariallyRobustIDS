# Training & Evaluation Pipeline — Adversarially Robust IDS

This document explains the complete training and evaluation flow across all three tiers.

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Data Loading & Preprocessing (Common to All Tiers)](#2-data-loading--preprocessing)
3. [Tier 1 — Signature-Based Detection](#3-tier-1--signature-based-detection)
4. [Tier 2 — ML-Based Detection](#4-tier-2--ml-based-detection)
5. [Tier 3 — Adversarial Defense](#5-tier-3--adversarial-defense)
6. [GAN Module (WGAN-GP) — Synthetic Data Generation](#6-gan-module-wgan-gp)
7. [Role of GAIN — Clarification](#7-role-of-gain--clarification)
8. [End-to-End Pipeline (Tier 1 → Tier 3)](#8-end-to-end-pipeline)
9. [Evaluation Framework](#9-evaluation-framework)

---

## 1. High-Level Architecture

```
Raw Traffic
    │
    ▼
┌──────────────────┐   Match found?
│  TIER 1           │──── YES ──→ ALERT (tier=1, confidence=CERTAIN)
│  Signature Match  │
└──────┬───────────┘
       │ NO
       ▼
┌──────────────────┐   Benign?
│  TIER 2           │──── YES ──→ RETURN (status=BENIGN)
│  ML Detection     │
│  (DNN/CNN/LSTM)   │
└──────┬───────────┘
       │ Attack detected
       ▼
┌──────────────────┐   Adversarial?
│  TIER 3           │──── YES ──→ ALERT (tier=3, severity=CRITICAL)
│  Adversarial      │──── NO  ──→ ALERT (tier=2, normal attack)
│  Defense          │
└──────────────────┘
```

---

## 2. Data Loading & Preprocessing

**Source:** `src/preprocessing/data_loader.py` and `src/preprocessing/preprocessor.py`

### 2.1 Data Loading

- **Datasets supported:** NSL-KDD, CICIDS2017
- **Label mapping:** All attack types are mapped to 5 categories:
  | Label | Category |
  |-------|----------|
  | 0     | Normal   |
  | 1     | DoS      |
  | 2     | Probe    |
  | 3     | R2L      |
  | 4     | U2R      |
- If no real dataset is present, the system generates synthetic data (configurable class distribution: 60% Normal, 40% Attack).

### 2.2 Preprocessing Pipeline (Step by Step)

| Step | Operation | Details |
|------|-----------|---------|
| 1 | **Clean data** | Replace infinity with NaN, fill missing values (median), remove duplicates |
| 2 | **Encode labels** | Multiclass: map to 0–4 using CATEGORY_MAP |
| 3 | **Encode categoricals** | One-hot encoding for `protocol_type`, `service`, `flag` |
| 4 | **Split data** | **70% Train / 15% Validation / 15% Test** (stratified) |
| 5 | **Scale features** | StandardScaler (z-score normalization), fitted on train only, applied to all splits |
| 6 | **Feature selection** | `SelectKBest` with `mutual_info_classif`, keeps **top 35 features** |
| 7 | **Handle imbalance** | **SMOTE** applied **only on training set** (k_neighbors=3) |

### 2.3 Output

```
X_train: (N_train, 35) float32  — balanced via SMOTE
X_val:   (N_val, 35)   float32  — original distribution
X_test:  (N_test, 35)  float32  — original distribution
y_train, y_val, y_test: integer labels (0–4)
```

This preprocessed data is shared by Tier 2 training, GAN training, and Tier 3 adversarial training.

---

## 3. Tier 1 — Signature-Based Detection

**Source:** `src/tier1_signature/signature_detector.py`, `signature_database.py`, `pattern_matcher.py`

### 3.1 How It Works

- **No training is needed.** Tier 1 is purely rule-based.
- A JSON signature database (`data/signatures/signatures.json`) contains known attack patterns (SYN Flood, Port Scan, Brute Force, etc.).
- Each signature has conditions on raw traffic features (e.g., `syn_flag_count > 100`, `protocol = TCP`).
- A traffic sample is matched against all signatures. If **≥ 80%** of a signature's conditions are satisfied, it is flagged as that attack type.

### 3.2 Evaluation

- Deterministic — confidence is always `CERTAIN` when matched.
- Evaluated by checking detection rate on known attack samples from the test set.
- Very fast (< 1ms per sample).
- **Limitation:** Cannot detect novel/unknown attacks → passes those to Tier 2.

---

## 4. Tier 2 — ML-Based Detection

**Source:** `src/tier2_ml_detection/models.py`, `train.py`, `ml_detector.py`, `feature_extractor.py`

### 4.1 Data Split

| Split | Percentage | Purpose | SMOTE Applied? |
|-------|-----------|---------|----------------|
| Train | 70% | Model training | Yes |
| Validation | 15% | Early stopping, LR scheduling, model selection | No |
| Test | 15% | Final evaluation (never seen during training) | No |

The split is **stratified** — each split preserves the original class distribution (before SMOTE).

### 4.2 Model Architectures

Three models are trained in parallel, and the **best one** (highest validation accuracy) is selected:

#### DNN (Dense Neural Network) — for tabular data
```
Input (35 features)
  → Dense(256) + BatchNorm + ReLU + Dropout(0.3)
  → Dense(128) + BatchNorm + ReLU + Dropout(0.3)
  → Dense(64)  + BatchNorm + ReLU + Dropout(0.3)
  → Dense(32)  + ReLU
  → Dense(5)   + Softmax
```

#### CNN (1D Convolutional) — for pattern recognition
```
Input reshaped to (35, 1)
  → Conv1D(64, kernel=3) + BatchNorm + MaxPool
  → Conv1D(128, kernel=3) + BatchNorm + MaxPool
  → Conv1D(256, kernel=3) + BatchNorm
  → Flatten
  → Dense(128) + Dropout(0.3)
  → Dense(64)
  → Dense(5) + Softmax
```

#### LSTM — for sequential patterns
```
Input reshaped to (1, 35) — single timestep
  → LSTM(128, return_sequences=True) + Dropout(0.3)
  → LSTM(64)
  → Dense(32) + ReLU
  → Dense(5) + Softmax
```

### 4.3 Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=0.001) |
| Loss | `sparse_categorical_crossentropy` |
| Epochs | 30 |
| Batch Size | 128 |
| Early Stopping | patience=10, monitor=val_loss |
| LR Reduction | ReduceLROnPlateau |
| Checkpoint | Save best model by val_loss |

### 4.4 Training Flow

1. Preprocessed `X_train, y_train` fed to each model.
2. Each model trains for up to 30 epochs (early stopping may stop sooner).
3. Validation set (`X_val, y_val`) is used for:
   - Early stopping (stops when val_loss stops improving for 10 epochs)
   - Learning rate reduction (halves LR when val_loss plateaus)
   - Model checkpointing (saves only the best epoch)
4. After all 3 models finish, the one with **highest validation accuracy** is chosen.
5. Best model saved to `models/tier2/best_model.h5`.

### 4.5 Evaluation

- **Test set** (`X_test, y_test`) is used for final evaluation.
- Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, AUC-ROC.
- Per-class FPR/FNR are also computed.

### 4.6 Role of GAIN in Tier 2

**There is no GAIN (Generative Adversarial Imputation Network) in this project.**

What exists is a **WGAN-GP (Wasserstein GAN with Gradient Penalty)** in `src/gan_generator/`. This is a standard GAN for **synthetic data generation**, not a GAIN for missing data imputation. See [Section 6](#6-gan-module-wgan-gp) for details.

The WGAN-GP generates synthetic attack samples that **can** be used to augment Tier 2 training data (more attack samples → better minority class representation), but the primary class balancing is done via **SMOTE** in the preprocessing step.

---

## 5. Tier 3 — Adversarial Defense

**Source:** `src/tier3_adversarial_defense/adversarial_training.py`, `adversarial_defense.py`, `input_transformation.py`, `ensemble_defense.py`

### 5.1 How Adversarial Data Is Produced

Adversarial examples are generated **on-the-fly during training** — they are NOT pre-generated from a static dataset. The process:

1. Take a batch of clean training samples `(X_batch, y_batch)`.
2. Generate **FGSM adversarial examples**:
   ```
   x_adv = x + ε × sign(∇_x Loss(model(x), y))
   ```
   - ε (epsilon) = 0.1
   - One gradient step — fast but less powerful
3. Generate **PGD adversarial examples**:
   ```
   For 40 iterations:
       x_adv = x_adv + α × sign(∇_x Loss(model(x_adv), y))
       x_adv = clip(x_adv, x - ε, x + ε)
   ```
   - ε = 0.1, α (step size) = 0.01, iterations = 40
   - Multi-step — stronger attack
4. **Mix the data** for that batch:
   - **40% clean** (original samples)
   - **30% FGSM** adversarial examples
   - **30% PGD** adversarial examples

### 5.2 Model Architecture (PyTorch DNN)

Mirrors the Keras DNN from Tier 2 but implemented in PyTorch (needed for gradient-based adversarial attacks):

```
Linear(35 → 256) + ReLU + BatchNorm + Dropout(0.3)
Linear(256 → 128) + ReLU + BatchNorm + Dropout(0.3)
Linear(128 → 64)  + ReLU + BatchNorm + Dropout(0.3)
Linear(64 → 32)   + ReLU
Linear(32 → 5)    — raw logits (CrossEntropyLoss handles softmax)
```

### 5.3 Training Flow

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss | CrossEntropyLoss |
| Epochs | 20 |
| Training data | Mixed (40% clean + 30% FGSM + 30% PGD) |
| Validation data | Clean data only (X_val, y_val) |

For each epoch:
1. Iterate over training batches.
2. For each batch:
   - Generate FGSM adversarial examples from the current batch.
   - Generate PGD adversarial examples from the current batch.
   - Concatenate: 40% clean + 30% FGSM + 30% PGD.
   - Forward pass on the mixed batch.
   - Compute CrossEntropyLoss.
   - Backpropagate and update weights.
3. Validate on the **clean** validation set (no adversarial examples).
4. Log training loss and validation accuracy.

**Output:** Robust model saved to `models/tier3/robust_model.pth`

### 5.4 Evaluation of Tier 3

Tier 3 is evaluated on two fronts:

#### A. Robustness Evaluation
- Generate adversarial examples from the **test set** using FGSM, PGD, C&W, and DeepFool.
- Measure **clean accuracy** (on original test set) vs **robust accuracy** (on adversarial test set).
- Key metrics:
  - **Accuracy drop** = clean_accuracy − robust_accuracy
  - **Robustness ratio** = robust_accuracy / clean_accuracy (closer to 1.0 = more robust)

#### B. Adversarial Detection Evaluation
At inference time, Tier 3 uses **feature squeezing** to detect adversarial inputs:
1. Get prediction on the original input → probability distribution P₁.
2. Apply feature squeezing (bit-depth reduction to 4 bits) → get prediction P₂.
3. Compute L1 distance: `distance = |P₁ - P₂|.sum()`
4. If `distance > 0.1` → **adversarial detected**.

This is evaluated by feeding both clean and adversarial test samples and measuring detection accuracy (TPR for adversarial, FPR for clean).

### 5.5 Is There GAIN Data in Tier 3?

**No.** There is no GAIN (Generative Adversarial Imputation Network) anywhere in the project. Tier 3 uses:
- **FGSM and PGD** to generate adversarial training data on-the-fly.
- **Feature squeezing** to detect adversarial inputs at inference.
- **Ensemble voting** for robust classification.

The GAN module (WGAN-GP) is a separate component for generating synthetic attack traffic — it is not used within Tier 3's adversarial training loop.

---



## 7. Role of GAIN — Clarification

**GAIN (Generative Adversarial Imputation Network) does NOT exist in this project.**

GAIN is a technique for imputing missing data using GANs. This project does not use GAIN because:
- Missing values are handled via **median imputation** in the preprocessing step.
- The GAN in this project is a **WGAN-GP** used for **synthetic data generation**, not imputation.

| | GAIN | WGAN-GP (this project) |
|---|------|------------------------|
| Purpose | Fill in missing data | Generate new synthetic samples |
| Input | Incomplete data matrix | Random noise vector |
| Output | Completed data matrix | Entirely new samples |
| Used here? | **No** | **Yes** |

---

## 8. End-to-End Pipeline

### 8.1 Training Pipeline (run via `python main.py --mode train`)

```
Step 1: DATA LOADING
  │  Load NSL-KDD / CICIDS2017 / synthetic data
  │  Map labels → 5 categories
  ▼
Step 2: PREPROCESSING
  │  Clean → Encode → Split (70/15/15) → Scale → Feature Select (35) → SMOTE
  │  Output: X_train, X_val, X_test, y_train, y_val, y_test
  ▼
Step 3: TIER 2 TRAINING
  │  Train DNN, CNN, LSTM in parallel
  │  Select best model by validation accuracy
  │  Save → models/tier2/best_model.h5
  ▼
Step 4: GAN TRAINING (WGAN-GP)
  │  Train on attack-only subset of X_train
  │  100 epochs, WGAN-GP with gradient penalty
  │  Save → models/gan/generator_final.pth
  ▼
Step 5: TIER 3 ADVERSARIAL TRAINING
  │  Use X_train with on-the-fly FGSM + PGD augmentation
  │  Mixed batches: 40% clean + 30% FGSM + 30% PGD
  │  20 epochs, validate on clean X_val
  │  Save → models/tier3/robust_model.pth
```

### 8.2 Evaluation Pipeline (run via `python main.py --mode evaluate`)

```
Step 1: LOAD MODELS
  │  Tier 2: best_model.h5 (Keras)
  │  Tier 3: robust_model.pth (PyTorch)
  ▼
Step 2: CLEAN EVALUATION
  │  Run Tier 2 model on X_test
  │  Compute: Accuracy, Precision, Recall, F1, AUC-ROC, Confusion Matrix
  ▼
Step 3: ADVERSARIAL EVALUATION
  │  Generate adversarial X_test using FGSM, PGD, C&W, DeepFool
  │  Run Tier 2 model → get accuracy (shows vulnerability)
  │  Run Tier 3 robust model → get accuracy (shows improvement)
  │  Compute: accuracy drop, robustness ratio
  ▼
Step 4: COMPARISON
  │  Standard model accuracy vs. Robust model accuracy
  │  Per-attack-type breakdown
  │  Save results → evaluation_results/
```

### 8.3 Inference Pipeline (run via `python main.py --mode detect`)

```
New Traffic Sample
  │
  ▼
TIER 1: Pattern match against signature DB
  │  Match found? → ALERT(tier=1) → DONE
  │  No match? → continue
  ▼
TIER 2: Preprocess → Feed to best ML model
  │  Benign (class=0)? → RETURN BENIGN → DONE
  │  Attack (class=1–4)? → continue
  ▼
TIER 3: Adversarial check
  │  1. Ensemble prediction on original input
  │  2. Feature squeezing → prediction on squeezed input
  │  3. L1 distance between predictions
  │  Distance > 0.1? → ALERT(tier=3, adversarial=True, severity=CRITICAL)
  │  Distance ≤ 0.1? → ALERT(tier=2, adversarial=False)
```

---

## 9. Evaluation Framework

**Source:** `src/evaluation/evaluator.py`, `metrics.py`

### 9.1 Metrics Summary

| Metric | Where Used | What It Measures |
|--------|-----------|------------------|
| Accuracy | Tier 2, Tier 3 | Overall correct predictions / total |
| Precision | Tier 2 | True attacks / predicted attacks (per class) |
| Recall | Tier 2 | Detected attacks / actual attacks (per class) |
| F1-Score | Tier 2 | Harmonic mean of precision & recall |
| AUC-ROC | Tier 2 | Area under ROC curve (binary or one-vs-rest) |
| Confusion Matrix | Tier 2 | Full breakdown of predictions vs. ground truth |
| FPR / FNR | Tier 2 | False positive & false negative rates (per class) |
| Clean Accuracy | Tier 3 | Accuracy on original (unperturbed) test data |
| Robust Accuracy | Tier 3 | Accuracy on adversarial test data |
| Accuracy Drop | Tier 3 | clean_accuracy − robust_accuracy |
| Robustness Ratio | Tier 3 | robust_accuracy / clean_accuracy (target: close to 1.0) |
| Adversarial Detection Rate | Tier 3 | % of adversarial inputs correctly flagged |

### 9.2 Key Insight

The system provides **defense in depth**:
- **Tier 1** catches known attacks instantly (< 1ms).
- **Tier 2** catches unknown attacks via learned patterns.
- **Tier 3** ensures that even if an attacker crafts adversarial inputs to fool Tier 2, the system can detect and correctly classify them.
