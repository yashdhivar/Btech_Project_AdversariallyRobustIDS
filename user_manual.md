# Adversarially Robust IDS -- User Manual

This manual covers everything you need to know to install, configure, train, and use the Adversarially Robust Intrusion Detection System. It is written for someone with basic Python knowledge who is new to this project.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Architecture Overview](#2-architecture-overview)
3. [Prerequisites](#3-prerequisites)
4. [Installation Step by Step](#4-installation-step-by-step)
5. [Dataset Setup](#5-dataset-setup)
6. [Quick Start](#6-quick-start)
7. [Training Models](#7-training-models)
8. [Running Detection](#8-running-detection)
9. [Using the Dashboard](#9-using-the-dashboard)
10. [Running Evaluation](#10-running-evaluation)
11. [Running Tests](#11-running-tests)
12. [Configuration Reference](#12-configuration-reference)
13. [Code Walkthrough](#13-code-walkthrough)
14. [How the Detection Flow Works](#14-how-the-detection-flow-works)
15. [Troubleshooting](#15-troubleshooting)
16. [Glossary](#16-glossary)

---

## 1. Introduction

### What Is This Project?

This is a network intrusion detection system (IDS) that can identify cyberattacks in network traffic while resisting adversarial evasion. Traditional IDS systems use either fixed rules (signatures) or machine learning models. Both have limitations:

- Signature-based systems miss unknown attacks (zero-day attacks).
- ML-based systems can be fooled by adversarial examples -- subtly modified inputs designed to cause misclassification.

This project solves both problems by combining three detection tiers into a single pipeline:

1. **Tier 1** catches known attacks instantly using pattern matching.
2. **Tier 2** catches unknown attacks using deep learning (DNN, CNN, LSTM).
3. **Tier 3** detects and resists adversarial evasion attempts using adversarial training, input transformation, and ensemble voting.

Additionally, a **GAN (Generative Adversarial Network)** module generates realistic synthetic attack traffic for augmenting training data and stress-testing the system.

### What Can You Do With It?

- Train ML models on the NSL-KDD or CICIDS2017 benchmark datasets
- Run intrusion detection on network traffic (CSV files or simulated real-time)
- Generate adversarial attacks (FGSM, PGD, C&W, DeepFool) against the models
- Train adversarially robust models that resist those attacks
- Generate synthetic attack traffic with a WGAN-GP
- Visualize everything through an interactive Streamlit dashboard
- Evaluate and compare clean accuracy vs. robust accuracy

---

## 2. Architecture Overview

The system has four major components. Here is a simplified diagram of how they interact:

```
                    +------------------+
                    |  Network Traffic |
                    +--------+---------+
                             |
                    +--------v---------+
                    |   TIER 1:        |
                    |   Signature      |     Match found?
                    |   Detection      +---> YES --> ALERT (known attack)
                    +--------+---------+
                             | NO
                    +--------v---------+
                    |   TIER 2:        |
                    |   ML Detection   |     Attack detected?
                    |   (DNN/CNN/LSTM) +---> YES --> ALERT (Tier 2)
                    +--------+---------+
                             | BENIGN (might be adversarial evasion)
                    +--------v---------+
                    |   TIER 3:        |
                    |   Adversarial    |     Evasion caught?
                    |   Defense        +---> YES --> ALERT (CRITICAL)
                    +--------+---------+
                             | NO
                    +--------v---------+
                    |   BENIGN         |  (all tiers agree: safe)
                    +------------------+

                    +------------------+
                    |   GAN Module     |  (Separate: generates synthetic attack data)
                    +------------------+
```

### Tier 1: Signature Detection

How it works: Takes raw traffic features (protocol, ports, flags, byte counts) and compares them against a database of known attack patterns stored in `data/signatures/signatures.json`. Each signature has a set of conditions (e.g., "SYN flag count > 100 AND ACK flag count < 5"). If 80% or more of a signature's conditions are satisfied, the traffic is flagged as that attack type.

What it catches: SYN floods, UDP floods, Slowloris, port scans, SSH/FTP brute force, SQL injection patterns, XSS patterns, botnet C2 communication, and NSL-KDD-specific patterns (Neptune, Smurf, Satan probe, buffer overflow).

Speed: Sub-millisecond per sample. This is the fastest tier.

Implementation: `src/tier1_signature/signature_detector.py`, `signature_database.py`, `pattern_matcher.py`.

### Tier 2: ML Detection

How it works: Takes preprocessed, scaled, and feature-selected numerical vectors and runs them through trained deep learning models. Three architectures are trained and compared; the best one is selected automatically:

- **DNN (Deep Neural Network):** Four dense layers (256 -> 128 -> 64 -> 32) with batch normalization and dropout. Best for tabular data. This is the default.
- **CNN (1D Convolutional Neural Network):** Three Conv1D layers (64 -> 128 -> 256) with max pooling. Detects local patterns in feature sequences.
- **LSTM (Long Short-Term Memory):** Two LSTM layers (128 -> 64). Captures sequential dependencies.

Classification: Five classes -- Normal (0), DoS (1), Probe (2), R2L (3), U2R (4).

Speed: ~10-15ms per sample (CPU).

Implementation: `src/tier2_ml_detection/models.py`, `train.py`, `ml_detector.py`, `feature_extractor.py`.

### Tier 3: Adversarial Defense

How it works: When Tier 2 classifies traffic as benign, Tier 3 provides a second opinion using an adversarially-trained model. This catches adversarial evasion attacks -- malicious traffic that has been carefully crafted to fool Tier 2's standard ML model into misclassifying it as normal. It uses three defense strategies:

1. **Adversarial Training:** The model is retrained on a mix of clean samples (40%), FGSM adversarial samples (30%), and PGD adversarial samples (30%). This makes the model inherently robust.

2. **Input Transformation (Feature Squeezing):** The input is transformed by reducing its bit depth (rounding values to discrete levels). If the model's prediction changes significantly between the original and transformed input, the input is flagged as adversarial.

3. **Ensemble Voting:** Multiple diverse models vote on the classification. Adversarial examples tend to fool individual models differently, so ensemble disagreement indicates a potential adversarial input.

Implementation: `src/tier3_adversarial_defense/adversarial_training.py`, `input_transformation.py`, `ensemble_defense.py`, `adversarial_defense.py`.


## 3. Prerequisites

Before you begin, make sure you have the following:

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.9 | 3.10 or 3.11 |
| RAM | 8 GB | 16 GB |
| Disk space | 2 GB (code + deps) | 10 GB (with datasets) |
| OS | Windows 10, Linux, macOS | Any of these |
| GPU | Not required | NVIDIA GPU with CUDA (optional, for faster training) |

You also need:
- **pip** (comes with Python)
- **git** (to clone the repository, or you can download as ZIP)
- A terminal or command prompt

---

## 4. Installation Step by Step

### Step 1: Get the Project

Option A -- Clone with git:
```bash
git clone <repository-url>
cd adversarial-robust-ids
```

Option B -- Download as ZIP, extract, and navigate into the folder:
```bash
cd adversarial-robust-ids
```

### Step 2: Create a Virtual Environment

A virtual environment keeps this project's packages separate from your system Python.

```bash
python -m venv venv
```

### Step 3: Activate the Virtual Environment

On **Windows** (Command Prompt):
```bash
venv\Scripts\activate
```

On **Windows** (PowerShell):
```bash
.\venv\Scripts\Activate.ps1
```

On **Linux / macOS**:
```bash
source venv/bin/activate
```

You should see `(venv)` at the beginning of your terminal prompt.

### Step 4: Install PyTorch (CPU Version)

PyTorch is installed separately because it has a different index URL for the CPU-only version (which is smaller and faster to install).

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

If you have an NVIDIA GPU and want GPU acceleration, use the official PyTorch install command from  / instead.

### Step 5: Install All Other Dependencies

```bash
pip install -r requirements.txt
```

This installs TensorFlow, scikit-learn, pandas, numpy, Streamlit, and all other packages listed in `requirements.txt`.

The installation may take 5-10 minutes depending on your internet speed.

### Step 6: Verify Installation

```bash
python -c "import torch; import tensorflow; import sklearn; print('All imports OK')"
```

If this prints "All imports OK" with no errors, you are ready to proceed.

---

## 5. Dataset Setup

### Option A: NSL-KDD (Primary, Recommended)

The NSL-KDD dataset is the default. It is a cleaned version of the classic KDD Cup 1999 dataset with 41 features per network connection and five traffic classes (Normal, DoS, Probe, R2L, U2R).

**How to get it:**

1. Go to https://www.unb.ca/cic/datasets/nsl.html and download the dataset.
   - Alternatively, search "NSL-KDD dataset" on [Kaggle](https://www.kaggle.com/) where it is freely available.
2. You need two files: `KDDTrain+.txt` and `KDDTest+.txt`.
3. Create the directory if it does not exist:
   ```bash
   mkdir -p data/raw/NSL-KDD
   ```
   On Windows:
   ```bash
   mkdir data\raw\NSL-KDD
   ```
4. Place both `.txt` files in `data/raw/NSL-KDD/`:
   ```
   data/raw/NSL-KDD/
   |-- KDDTrain+.txt     (~125,000 samples)
   |-- KDDTest+.txt      (~22,000 samples)
   ```

The files are headerless CSVs with 43 columns (41 features + label + difficulty score). The data loader assigns proper column names automatically.

**Dataset size:** ~18 MB total. Small enough to train on a laptop.

### Option B: CICIDS2017 (Secondary, Larger)

The CICIDS2017 dataset is a more modern benchmark with realistic traffic from 2017. It is much larger (~5 GB) and contains different attack types.

**How to get it:**

1. Go to https://www.unb.ca/cic/datasets/ids-2017.html
2. Download the CSV files (each day's traffic is a separate CSV).
3. Place all CSV files in `data/raw/CICIDS2017/`.
4. Change `config/config.yaml`:
   ```yaml
   dataset:
     primary: "CICIDS2017"
   ```

Note: CICIDS2017 requires significantly more RAM (16 GB+) due to its size.

### Option C: No Dataset (Synthetic Data)

If you do not have a dataset, the system automatically generates synthetic data that mimics the NSL-KDD structure. This happens in both `demo` and `train` modes when no real dataset is found. Synthetic data is useful for testing that everything works, but model accuracy on real traffic will be lower.

---

## 6. Quick Start

The fastest way to see the system in action:

```bash
python main.py --mode demo
```

### What Happens During Demo

1. **Synthetic data generation:** 500 samples with NSL-KDD-like features (41 numeric + 3 categorical columns), 60% normal / 40% attack split.

2. **Preprocessing:** The data goes through cleaning, label encoding, one-hot encoding of categorical features, standard scaling, mutual-information feature selection (top 35 features), and SMOTE oversampling.

3. **Tier 1 test:** Two test samples (one attack, one benign) are checked against the signature database. The attack sample simulates a SYN flood (200 SYN flags, 2 ACK flags, 500ms flow). Expected output: `is_attack=True`, type "SYN Flood".

4. **DNN training:** A quick 5-epoch training run of the DNN model. You will see training progress with loss and accuracy per epoch.

5. **FGSM attack test:** Generates adversarial examples on 20 test samples using FGSM with epsilon=0.1. Reports the average perturbation magnitude.

6. **GAN test:** Trains the WGAN-GP for 5 epochs on attack traffic and generates 10 synthetic samples.

7. **Summary:** Prints a recap of all component results.

### Expected Output

```
============================================================
DEMO COMPLETE - All components working!
============================================================
  Synthetic Data: 500 samples
  Features Selected: 35
  Classes: 5
  DNN Val Accuracy: 0.XXXX
  Tier 1 Signature Detection: Working
  FGSM Attack: Working (avg perturbation: 0.XXXX)
  GAN Generator: Working
============================================================
```

The demo takes approximately 1-3 minutes on a modern CPU.

---

## 7. Training Models

### Full Training Command

```bash
python main.py --mode train
```

### What Happens During Training

The training pipeline executes four stages in sequence:

**Stage 1: Data Loading and Preprocessing (~1-2 minutes)**

1. Loads the NSL-KDD dataset (or generates 5000 synthetic samples if not found).
2. Cleans data: removes duplicates, replaces infinity with NaN, fills missing values with median.
3. Encodes labels: creates binary (Normal vs Attack) and multi-class (5 categories) labels.
4. One-hot encodes categorical features (protocol_type, service, flag).
5. Splits data: 70% train, 15% validation, 15% test (stratified).
6. Scales features using StandardScaler (zero mean, unit variance).
7. Selects top 35 features using mutual information scoring.
8. Applies SMOTE to balance attack classes in the training set.
9. Saves preprocessing artifacts (scaler, encoder, feature selector) to `models/preprocessing/`.

**Stage 2: Tier 2 Model Training (~5-20 minutes per model)**

Trains three models sequentially:

1. **DNN:** 4-layer deep neural network. Input is a flat feature vector.
2. **CNN:** 1D convolutional network. Input is reshaped to (features, 1).
3. **LSTM:** Recurrent network. Input is reshaped to (1, features).

Each model trains for up to 30 epochs with:
- Early stopping (patience=10 on validation loss)
- Learning rate reduction on plateau (factor=0.5, patience=5)
- Model checkpointing (saves best model by validation accuracy)

After all three are trained, the system automatically selects the one with the highest validation accuracy and saves it to `models/tier2/best_model.h5`.

**Stage 3: GAN Training (~5-15 minutes)**

Trains the WGAN-GP on attack traffic samples only:
- 100 epochs (configurable)
- Discriminator is updated 5 times per generator update (n_critic=5)
- Gradient penalty weight: 10
- Saves generator checkpoints every 25 epochs to `models/gan/`
- Final weights saved as `generator_final.pth` and `discriminator_final.pth`

**Stage 4: Adversarial Training for Tier 3 (~10-30 minutes)**

1. Creates a PyTorch version of the DNN model.
2. For each training batch:
   - Generates FGSM adversarial examples (epsilon=0.1)
   - Generates PGD adversarial examples (epsilon=0.1, 10 iterations)
   - Mixes: 40% clean + 30% FGSM + 30% PGD
   - Trains on the mixed batch
3. Runs for 20 epochs.
4. Saves the best model (by validation accuracy) to `models/tier3/robust_model.pth`.

### Expected Total Training Time

| Hardware | Approximate Time |
|----------|-----------------|
| Laptop CPU (Intel i5/i7) | 30-60 minutes |
| Desktop CPU (Ryzen/i7) | 20-40 minutes |
| With NVIDIA GPU | 10-20 minutes |

### Training Output

All trained models are saved to the `models/` directory:
```
models/
|-- preprocessing/
|   |-- scaler.pkl
|   |-- label_encoder.pkl
|   |-- feature_selector.pkl
|-- tier2/
|   |-- dnn_model.h5
|   |-- cnn_model.h5
|   |-- lstm_model.h5
|   |-- best_model.h5          <-- Best of the three
|-- tier3/
|   |-- robust_model.pth       <-- Adversarially trained model
|-- gan/
|   |-- generator_final.pth
|   |-- discriminator_final.pth
|   |-- generator_epoch_25.pth  (checkpoints)
|   |-- generator_epoch_50.pth
|   |-- ...
```

---

## 8. Running Detection

### Demo Mode (No Trained Models Required)

```bash
python main.py --mode demo
```

See [Section 6](#6-quick-start) for details.

### Batch Detection on a CSV File

```bash
python main.py --mode detect --input path/to/traffic_data.csv
```

This requires trained models (run `python main.py --mode train` first).

The CSV file should contain network traffic features matching the dataset schema (e.g., NSL-KDD 41 features or CICIDS2017 features). The system will preprocess and run the full three-tier pipeline on each row.

### Understanding Detection Output

Each detection result is a dictionary containing:

| Field | Example | Meaning |
|-------|---------|---------|
| `tier` | `1`, `2`, or `3` | Which tier made the final decision |
| `attack_type` | `"SYN Flood"`, `"DoS"`, `"BENIGN"` | The detected attack type |
| `severity` | `"CRITICAL"`, `"HIGH"`, `"MEDIUM"`, `"LOW"` | Threat severity |
| `confidence` | `0.95` | Model's confidence in the prediction (0.0 to 1.0) |
| `is_adversarial` | `true` or `false` | Whether Tier 3 flagged this as adversarial |
| `priority` | `1` (highest) to `4` (lowest) | Alert priority score |
| `total_time_ms` | `15.3` | Total detection time in milliseconds |

**How severity is determined:**

- `CRITICAL`: Adversarial attack detected (Tier 3) or known critical signature (Tier 1)
- `HIGH`: High-confidence ML detection (>90%) or high-severity signature match
- `MEDIUM`: Moderate-confidence ML detection or medium-severity signature
- `LOW`: Low-confidence detection

**How priority is calculated:**

- Priority 1: Adversarial attacks or critical signature matches
- Priority 2: High-severity detections
- Priority 3: Medium-severity detections
- Priority 4: Everything else

---

## 9. Using the Dashboard

The dashboard provides a visual interface to the IDS system.

### Launching the Dashboard

Using main.py:
```bash
python main.py --mode dashboard
```

Or directly with Streamlit:
```bash
streamlit run src/dashboard/app.py
```

Both commands start the dashboard at `http://localhost:8501`. Open this URL in your browser.

### Dashboard Modes

Use the sidebar dropdown to switch between four modes:

#### Mode 1: Real-time Monitor

What you see:
- **Top row:** Five KPI metrics -- total traffic count, attacks detected, adversarial attacks, false positive rate, system uptime.
- **Left panel:** A time-series line chart showing traffic classified as Benign, Attacks, and Adversarial over the last 60 seconds.
- **Right panel:** A scrollable alert feed showing the most recent detections with severity indicators, tier number, attack type, and timestamp.
- **Bottom:** Tier-wise breakdown showing detection counts, average speed, and detection rate for each tier.

#### Mode 2: Batch Analysis

What you see:
- A file upload widget. Upload a CSV file containing network traffic features.
- After upload, a preview of the first 10 rows.
- A "Run Detection" button that processes all rows through the three-tier pipeline.
- Results showing counts of benign, Tier 1, Tier 2, and Tier 3 detections.

#### Mode 3: Attack Simulation

What you see:
- A dropdown to select the attack type (FGSM, PGD, C&W, DeepFool).
- A slider to set the perturbation epsilon (0.01 to 0.5).
- A number input for sample count (10 to 1000).
- A "Launch Simulation" button that generates adversarial samples.
- Results showing attack success rate vs. Tier 3 detection rate, displayed as both JSON and a grouped bar chart.

#### Mode 4: Model Performance

What you see:
- A comparison table of four system configurations:
  - Signature Only
  - ML Only (Non-Robust)
  - Dual-Tier (Non-Robust)
  - Our Three-Tier Robust system
- Metrics: Accuracy, FPR, Detection Rate, Robust Accuracy, Adversarial Detection Rate.
- A grouped bar chart comparing clean accuracy vs. robust accuracy for each configuration.

---

## 10. Running Evaluation

```bash
python main.py --mode evaluate
```

### What It Does

1. Loads the dataset (or generates synthetic data).
2. Preprocesses it using the same pipeline as training.
3. Loads the trained Tier 2 model from `models/tier2/best_model.h5`.
4. Runs the model on the test set and computes:
   - **Accuracy:** Percentage of correct predictions.
   - **Precision:** Of samples predicted as attacks, how many actually are attacks.
   - **Recall:** Of actual attacks, how many were correctly detected.
   - **F1-Score:** Harmonic mean of precision and recall.
   - **FPR (False Positive Rate):** Percentage of normal traffic incorrectly flagged as attacks.
   - **FNR (False Negative Rate):** Percentage of attacks incorrectly classified as normal.
   - **AUC-ROC:** Area under the ROC curve (if probability outputs are available).
   - **Confusion Matrix:** Full breakdown of predictions vs. ground truth.
5. Saves all results to `evaluation_results/evaluation_results.json`.

### Requirements

You must have a trained model. If `models/tier2/best_model.h5` does not exist, the command will print an error asking you to run training first.

---

## 11. Running Tests

The test suite validates each component of the system independently and together.

```bash
pytest tests/ -v
```

### Available Test Files

| File | What It Tests |
|------|--------------|
| `test_preprocessing.py` | Data loading, cleaning, encoding, scaling, feature selection, SMOTE |
| `test_tier1.py` | Signature matching, database loading, known attack detection |
| `test_tier2.py` | DNN/CNN/LSTM model building, training, and prediction |
| `test_attacks.py` | FGSM, PGD attack generation and perturbation validation |
| `test_gan.py` | Generator/discriminator construction, WGAN-GP training loop |
| `test_tier3.py` | Adversarial detection, input transformation, ensemble voting |
| `test_integration.py` | Full three-tier pipeline end-to-end |
| `test_performance.py` | Timing benchmarks and resource usage |

### Running Specific Tests

```bash
# Run only Tier 1 tests
pytest tests/test_tier1.py -v

# Run only GAN tests
pytest tests/test_gan.py -v

# Run with test coverage report
pytest tests/ -v --cov=src --cov-report=html
```

### Shared Test Fixtures

The file `tests/conftest.py` contains shared pytest fixtures (e.g., synthetic data, config loading) used across all test modules.

---

## 12. Configuration Reference

All configuration is in `config/config.yaml`. Below is every field explained.

### system

```yaml
system:
  name: "Adversarial Robust IDS"    # System name (for logging)
  version: "1.0"                     # Version string
  log_level: "INFO"                  # Logging level: DEBUG, INFO, WARNING, ERROR
  log_file: "logs/ids.log"           # Log file path (relative to project root)
```

### dataset

```yaml
dataset:
  primary: "NSL-KDD"                # Primary dataset: "NSL-KDD" or "CICIDS2017"
  path: "data/raw/NSL-KDD"          # Path to primary dataset directory
  secondary: "CICIDS2017"           # Secondary dataset name
  secondary_path: "data/raw/CICIDS2017"  # Path to secondary dataset
  train_split: 0.70                  # Fraction of data for training
  val_split: 0.15                    # Fraction of data for validation
  test_split: 0.15                   # Fraction of data for testing
  random_seed: 42                    # Random seed for reproducibility
```

### preprocessing

```yaml
preprocessing:
  handle_missing: "median"           # How to fill missing values: "median", "mean", or "drop"
  encoding: "onehot"                 # Categorical encoding: "onehot" or "label"
  scaling: "standard"                # Feature scaling: "standard" (zero mean) or "minmax" (0-1 range)
  imbalance_method: "smote"          # Class balancing: "smote" or "none"
  top_features: 35                   # Number of features to select via mutual information
```

### tier1

```yaml
tier1:
  enabled: true                      # Enable/disable Tier 1
  signature_db: "data/signatures/signatures.json"  # Path to signature database
  match_threshold: 0.8               # Fraction of conditions that must match (0.0-1.0)
```

### tier2

```yaml
tier2:
  enabled: true                      # Enable/disable Tier 2
  model_type: "DNN"                  # Model type: "DNN", "CNN", or "LSTM"
  model_path: "models/tier2/best_model.h5"  # Where to save/load the best model
  confidence_threshold: 0.5          # Minimum confidence to flag as attack (0.0-1.0)
  training:
    epochs: 30                       # Maximum training epochs
    batch_size: 128                  # Samples per training batch
    learning_rate: 0.001             # Adam optimizer learning rate
    optimizer: "adam"                # Optimizer (currently only adam is used)
    early_stopping_patience: 10      # Stop if val_loss doesn't improve for N epochs
    dropout_rate: 0.3                # Dropout probability (0.0-1.0)
```

### tier3

```yaml
tier3:
  enabled: true                      # Enable/disable Tier 3
  robust_model_path: "models/tier3/robust_model.pth"  # Adversarially trained model path
  ensemble_models:                   # Paths to ensemble member models
    - "models/tier3/ensemble_dnn.h5"
    - "models/tier3/ensemble_cnn.h5"
    - "models/tier3/ensemble_lstm.h5"
  adversarial_detection_threshold: 0.1  # Feature squeezing distance threshold
  defense_methods:                   # Active defense methods
    - "adversarial_training"
    - "input_transformation"
    - "ensemble_voting"
```

### gan

```yaml
gan:
  latent_dim: 100                    # Size of random noise input to generator
  generator_hidden: [128, 256, 512]  # Generator hidden layer sizes
  discriminator_hidden: [512, 256]   # Discriminator hidden layer sizes
  epochs: 100                        # GAN training epochs
  batch_size: 64                     # GAN training batch size
  learning_rate: 0.0002              # Adam learning rate for both G and D
  beta1: 0.5                         # Adam beta1 parameter
  beta2: 0.999                       # Adam beta2 parameter
  gradient_penalty_weight: 10        # Lambda for gradient penalty term
  save_interval: 25                  # Save generator checkpoint every N epochs
```

### adversarial_attacks

```yaml
adversarial_attacks:
  fgsm:
    epsilon: 0.1                     # FGSM perturbation magnitude (0.01-0.3 typical)
  pgd:
    epsilon: 0.1                     # PGD max perturbation
    alpha: 0.01                      # PGD step size per iteration
    num_iterations: 40               # Number of PGD steps
  cw:
    confidence: 0.0                  # C&W attack confidence parameter
    max_iterations: 100              # C&W optimization iterations
    learning_rate: 0.01              # C&W optimizer step size
  deepfool:
    max_iterations: 50               # DeepFool max iterations
```

### adversarial_training

```yaml
adversarial_training:
  epochs: 20                         # Adversarial training epochs
  clean_ratio: 0.4                   # Fraction of clean samples in each batch
  attack_mix:                        # How to split the adversarial portion
    fgsm: 0.3                        # 30% of adversarial samples from FGSM
    pgd: 0.3                         # 30% from PGD
    cw: 0.2                          # 20% from C&W (placeholder in current implementation)
    deepfool: 0.2                    # 20% from DeepFool (placeholder)
```

### dashboard

```yaml
dashboard:
  host: "0.0.0.0"                   # Dashboard bind address
  port: 8501                         # Dashboard port
  refresh_interval: 2                # Auto-refresh interval in seconds
```

---

## 13. Code Walkthrough

This section explains what each source file does, organized by module.

### src/utils/

**`config.py`** -- Loads the YAML configuration file and resolves relative paths to absolute paths using the project root directory. Two key functions:
- `load_config(config_path)`: Reads and parses the YAML file.
- `resolve_path(relative_path)`: Converts a relative path like `"models/tier2/best_model.h5"` to an absolute path.

**`logger.py`** -- Sets up Python logging with a consistent format (`timestamp | name | level | message`). Supports both console output and optional file logging. Called by all major modules.

### src/preprocessing/

**`data_loader.py`** -- The `DataLoader` class handles loading data from three sources:
- `load_nsl_kdd(path)`: Reads `KDDTrain+.txt` and `KDDTest+.txt`, assigns the 43 column names, drops the difficulty column, and maps attack names (e.g., "neptune", "smurf") to categories (DoS, Probe, R2L, U2R, Normal).
- `load_cicids2017(path)`: Reads all CSV files in the directory and maps labels.
- `generate_synthetic(n_samples)`: Creates random data with the same structure as NSL-KDD. Useful for testing.

**`preprocessor.py`** -- The `DataPreprocessor` class implements the full pipeline:
1. `clean_data()`: Remove duplicates, replace infinities, fill missing values.
2. `encode_labels()`: Create binary and multi-class integer labels.
3. `encode_categorical()`: One-hot encode protocol_type, service, flag.
4. `split_data()`: 70/15/15 stratified split.
5. `scale_features()`: Fit StandardScaler on train, transform all splits.
6. `select_features()`: SelectKBest with mutual information, keep top-k.
7. `handle_imbalance()`: SMOTE oversampling on training set.
8. `run_pipeline()`: Runs all steps in order, returns a dict with arrays and metadata.

**`feature_engineering.py`** -- Standalone functions for feature analysis:
- `compute_feature_importance()`: Uses mutual information, chi-squared, or random forest to rank features.
- `get_top_features()`: Returns names and indices of the top-k features.
- `get_correlation_matrix()`: Computes feature correlation for analysis.

### src/tier1_signature/

**`signature_detector.py`** -- The `Tier1SignatureDetector` class:
- Loads the signature JSON database on initialization.
- `detect(traffic_sample)`: Iterates through all signature categories. For each signature, checks what fraction of conditions are met. If >= 80% (configurable), returns an alert with attack type, severity, confidence=CERTAIN, tier=1.
- `detect_batch(batch)`: Runs `detect()` on each sample.

**`signature_database.py`** -- The `SignatureDatabase` class for managing signatures:
- CRUD operations: load, add, validate, save.
- Handles both flat signature lists and nested structures (like `nsl_kdd_rules`).

**`pattern_matcher.py`** -- The `PatternMatcher` class implements the matching algorithm:
- Supports three rule types: range-based (`{min: X, max: Y}`), list-based (`[val1, val2]`), and exact match.
- Returns both a boolean match result and the match ratio (useful for ranking).

### src/tier2_ml_detection/

**`models.py`** -- Defines three TensorFlow/Keras model architectures:
- `build_dnn(input_dim, num_classes)`: Sequential model with Dense layers.
- `build_cnn(input_shape, num_classes)`: Sequential model with Conv1D layers.
- `build_lstm(input_shape, num_classes)`: Sequential model with LSTM layers.
- `get_training_callbacks(model_path)`: Returns EarlyStopping, ReduceLROnPlateau, ModelCheckpoint.

**`train.py`** -- The `Tier2Trainer` class:
- `train_model(model_type, ...)`: Builds, trains, and evaluates a single model type.
- `train_all_models(data_dict)`: Trains DNN, CNN, LSTM sequentially. Selects the best by validation accuracy. Saves it to `best_model.h5`.

**`ml_detector.py`** -- The `Tier2MLDetector` class for inference:
- Loads a saved Keras model.
- `detect(features)`: Takes a preprocessed feature vector, predicts class probabilities, returns attack type, confidence, tier=2.
- `detect_batch(features_batch)`: Batch version.
- Maps class IDs to names: 0=BENIGN, 1=DoS, 2=Probe, 3=R2L, 4=U2R.

**`feature_extractor.py`** -- The `FeatureExtractor` class reshapes input arrays for each model type:
- DNN: (samples, features) -- no reshaping needed.
- CNN: (samples, features, 1) -- adds a channel dimension.
- LSTM: (samples, 1, features) -- adds a timestep dimension.

### src/adversarial_attacks/

**`fgsm.py`** -- Implements the Fast Gradient Sign Method. One-step attack: compute loss gradient with respect to input, take the sign, multiply by epsilon, add to input.

**`pgd.py`** -- Implements Projected Gradient Descent. Multi-step attack: similar to FGSM but iteratively applies small steps (alpha) and projects back into the epsilon-ball around the original input.

**`cw_attack.py`** -- Carlini & Wagner L2 attack using the Adversarial Robustness Toolbox (ART) library. Optimizes a special objective to find the smallest perturbation that changes the prediction.

**`deepfool.py`** -- DeepFool attack using the ART library. Iteratively finds the closest decision boundary and computes the minimal perturbation to cross it.

**`attack_utils.py`** -- Shared utilities:
- `PyTorchDNN`: A PyTorch mirror of the Keras DNN architecture. Needed because adversarial attacks require gradient computation via PyTorch.
- `evaluate_attack()`: Computes attack success rate, L2 perturbation, L-infinity perturbation.
- `generate_mixed_adversarial_dataset()`: Creates a mixed dataset with 30% FGSM, 30% PGD, 20% C&W, 20% DeepFool samples.

### src/tier3_adversarial_defense/

**`adversarial_training.py`** -- The `AdversarialTrainer` class:
- `train_epoch(dataloader)`: For each batch, generates FGSM and PGD adversarial examples on the fly, mixes them with clean samples (40% clean, 30% FGSM, 30% PGD), and trains the model.
- `train(train_loader, val_loader)`: Full training loop with best model saving.

**`input_transformation.py`** -- Three transformation functions:
- `bit_depth_reduction(x, depth)`: Rounds feature values to 2^depth discrete levels.
- `gaussian_smoothing(x, sigma)`: Adds small random noise to mask precise perturbations.
- `feature_squeezing(x)`: Applies bit depth reduction (the primary defense).

**`ensemble_defense.py`** -- The `EnsembleDefense` class:
- `ModelWrapper`: Unified predict interface for Keras and PyTorch models.
- `predict(x)`: Averages probability distributions from all models (soft voting). Returns predictions, confidence, and agreement score.

**`adversarial_defense.py`** -- The `Tier3AdversarialDefense` class (main Tier 3 module):
- `detect_adversarial(x)`: Compares prediction on original input vs. feature-squeezed input. Large difference = adversarial.
- `detect_and_classify(x)`: Full pipeline -- detect adversarial + ensemble classify + severity scoring.

### src/integration/

**`ids_pipeline.py`** -- The `AdversarialRobustIDS` class ties everything together:
- Initializes all three tiers from config.
- `detect(traffic_sample)`: Runs the three-tier pipeline (see [Section 14](#14-how-the-detection-flow-works)).
- `detect_batch(samples)`: Processes multiple samples.
- `get_statistics()`: Returns detection counts per tier.

**`alert_manager.py`** -- The `AlertManager` class:
- `create_alert(tier, attack_type, severity, ...)`: Creates an alert dict with UUID, timestamp, priority score. Logs to `logs/alerts.log`.
- `_calculate_priority()`: Assigns priority 1-4 based on tier, severity, and adversarial flag.
- `get_recent_alerts(n)` and `get_alert_summary()`: Query functions.

### src/evaluation/

**`metrics.py`** -- Metric computation functions:
- `compute_all_metrics()`: Computes accuracy, precision, recall, F1, FPR, FNR, AUC-ROC, confusion matrix.
- `compute_robust_accuracy()`: Compares clean accuracy vs. adversarial accuracy.

**`evaluator.py`** -- The `SystemEvaluator` class:
- `evaluate_clean()`: Evaluate on unperturbed test data.
- `evaluate_robust()`: Evaluate clean vs. adversarial accuracy.
- `compare_baselines()`: Compare different system configurations.
- `save_results()`: Write results to JSON.

**`visualizations.py`** -- Plotting functions using matplotlib and seaborn:
- Confusion matrix heatmaps
- Multi-class ROC curves
- Epsilon sensitivity plots (accuracy vs. perturbation magnitude)
- Tier breakdown pie charts
- Training history (loss/accuracy curves)
- Baseline comparison bar charts
- Feature importance bar charts

### src/dashboard/

**`app.py`** -- The Streamlit application entry point. Defines the page layout with a sidebar for mode selection and four main content areas.

**`components.py`** -- Reusable dashboard components:
- `render_metric_row()`: Top KPI cards.
- `render_alert_feed()`: Scrollable alert list.
- `render_tier_breakdown()`: Three-column tier statistics.
- `render_attack_simulation()`: Attack parameter controls and result visualization.
- `render_model_performance()`: Baseline comparison table and chart.

### main.py

The entry point. Parses command-line arguments (`--mode`, `--input`, `--config`) and dispatches to the appropriate function:
- `demo(config)`: Quick demo with synthetic data.
- `train(config)`: Full training pipeline.
- `evaluate(config)`: Run evaluation.
- `dashboard(config)`: Launch Streamlit.
- `detect`: Batch detection on CSV (requires `--input` argument).

---

## 14. How the Detection Flow Works

Here is exactly what happens when a piece of network traffic enters the system, step by step.

### Step 1: Traffic Arrives

The input is either:
- A **dictionary** of raw features (e.g., `{protocol: TCP, syn_flag_count: 200, ...}`) for Tier 1
- A **numpy array** of preprocessed features (after scaling and feature selection) for Tier 2/3

### Step 2: Tier 1 -- Signature Check

If Tier 1 is enabled and the input is a raw dictionary:

1. The system loads the signature database from `signatures.json`.
2. It iterates through every signature category (dos_attacks, port_scan, brute_force, web_attacks, botnet, nsl_kdd_rules).
3. For each signature, it checks every condition:
   - Range conditions: Is the value within min/max bounds?
   - List conditions: Is the value in the allowed list?
   - Exact conditions: Does the value match exactly?
4. If >= 80% of conditions match --> **ALERT: Known attack detected.** The detection stops here. An alert is created with tier=1, confidence=CERTAIN, and the specific attack name and severity.
5. If no signature matches --> Continue to Tier 2.

### Step 3: Tier 2 -- ML Classification

If Tier 2 is enabled and preprocessed features are available:

1. The feature vector is reshaped for the model type (DNN=flat, CNN=add channel dim, LSTM=add timestep dim).
2. The Keras model predicts class probabilities.
3. The predicted class is the one with the highest probability.
4. If an attack is detected with sufficient confidence (class != 0 and confidence >= threshold) --> **ALERT: Attack detected (Tier 2).** An alert is created with the attack type and confidence.
5. If the predicted class is 0 (Normal) or the confidence is below the threshold (default 0.5) --> Tier 2 says benign, but this might be an adversarial evasion attack. Continue to Tier 3.

### Step 4: Tier 3 -- Adversarial Evasion Check

If Tier 3 is enabled, it examines samples that Tier 2 classified as benign. The purpose is to catch adversarial evasion attacks -- malicious traffic crafted to fool Tier 2's standard ML model.

1. **Adversarial Detection (Feature Squeezing):**
   - The original feature vector is passed to the adversarially-trained robust model to get prediction A.
   - The feature vector is "squeezed" (bit-depth reduced to 16 levels).
   - The squeezed vector is passed to the robust model to get prediction B.
   - The L1 distance between prediction A and prediction B is computed.
   - If the distance exceeds the threshold (default 0.1), the input is flagged as adversarial.

2. **Robust Classification (Ensemble Voting):**
   - All ensemble models independently predict class probabilities.
   - The probabilities are averaged (soft voting).
   - The final prediction is the class with the highest averaged probability.
   - The agreement score measures how many models agree on the prediction.

3. **Decision:**
   - If adversarial evasion is detected OR the robust model reclassifies the sample as an attack --> **ALERT (Tier 3):** Adversarial evasion caught.
   - Severity: CRITICAL if adversarial, HIGH if reclassified as attack.
   - If the robust model also agrees it is benign --> **BENIGN: All tiers confirm safe traffic.**

4. **Alert Creation:**
   - An alert is generated with tier=3, the attack type, severity, confidence, adversarial flag, priority score, and timestamp.

### Step 5: Alert Output

The final alert dict is returned and also logged to `logs/alerts.log` with a unique ID and the detection time in milliseconds.

---

## 15. Troubleshooting

### "ModuleNotFoundError: No module named 'src'"

**Cause:** Python cannot find the `src` package. This usually means you are running the script from the wrong directory.

**Fix:** Always run from the project root:
```bash
cd adversarial-robust-ids
python main.py --mode demo
```

### "ModuleNotFoundError: No module named 'torch'"

**Cause:** PyTorch is not installed.

**Fix:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### "ModuleNotFoundError: No module named 'tensorflow'"

**Cause:** TensorFlow is not installed.

**Fix:**
```bash
pip install tensorflow
```

### "FileNotFoundError: NSL-KDD dataset not found"

**Cause:** The dataset files are not in the expected directory.

**Fix:** Place `KDDTrain+.txt` and `KDDTest+.txt` in `data/raw/NSL-KDD/`. Alternatively, the system will use synthetic data if you run demo or train mode.

### "No trained model found at models/tier2/best_model.h5"

**Cause:** You are trying to run evaluation or detection before training.

**Fix:** Run training first:
```bash
python main.py --mode train
```

### "CUDA out of memory" or slow training

**Cause:** Not enough GPU memory, or running on CPU when a GPU is expected.

**Fix:**
- For CPU-only usage, the system automatically detects `cuda` availability and falls back to CPU. No action needed.
- If you have a GPU but it is running out of memory, reduce the batch size in `config/config.yaml`:
  ```yaml
  tier2:
    training:
      batch_size: 64    # Reduce from 128
  ```

### "ValueError: Found input variables with inconsistent numbers of samples"

**Cause:** Data preprocessing produced mismatched array sizes, often due to a corrupted or incorrectly formatted dataset.

**Fix:** Verify your dataset files are complete. For NSL-KDD, each row should have 43 comma-separated values. Delete `data/processed/` and `models/preprocessing/` to force reprocessing:
```bash
# Windows
del /q data\processed\*
del /q models\preprocessing\*

# Linux/Mac
rm -f data/processed/* models/preprocessing/*
```

### "Warning: ART library not available. Returning original inputs."

**Cause:** The Adversarial Robustness Toolbox is not installed, so C&W and DeepFool attacks cannot run.

**Fix:**
```bash
pip install adversarial-robustness-toolbox
```

### Dashboard shows "Connection refused" or blank page

**Cause:** Streamlit is not running or is running on a different port.

**Fix:**
```bash
streamlit run src/dashboard/app.py --server.port 8501
```

Then open `http://localhost:8501` in your browser.

### Streamlit error: "No module named 'src.dashboard.components'"

**Cause:** Streamlit is not being run from the project root.

**Fix:** Run Streamlit from the project root directory:
```bash
cd adversarial-robust-ids
streamlit run src/dashboard/app.py
```

### SMOTE fails with "Expected n_neighbors <= n_samples_fit"

**Cause:** One of the attack classes has very few samples (fewer than k_neighbors+1).

**Fix:** The preprocessor already handles this by setting `k_neighbors=min(3, min_class_size - 1)`. If it still fails, increase the synthetic data size or reduce the number of attack classes.

---

## 16. Glossary

**Adversarial Example:** An input that has been intentionally modified with small, often imperceptible perturbations to cause a machine learning model to make an incorrect prediction.

**AUC-ROC (Area Under the Receiver Operating Characteristic Curve):** A metric that measures how well a model distinguishes between classes across all possible classification thresholds. Ranges from 0.0 (worst) to 1.0 (perfect).

**Batch Normalization:** A technique that normalizes the inputs to each layer during training, stabilizing and accelerating the learning process.

**C&W Attack (Carlini & Wagner):** An optimization-based adversarial attack that finds the smallest perturbation (in L2 norm) that causes misclassification. Slower but more effective than FGSM.

**CICIDS2017:** A modern network intrusion detection dataset created by the Canadian Institute for Cybersecurity in 2017, containing realistic network traffic with labeled attacks.

**CNN (Convolutional Neural Network):** A neural network architecture that uses convolutional filters to detect local patterns. In this project, 1D convolutions are applied to feature vectors.

**DeepFool:** An adversarial attack that iteratively finds the closest decision boundary and computes the minimal perturbation needed to cross it.

**DNN (Deep Neural Network):** A feedforward neural network with multiple hidden layers of fully connected (dense) neurons.

**DoS (Denial of Service):** An attack that aims to make a network service unavailable by overwhelming it with traffic.

**Dropout:** A regularization technique that randomly deactivates a fraction of neurons during training to prevent overfitting.

**Early Stopping:** A training technique that stops training when the validation loss stops improving, preventing overfitting.

**Ensemble Voting:** A defense strategy where multiple models independently classify the input and their predictions are combined (averaged). Adversarial perturbations that fool one model may not fool others.

**Epsilon:** The maximum allowed perturbation magnitude in adversarial attacks. Larger epsilon means stronger (but more detectable) attacks.

**Feature Squeezing:** A defense technique that transforms the input (e.g., reducing bit depth) and compares predictions before and after. If the predictions differ significantly, the input may be adversarial.

**FGSM (Fast Gradient Sign Method):** A single-step adversarial attack that adds a perturbation in the direction of the loss gradient's sign, scaled by epsilon.

**FNR (False Negative Rate):** The fraction of actual attacks that were incorrectly classified as benign. Lower is better.

**FPR (False Positive Rate):** The fraction of normal traffic that was incorrectly flagged as an attack. Lower is better.

**GAN (Generative Adversarial Network):** A framework where two neural networks (generator and discriminator) compete: the generator creates fake data, and the discriminator tries to distinguish fake from real. Over time, the generator learns to produce realistic data.

**Gradient Penalty:** A regularization term in WGAN-GP training that penalizes the discriminator's gradient norm deviating from 1, enforcing the Lipschitz constraint for stable training.

**IDS (Intrusion Detection System):** A system that monitors network traffic for suspicious activity and generates alerts when potential attacks are detected.

**LSTM (Long Short-Term Memory):** A type of recurrent neural network that can learn long-term dependencies in sequential data using memory cells and gating mechanisms.

**Mutual Information:** A measure of how much information a feature provides about the target variable. Used for feature selection -- features with high mutual information are most useful for classification.

**NSL-KDD:** A benchmark dataset for network intrusion detection, derived from the KDD Cup 1999 dataset with improvements (removal of duplicates and better balance).

**PGD (Projected Gradient Descent):** A multi-step iterative adversarial attack. More powerful than FGSM because it takes multiple small steps (each of size alpha) and projects back into the epsilon-ball after each step.

**Probe:** A category of network attack that scans ports or hosts to gather information about the network (e.g., nmap, portsweep).

**R2L (Remote to Local):** A category of attack where an attacker gains unauthorized access from a remote machine (e.g., password guessing, exploiting FTP).

**SMOTE (Synthetic Minority Over-sampling Technique):** A technique that creates synthetic training examples for underrepresented classes by interpolating between existing samples. Used to balance the training dataset.

**Softmax:** An activation function that converts a vector of raw scores into a probability distribution (values between 0 and 1 that sum to 1).

**StandardScaler:** A preprocessing technique that transforms features to have zero mean and unit variance.

**U2R (User to Root):** A category of attack where a normal user exploits vulnerabilities to gain root/administrator access (e.g., buffer overflow).

**WGAN-GP (Wasserstein GAN with Gradient Penalty):** A variant of GAN that uses the Wasserstein distance (Earth Mover's Distance) instead of JS divergence, combined with gradient penalty instead of weight clipping. This produces more stable training and avoids mode collapse.
