<<<<<<< HEAD
# Dual-Tier Intrusion Detection System

BTech Final Year Project

## Description
A hybrid intrusion detection system combining signature-based and machine learning-based detection.

## Features
- Tier 1: Signature-based detection
- Tier 2: ML-based anomaly detection
- Real-time monitoring dashboard
- Alert management system

## Tech Stack
- Python 3.8+
- Scikit-learn
- Flask
- PostgreSQL

## Project Status
Week 1: Foundation & Literature Review (In Progress)

## Author
Onkar Bhagwat
=======
# Adversarially Robust Intrusion Detection System

A three-tier network intrusion detection system (IDS) that combines signature-based detection, machine learning classification, and adversarial robustness defenses. Built with TensorFlow, PyTorch, and Streamlit.

## Overview

Traditional IDS systems are vulnerable to adversarial attacks -- carefully crafted inputs that fool ML classifiers into misclassifying malicious traffic as benign. This project addresses that problem with a layered architecture:

- **Tier 1 (Signature Detection):** Fast pattern matching against a database of known attack signatures.
- **Tier 2 (ML Detection):** Deep learning models (DNN, CNN, LSTM) trained to classify network traffic into Normal, DoS, Probe, R2L, or U2R categories.
- **Tier 3 (Adversarial Defense):** When Tier 2 classifies traffic as benign, Tier 3 uses an adversarially-trained model to catch evasion attacks -- malicious traffic crafted to fool standard ML models. Uses adversarial training, input transformation, and ensemble voting.
- **GAN Module:** A Wasserstein GAN with Gradient Penalty (WGAN-GP) that generates synthetic attack traffic for data augmentation and testing.

## Features

- Three-tier detection pipeline with configurable enable/disable per tier
- Signature database with rules for DoS, port scan, brute force, web attacks, botnet, and NSL-KDD-specific patterns
- Three ML architectures: DNN, 1D-CNN, and LSTM with automatic best-model selection
- Four adversarial attack implementations: FGSM, PGD, C&W (L2), and DeepFool
- Three defense strategies: adversarial training, input transformation (feature squeezing), and ensemble voting
- WGAN-GP for generating realistic synthetic attack traffic
- SMOTE-based class imbalance handling
- Mutual-information-based feature selection (top-k features)
- Full evaluation suite: accuracy, precision, recall, F1, FPR, FNR, AUC-ROC, confusion matrices
- Interactive Streamlit dashboard with real-time monitoring, batch analysis, attack simulation, and model comparison
- Alert management with priority scoring and logging
- Works with NSL-KDD and CICIDS2017 datasets, or synthetic data for demo/testing
- Comprehensive configuration via a single YAML file

## Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager
- 16 GB RAM recommended (8 GB minimum for demo mode)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd adversarial-robust-ids

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install PyTorch (CPU version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install all other dependencies
pip install -r requirements.txt
```

### Run the Demo (No Dataset Required)

```bash
python main.py --mode demo
```

This generates synthetic data and runs all components: preprocessing, Tier 1 signature detection, DNN training (5 epochs), FGSM adversarial attack, and GAN sample generation.

## Project Structure

```
adversarial-robust-ids/
|-- main.py                          # Entry point (train, evaluate, detect, dashboard, demo)
|-- config/
|   |-- config.yaml                  # All system configuration
|-- data/
|   |-- raw/                         # Place datasets here (NSL-KDD/, CICIDS2017/)
|   |-- processed/                   # Preprocessed data (auto-generated)
|   |-- adversarial/                 # Generated adversarial samples
|   |-- signatures/
|       |-- signatures.json          # Tier 1 signature database
|-- src/
|   |-- preprocessing/
|   |   |-- data_loader.py           # Load NSL-KDD, CICIDS2017, or synthetic data
|   |   |-- preprocessor.py          # Full preprocessing pipeline
|   |   |-- feature_engineering.py   # Feature importance and selection
|   |-- tier1_signature/
|   |   |-- signature_detector.py    # Tier 1 main detector
|   |   |-- signature_database.py    # Signature DB management
|   |   |-- pattern_matcher.py       # Pattern matching algorithms
|   |-- tier2_ml_detection/
|   |   |-- models.py                # DNN, CNN, LSTM model definitions
|   |   |-- train.py                 # Training pipeline for all three models
|   |   |-- ml_detector.py           # Inference-time ML detector
|   |   |-- feature_extractor.py     # Input reshaping for each model type
|   |-- tier3_adversarial_defense/
|   |   |-- adversarial_training.py  # Adversarial training loop (FGSM + PGD mix)
|   |   |-- input_transformation.py  # Feature squeezing, bit-depth reduction
|   |   |-- ensemble_defense.py      # Ensemble soft voting defense
|   |   |-- adversarial_defense.py   # Tier 3 main module combining all defenses
|   |-- adversarial_attacks/
|   |   |-- fgsm.py                  # Fast Gradient Sign Method
|   |   |-- pgd.py                   # Projected Gradient Descent
|   |   |-- cw_attack.py             # Carlini & Wagner L2 attack
|   |   |-- deepfool.py              # DeepFool minimal perturbation attack
|   |   |-- attack_utils.py          # PyTorch DNN wrapper, attack evaluation
|   |-- gan_generator/
|   |   |-- generator.py             # Generator network
|   |   |-- discriminator.py         # Discriminator network
|   |   |-- gan_model.py             # WGAN-GP wrapper
|   |   |-- train_gan.py             # WGAN-GP training with gradient penalty
|   |-- integration/
|   |   |-- ids_pipeline.py          # Main three-tier pipeline orchestrator
|   |   |-- alert_manager.py         # Alert creation, prioritization, logging
|   |-- evaluation/
|   |   |-- metrics.py               # Classification metrics computation
|   |   |-- evaluator.py             # Full evaluation framework
|   |   |-- visualizations.py        # Plots: confusion matrix, ROC, epsilon curves
|   |-- dashboard/
|   |   |-- app.py                   # Streamlit dashboard main page
|   |   |-- components.py            # Dashboard UI components
|   |-- utils/
|       |-- config.py                # YAML config loader, path resolver
|       |-- logger.py                # Logging setup
|-- models/                          # Saved model weights (auto-generated)
|   |-- tier2/                       # DNN/CNN/LSTM .h5 files
|   |-- tier3/                       # Robust PyTorch model .pth files
|   |-- gan/                         # Generator/discriminator checkpoints
|   |-- preprocessing/               # Scaler, encoder, feature selector .pkl files
|-- tests/                           # Unit and integration tests
|-- notebooks/                       # Jupyter notebooks for experimentation
|-- logs/                            # Runtime and alert logs
|-- requirements.txt                 # Python dependencies
```

## Usage

### Train All Models

```bash
python main.py --mode train
```

Trains Tier 2 models (DNN, CNN, LSTM), the WGAN-GP generator, and the Tier 3 adversarially robust model. Falls back to synthetic data if no dataset is found.

### Run Evaluation

```bash
python main.py --mode evaluate
```

Evaluates the trained model on test data and saves results to `evaluation_results/`.

### Run Detection on a CSV File

```bash
python main.py --mode detect --input path/to/traffic.csv
```

Requires a trained model. Runs the full three-tier pipeline on the provided CSV file.

### Launch the Dashboard

```bash
python main.py --mode dashboard
```

Opens the Streamlit dashboard at `http://localhost:8501` with four modes:
- **Real-time Monitor:** Live traffic flow visualization and alert feed
- **Batch Analysis:** Upload CSV files for bulk detection
- **Attack Simulation:** Generate adversarial samples and test detection
- **Model Performance:** Compare clean vs. robust accuracy across system configurations

### Run Tests

```bash
pytest tests/ -v
```

## Dataset Setup

### NSL-KDD (Primary, Recommended)

1. Download from [UNB CIC Datasets](https://www.unb.ca/cic/datasets/nsl.html) or search "NSL-KDD" on Kaggle.
2. Place `KDDTrain+.txt` and `KDDTest+.txt` in `data/raw/NSL-KDD/`.

### CICIDS2017 (Secondary, Optional)

1. Download from [UNB CIC IDS 2017](https://www.unb.ca/cic/datasets/ids-2017.html).
2. Place the CSV files in `data/raw/CICIDS2017/`.

## Configuration

All settings are in `config/config.yaml`. Key options:

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `dataset.primary` | Dataset name | `NSL-KDD` | Primary dataset to use |
| `preprocessing.top_features` | Feature count | `35` | Number of features to select |
| `preprocessing.imbalance_method` | Balancing | `smote` | Class imbalance strategy |
| `tier2.training.epochs` | Epochs | `30` | Training epochs for Tier 2 models |
| `tier2.training.batch_size` | Batch size | `128` | Training batch size |
| `gan.epochs` | GAN epochs | `100` | WGAN-GP training epochs |
| `gan.latent_dim` | Latent size | `100` | GAN latent vector dimension |
| `adversarial_attacks.fgsm.epsilon` | Epsilon | `0.1` | FGSM perturbation magnitude |
| `adversarial_attacks.pgd.num_iterations` | PGD steps | `40` | PGD attack iterations |
| `adversarial_training.epochs` | Robust epochs | `20` | Adversarial training epochs |

## Requirements

- TensorFlow 2.13+ (Tier 2 models)
- PyTorch (adversarial attacks, GAN, Tier 3 defenses)
- scikit-learn (preprocessing, metrics)
- pandas, numpy (data handling)
- imbalanced-learn (SMOTE)
- adversarial-robustness-toolbox (C&W, DeepFool attacks)
- Streamlit, Plotly (dashboard)
- matplotlib, seaborn (static visualizations)

See `requirements.txt` for exact version constraints.

## Documentation

- **[User Manual](docs/user_manual.md)** -- Complete step-by-step guide from installation to usage
- **[Architecture](docs/architecture.md)** -- System architecture, data flow, and component design

## License

This project is licensed under the MIT License.
>>>>>>> 6a43f67 (Initial commit: Adversarially Robust IDS - Complete Implementation)
