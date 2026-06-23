# Architecture Documentation

This document describes the system architecture, component design, data flow, and implementation details of the Adversarially Robust Intrusion Detection System.
---
## Table of Contents

1. [High-Level System Diagram](#1-high-level-system-diagram)
2. [Component Interaction Diagram](#2-component-interaction-diagram)
3. [Data Flow](#3-data-flow)
4. [Tier 1: Signature Detection](#4-tier-1-signature-detection)
5. [Tier 2: ML Detection](#5-tier-2-ml-detection)
6. [Tier 3: Adversarial Defense](#6-tier-3-adversarial-defense)
7. [GAN Training Pipeline](#7-gan-training-pipeline)
8. [Adversarial Attack and Defense Flow](#8-adversarial-attack-and-defense-flow)
9. [Preprocessing Pipeline](#9-preprocessing-pipeline)
10. [Integration Layer](#10-integration-layer)
11. [Evaluation Framework](#11-evaluation-framework)
12. [Dashboard Architecture](#12-dashboard-architecture)

---

## 1. High-Level System Diagram

```
+============================================================================+
|                    ADVERSARIALLY ROBUST IDS - SYSTEM OVERVIEW               |
+============================================================================+
|                                                                            |
|   +--------------------+     +-----------------------------------------+   |
|   |   DATA SOURCES     |     |          DETECTION PIPELINE             |   |
|   |                    |     |                                         |   |
|   |  NSL-KDD Dataset   |     |   +----------+  +--------+  +--------+ |   |
|   |  CICIDS2017 Dataset+---->+   TIER 1   |  | TIER 2 |  | TIER 3 | |   |
|   |  Synthetic Data    |     |   | Signature|->| ML     |->| Advers.| |   |
|   |  Live Traffic      |     |   | Matching |  | Models |  | Defense| |   |
|   +--------------------+     |   +----------+  +--------+  +--------+ |   |
|                              +-----------+-----------------------------+   |
|   +--------------------+                 |                                 |
|   |   PREPROCESSING    |                 v                                 |
|   |                    |     +-----------+-----------+                     |
|   |  Data Cleaning     |     |   ALERT MANAGEMENT    |                     |
|   |  Encoding          |     |                       |                     |
|   |  Scaling           |     |  Priority Scoring     |                     |
|   |  Feature Selection |     |  Alert Logging        |                     |
|   |  SMOTE Balancing   |     |  Statistics Tracking  |                     |
|   +--------------------+     +-----------------------+                     |
|                                                                            |
|   +--------------------+     +--------------------+                        |
|   |   GAN MODULE       |     |   EVALUATION       |                        |
|   |                    |     |                    |                        |
|   |  WGAN-GP Generator |     |  Metrics (Acc,     |                        |
|   |  Discriminator     |     |   P, R, F1, FPR)   |                        |
|   |  Gradient Penalty  |     |  Robustness Tests  |                        |
|   |  Synthetic Traffic |     |  Visualizations    |                        |
|   +--------------------+     +--------------------+                        |
|                                                                            |
|   +--------------------------------------------------------------------+  |
|   |                      STREAMLIT DASHBOARD                            |  |
|   |   Real-time Monitor | Batch Analysis | Attack Sim | Model Compare   |  |
|   +--------------------------------------------------------------------+  |
|                                                                            |
+============================================================================+
```

---

## 2. Component Interaction Diagram

This diagram shows how the source modules call each other at runtime.

```
main.py
  |
  +---> src/utils/config.py          (load_config, resolve_path)
  +---> src/utils/logger.py          (setup_logger)
  |
  +--[demo/train]---> src/preprocessing/data_loader.py    (DataLoader)
  |                   src/preprocessing/preprocessor.py    (DataPreprocessor)
  |
  +--[demo/train]---> src/tier1_signature/signature_detector.py (Tier1SignatureDetector)
  |                       |
  |                       +---> data/signatures/signatures.json
  |
  +--[demo/train]---> src/tier2_ml_detection/models.py       (build_dnn, build_cnn, build_lstm)
  |                   src/tier2_ml_detection/train.py         (Tier2Trainer)
  |                       |
  |                       +---> src/tier2_ml_detection/feature_extractor.py (FeatureExtractor)
  |
  +--[demo/train]---> src/adversarial_attacks/fgsm.py        (fgsm_attack)
  |                   src/adversarial_attacks/attack_utils.py (PyTorchDNN)
  |
  +--[train]-------> src/gan_generator/gan_model.py          (WGANGP)
  |                      |
  |                      +---> src/gan_generator/generator.py      (Generator)
  |                      +---> src/gan_generator/discriminator.py   (Discriminator)
  |                      +---> src/gan_generator/train_gan.py       (WGANGPTrainer)
  |
  +--[train]-------> src/tier3_adversarial_defense/adversarial_training.py (AdversarialTrainer)
  |                      |
  |                      +---> src/adversarial_attacks/fgsm.py
  |                      +---> src/adversarial_attacks/pgd.py
  |
  +--[evaluate]----> src/evaluation/evaluator.py             (SystemEvaluator)
  |                      |
  |                      +---> src/evaluation/metrics.py     (compute_all_metrics)
  |
  +--[dashboard]---> src/dashboard/app.py                    (Streamlit app)
                         |
                         +---> src/dashboard/components.py   (UI components)


Integration Pipeline (used by detect mode):

src/integration/ids_pipeline.py (AdversarialRobustIDS)
  |
  +---> src/tier1_signature/signature_detector.py
  +---> src/tier2_ml_detection/ml_detector.py
  |         |
  |         +---> src/tier2_ml_detection/feature_extractor.py
  +---> src/tier3_adversarial_defense/adversarial_defense.py
  |         |
  |         +---> src/tier3_adversarial_defense/input_transformation.py
  |         +---> src/tier3_adversarial_defense/ensemble_defense.py
  +---> src/integration/alert_manager.py
```

---

## 3. Data Flow

### Training Data Flow

```
+----------------+     +-------------------+     +------------------+
| Raw Dataset    |     | Data Cleaning     |     | Label Encoding   |
| (NSL-KDD CSV)  +--->+ Remove duplicates  +--->+ Binary: 0/1      |
| 43 columns     |     | Fill NaN (median) |     | Multi: 0,1,2,3,4 |
+----------------+     | Replace infinity  |     +--------+---------+
                       +-------------------+              |
                                                          v
+------------------+     +------------------+     +-------+----------+
| SMOTE            |     | Feature Selection|     | One-Hot Encoding |
| Oversample       +<---+ Top 35 features  +<---+ protocol_type    |
| minority classes |     | Mutual info      |     | service, flag    |
+--------+---------+     +------------------+     +------------------+
         |                                                 ^
         v                                                 |
+--------+---------+     +------------------+     +--------+---------+
| Training Data    |     | Validation Data  |     | Standard Scaling |
| X_train, y_train |     | X_val, y_val     |     | Zero mean,       |
| (70%, balanced)  |     | (15%)            |     | unit variance    |
+------------------+     +------------------+     +------------------+
                                                           ^
                                                           |
                                                  +--------+---------+
                                                  | Data Split       |
                                                  | 70% / 15% / 15% |
                                                  | Stratified       |
                                                  +------------------+
```

### Detection Data Flow

```
+--------------------+
| Incoming Traffic   |
| (dict or array)    |
+---------+----------+
          |
          v
+---------+----------+     +-------------------+
| Tier 1: Signature  |     |                   |
| Pattern Matching   +---->+ MATCH FOUND       |
| threshold >= 0.8   |     | -> ALERT (Tier 1) |
+---------+----------+     | severity, type,   |
          |                 | confidence=1.0    |
          | NO MATCH        +-------------------+
          v
+---------+----------+     +-------------------+
| Tier 2: ML Model   |     |                   |
| DNN/CNN/LSTM       +---->+ ATTACK DETECTED   |
| softmax -> argmax  |     | -> ALERT (Tier 2) |
+---------+----------+     +-------------------+
          |
          | BENIGN (class=0 or confidence < 0.5)
          | (might be adversarial evasion)
          v
+---------+----------+     +-------------------+
| Tier 3: Adversarial|     |                   |
| Robust Model       +---->+ EVASION CAUGHT    |
| Feature Squeezing  |     | -> ALERT (Tier 3) |
| Ensemble Voting    |     | CRITICAL severity |
+---------+----------+     | + adversarial flag|
          |                 +-------------------+
          | ALL TIERS AGREE
          v
+--------------------+
| BENIGN             |
| (confirmed safe)   |
+--------------------+
```

---

## 4. Tier 1: Signature Detection

### Purpose

Provide fast, deterministic detection of known attack patterns with zero false negatives for well-defined signatures.

### Architecture

```
+---------------------------+
|   Tier1SignatureDetector   |
+---------------------------+
| - signatures: Dict        |
| - detection_count: int    |
+---------------------------+
| + detect(sample) -> Dict  |
| + detect_batch(batch)     |
+---------------------------+
        |
        | loads
        v
+---------------------------+
|   signatures.json         |
+---------------------------+
| dos_attacks:              |
|   - SYN Flood (DOS001)   |
|   - UDP Flood (DOS002)   |
|   - Slowloris (DOS003)   |
| port_scan:                |
|   - TCP Port Scan         |
| brute_force:              |
|   - SSH Brute Force       |
|   - FTP Brute Force       |
| web_attacks:              |
|   - SQL Injection         |
|   - XSS Attempt           |
| botnet:                   |
|   - C2 Communication      |
| nsl_kdd_rules:            |
|   dos: Neptune, Smurf     |
|   probe: Port Sweep, Satan|
|   r2l: Guess Password     |
|   u2r: Buffer Overflow    |
+---------------------------+
```

### Matching Algorithm

```
For each signature category:
    For each signature in category:
        match_count = 0
        total_conditions = len(signature.conditions)

        For each (feature, rule) in conditions:
            value = sample[feature]

            If rule is dict (range):
                Check value >= rule.min AND value <= rule.max
            If rule is list:
                Check value in rule
            If rule is scalar:
                Check value == rule

            If condition met: match_count += 1

        match_ratio = match_count / total_conditions
        If match_ratio >= threshold (0.8):
            RETURN: attack detected
```

### Performance

- Time complexity: O(S * C) where S = total signatures, C = average conditions per signature
- Typical: < 0.5ms per sample
- Memory: Signature database is ~5 KB in memory

---

## 5. Tier 2: ML Detection

### Purpose

Classify unknown or novel attacks that do not match any known signature pattern, using learned representations of network traffic.

### Model Architectures

```
DNN Architecture:                   CNN Architecture:
+------------------------+         +------------------------+
| Input (n_features)     |         | Input (n_features, 1)  |
+------------------------+         +------------------------+
| Dense(256) + ReLU      |         | Conv1D(64, k=3) + ReLU |
| BatchNorm + Dropout(0.3)|        | BatchNorm               |
+------------------------+         | MaxPool1D(2)           |
| Dense(128) + ReLU      |         +------------------------+
| BatchNorm + Dropout(0.3)|        | Conv1D(128, k=3) + ReLU|
+------------------------+         | BatchNorm               |
| Dense(64) + ReLU       |         | MaxPool1D(2)           |
| BatchNorm + Dropout(0.3)|        +------------------------+
+------------------------+         | Conv1D(256, k=3) + ReLU|
| Dense(32) + ReLU       |         | BatchNorm               |
+------------------------+         +------------------------+
| Dense(n_classes)       |         | Flatten                |
| Softmax                |         | Dense(128) + Dropout   |
+------------------------+         | Dense(64) + ReLU       |
                                   | Dense(n_classes)       |
                                   | Softmax                |
                                   +------------------------+

LSTM Architecture:
+------------------------+
| Input (1, n_features)  |
+------------------------+
| LSTM(128, return_seq)  |
| Dropout(0.3)           |
+------------------------+
| LSTM(64)               |
| Dropout(0.3)           |
+------------------------+
| Dense(32) + ReLU       |
+------------------------+
| Dense(n_classes)        |
| Softmax                |
+------------------------+
```

### Training Pipeline

```
+------------------+     +-------------------+     +------------------+
| Train DNN        |     | Train CNN         |     | Train LSTM       |
| (up to 30 epochs)|     | (up to 30 epochs) |     | (up to 30 epochs)|
+--------+---------+     +---------+---------+     +--------+---------+
         |                         |                         |
         v                         v                         v
+--------+---------+     +---------+---------+     +--------+---------+
| Val Acc: X.XX    |     | Val Acc: Y.YY     |     | Val Acc: Z.ZZ    |
+--------+---------+     +---------+---------+     +--------+---------+
         |                         |                         |
         +-------------------------+-------------------------+
                                   |
                                   v
                          +--------+---------+
                          | SELECT BEST      |
                          | (highest val acc)|
                          +--------+---------+
                                   |
                                   v
                          +--------+---------+
                          | Save as          |
                          | best_model.h5    |
                          +------------------+
```

### Classification Mapping

```
Class 0 --> Normal  (benign traffic)
Class 1 --> DoS     (denial of service: neptune, smurf, back, land, etc.)
Class 2 --> Probe   (reconnaissance: ipsweep, nmap, portsweep, satan)
Class 3 --> R2L     (remote to local: ftp_write, guess_passwd, imap, phf)
Class 4 --> U2R     (user to root: buffer_overflow, loadmodule, rootkit)
```

---

## 6. Tier 3: Adversarial Defense

### Purpose

Detect adversarial evasion attempts and provide robust classification even when the input has been deliberately perturbed.

### Three Defense Strategies

```
+================================================================+
|                    TIER 3 DEFENSE PIPELINE                      |
+================================================================+
|                                                                  |
|   Strategy 1: ADVERSARIAL TRAINING                              |
|   +----------------------------------------------------------+  |
|   | During training:                                          |  |
|   |   For each batch:                                        |  |
|   |     1. Take clean samples (40%)                          |  |
|   |     2. Generate FGSM adversarial (30%)                   |  |
|   |     3. Generate PGD adversarial (30%)                    |  |
|   |     4. Mix all together                                  |  |
|   |     5. Train on mixed batch                              |  |
|   |   Result: Model learns to classify correctly even        |  |
|   |           when input is perturbed                        |  |
|   +----------------------------------------------------------+  |
|                                                                  |
|   Strategy 2: INPUT TRANSFORMATION (Feature Squeezing)          |
|   +----------------------------------------------------------+  |
|   | At detection time:                                       |  |
|   |   1. Get prediction on original input     --> pred_A     |  |
|   |   2. Reduce bit depth (16 levels)                        |  |
|   |   3. Get prediction on squeezed input     --> pred_B     |  |
|   |   4. Compute L1 distance(pred_A, pred_B)                 |  |
|   |   5. If distance > threshold (0.1):                      |  |
|   |        FLAG AS ADVERSARIAL                               |  |
|   +----------------------------------------------------------+  |
|                                                                 |
|   Strategy 3: ENSEMBLE VOTING                                   |
|   +----------------------------------------------------------+  |
|   | At detection time:                                       |  |
|   |   1. Model_1 predicts probabilities                      |  |
|   |   2. Model_2 predicts probabilities                      |  |
|   |   3. Model_N predicts probabilities                      |  |
|   |   4. Average all probability distributions               |  |
|   |   5. Final class = argmax(averaged probs)                |  |
|   |   6. Agreement = fraction of models agreeing             |  |
|   |   Low agreement --> possible adversarial input           |  |
|   +----------------------------------------------------------+  |
|                                                                 |
+================================================================+
```

### Adversarial Detection Decision Flow

```
Input: feature vector x
                |
                v
+---------------+------------------+
| Predict on original: pred_orig   |
+---------------+------------------+
                |
                v
+---------------+------------------+
| Squeeze input: x_sq = round(x)  |
+---------------+------------------+
                |
                v
+---------------+------------------+
| Predict on squeezed: pred_sq     |
+---------------+------------------+
                |
                v
+---------------+------------------+
| distance = |pred_orig - pred_sq| |
+---------------+------------------+
                |
        +-------+-------+
        |               |
  distance > 0.1   distance <= 0.1
        |               |
        v               v
  ADVERSARIAL       NOT ADVERSARIAL
  severity=CRITICAL  continue to
                     ensemble voting
```

### Severity Assignment Logic

```
If is_adversarial == True:
    severity = CRITICAL

Else if confidence > 0.9 AND agreement > 0.8:
    severity = HIGH

Else if confidence > 0.7:
    severity = MEDIUM

Else:
    severity = LOW
```

---

## 7. GAN Training Pipeline

### Purpose

Generate realistic synthetic attack traffic samples for:
- Augmenting training data (especially rare attack classes)
- Stress-testing the detection system
- Evaluating robustness against novel attack patterns

### WGAN-GP Architecture

```
+================================+     +================================+
|         GENERATOR              |     |        DISCRIMINATOR           |
+================================+     +================================+
|                                |     |                                |
| Input: z (100-dim noise)       |     | Input: x (feature_dim)         |
|            |                   |     |            |                   |
|            v                   |     |            v                   |
| Linear(100 -> 128) + LeakyReLU|     | Linear(feat -> 512) + LeakyReLU|
|            |                   |     | Dropout(0.3)                   |
|            v                   |     |            |                   |
| Linear(128 -> 256) + BN       |     |            v                   |
| + LeakyReLU                   |     | Linear(512 -> 256) + LeakyReLU|
|            |                   |     | Dropout(0.3)                   |
|            v                   |     |            |                   |
| Linear(256 -> 512) + BN       |     |            v                   |
| + LeakyReLU                   |     | Linear(256 -> 1)               |
|            |                   |     | (no sigmoid -- raw score)      |
|            v                   |     |                                |
| Linear(512 -> feature_dim)    |     | Output: realness score         |
| + Tanh                        |     |                                |
|                                |     +================================+
| Output: fake feature vector    |
|                                |
+================================+
```

### Training Loop

```
For each epoch (1 to 100):
    For each batch of real attack data:

        +----- TRAIN DISCRIMINATOR (5 times per generator update) -----+
        |                                                               |
        |  1. Sample real batch from dataset                           |
        |  2. Generate fake batch: z -> Generator -> fake              |
        |  3. Compute scores:                                          |
        |       disc_real = D(real).mean()                             |
        |       disc_fake = D(fake).mean()                             |
        |  4. Compute gradient penalty:                                |
        |       interpolate = alpha * real + (1-alpha) * fake          |
        |       grad = gradient(D(interpolate), interpolate)           |
        |       penalty = (||grad||_2 - 1)^2                          |
        |  5. D_loss = disc_fake - disc_real + 10 * penalty            |
        |  6. Update D weights                                         |
        +--------------------------------------------------------------+

        +----- TRAIN GENERATOR (1 time) -----+
        |                                      |
        |  1. Generate fake: z -> G -> fake   |
        |  2. G_loss = -D(fake).mean()         |
        |  3. Update G weights                 |
        +--------------------------------------+

    If epoch % 25 == 0:
        Save generator checkpoint

Save final generator and discriminator weights
```

### Why WGAN-GP Over Vanilla GAN

```
Vanilla GAN:                         WGAN-GP:
- JS divergence loss                 - Wasserstein distance loss
- Unstable training                  - Stable training
- Mode collapse common               - No mode collapse
- Requires careful balancing          - Gradient penalty auto-regulates
- Binary cross-entropy loss           - Linear critic output (no sigmoid)
```

---

## 8. Adversarial Attack and Defense Flow

### Attack Generation Pipeline

```
                    +-------------------+
                    |  Clean Input x    |
                    +--------+----------+
                             |
              +--------------+--------------+
              |              |              |
              v              v              v
        +-----+-----+  +----+------+  +----+------+
        |   FGSM    |  |    PGD    |  |   C&W     |
        |           |  |           |  |           |
        | x + eps * |  | Iterative |  | Optimize  |
        | sign(grad)|  | FGSM with |  | min ||d|| |
        |           |  | projection|  | s.t.      |
        | 1 step    |  | 40 steps  |  | misclass. |
        +-----+-----+  +----+------+  +----+------+
              |              |              |
              v              v              v
        +-----+-----+  +----+------+  +----+------+
        | x_adv_fgsm|  | x_adv_pgd|  | x_adv_cw  |
        +-----+-----+  +----+------+  +----+------+
              |              |              |
              +--------------+--------------+
                             |
                             v
                    +--------+----------+
                    |  Mixed Adversarial |
                    |  Dataset           |
                    | 30% FGSM          |
                    | 30% PGD           |
                    | 20% C&W           |
                    | 20% DeepFool      |
                    +-------------------+
```

### Attack Characteristics

```
+----------+--------+---------+----------+----------------------------------+
| Attack   | Steps  | Speed   | Strength | Mechanism                        |
+----------+--------+---------+----------+----------------------------------+
| FGSM     | 1      | Fast    | Moderate | Sign of loss gradient * epsilon  |
| PGD      | 40     | Slow    | Strong   | Iterative FGSM with projection   |
| C&W      | 100    | Slowest | Strongest| Optimization to find min L2 pert |
| DeepFool | 50     | Medium  | Strong   | Find closest decision boundary   |
+----------+--------+---------+----------+----------------------------------+
```

### Defense Evaluation Flow

```
+------------------+     +-------------------+     +------------------+
| Clean Test Data  |     | Generate Attacks  |     | Evaluate Model   |
| X_test, y_test   +---->+ FGSM, PGD, etc.  +---->+ on adversarial   |
+------------------+     | X_adv             |     | data             |
                         +-------------------+     +--------+---------+
                                                            |
                                                            v
                                                   +--------+---------+
                                                   | Compute Metrics  |
                                                   |                  |
                                                   | clean_accuracy   |
                                                   | robust_accuracy  |
                                                   | accuracy_drop    |
                                                   | robustness_ratio |
                                                   +------------------+
```

---

## 9. Preprocessing Pipeline

### Full Pipeline Flow

```
Step 1: LOAD DATA
+--------------------+
| DataLoader.load()  |  Reads CSV, assigns column names,
| or generate_       |  maps attack labels to categories
| synthetic()        |
+--------+-----------+
         |
         v
Step 2: CLEAN DATA
+--------------------+
| clean_data()       |  Replace inf -> NaN
|                    |  Fill NaN with median
|                    |  Remove duplicates
+--------+-----------+
         |
         v
Step 3: ENCODE LABELS
+--------------------+
| encode_labels()    |  Binary: Normal=0, Attack=1
|                    |  Multi: Normal=0, DoS=1, Probe=2, R2L=3, U2R=4
+--------+-----------+
         |
         v
Step 4: ENCODE CATEGORICALS
+--------------------+
| encode_categorical |  One-hot encode: protocol_type, service, flag
| ()                 |  Expands ~3 columns to ~70+ binary columns
+--------+-----------+
         |
         v
Step 5: SPLIT DATA
+--------------------+
| split_data()       |  70% train / 15% validation / 15% test
|                    |  Stratified split (preserves class ratios)
+--------+-----------+
         |
         v
Step 6: SCALE FEATURES
+--------------------+
| scale_features()   |  StandardScaler: fit on train, transform all
|                    |  Result: zero mean, unit variance
+--------+-----------+
         |
         v
Step 7: SELECT FEATURES
+--------------------+
| select_features()  |  SelectKBest with mutual_info_classif
|                    |  Keep top 35 features (configurable)
+--------+-----------+
         |
         v
Step 8: BALANCE CLASSES
+--------------------+
| handle_imbalance() |  SMOTE on training set only
|                    |  Generates synthetic minority samples
+--------+-----------+
         |
         v
Step 9: SAVE ARTIFACTS
+--------------------+
| _save_artifacts()  |  Save scaler.pkl, label_encoder.pkl,
|                    |  feature_selector.pkl to models/preprocessing/
+--------------------+
```

### Data Shape Transformations

```
Raw NSL-KDD:    (125,973 rows x 43 columns)
                         |
After cleaning:          ~125,000 rows (duplicates removed)
                         |
After encoding:          ~125,000 rows x ~110 columns (one-hot expansion)
                         |
After split:             Train: ~87,500   Val: ~18,750   Test: ~18,750
                         |
After scaling:           Same shape, values standardized
                         |
After feature select:    Train: ~87,500 x 35   Val: ~18,750 x 35   Test: ~18,750 x 35
                         |
After SMOTE:             Train: ~100,000+ x 35 (minority classes oversampled)
```

---

## 10. Integration Layer

### Pipeline Orchestration

The `AdversarialRobustIDS` class in `src/integration/ids_pipeline.py` orchestrates the full detection flow.

```
+==========================================+
|        AdversarialRobustIDS              |
+==========================================+
| - config: Dict                           |
| - tier1: Tier1SignatureDetector | None   |
| - tier2: Tier2MLDetector | None          |
| - tier3: Tier3AdversarialDefense | None  |
| - alert_manager: AlertManager            |
| - stats: Dict                            |
+==========================================+
| + __init__(config)                       |
|     _init_tier1()                        |
|     _init_tier2()                        |
|     _init_tier3()                        |
+------------------------------------------+
| + detect(traffic_sample) -> Dict         |
| + detect_batch(samples) -> List[Dict]    |
| + get_statistics() -> Dict               |
+==========================================+
```

### Initialization Sequence

```
1. Load config from YAML
2. Create AlertManager (always)
3. Init Tier 1:
   - If enabled: load signatures.json -> Tier1SignatureDetector
   - If error: log warning, tier1 = None
4. Init Tier 2:
   - If enabled AND best_model.h5 exists: load -> Tier2MLDetector
   - If model missing: log warning, tier2 = None
5. Init Tier 3:
   - If enabled AND robust_model.pth exists:
     - Load state dict, infer dimensions
     - Create PyTorchDNN, load weights
     - Wrap in ModelWrapper
     - Create Tier3AdversarialDefense with ensemble
   - If model missing: log warning, tier3 = None
```

### Alert Manager

```
+==================================+
|        AlertManager              |
+==================================+
| - alerts: List[Dict]            |
| - logger: Logger                |
+==================================+
| + create_alert(tier, type,      |
|   severity, confidence, ...)    |
|   -> Dict with:                 |
|     alert_id: UUID              |
|     timestamp: ISO format       |
|     tier: 1, 2, or 3           |
|     attack_type: string         |
|     severity: CRITICAL/HIGH/... |
|     confidence: 0.0-1.0        |
|     is_adversarial: bool        |
|     priority: 1-4              |
+----------------------------------+
| + get_recent_alerts(n) -> List  |
| + get_alert_summary() -> Dict   |
+==================================+

Priority Assignment:
  1 (HIGHEST): Adversarial attack OR Tier 1 CRITICAL
  2          : HIGH severity
  3          : MEDIUM severity
  4 (LOWEST) : Everything else
```

---

## 11. Evaluation Framework

### Metrics Computed

```
+---------------------------+------------------------------------------+
| Metric                    | Description                              |
+---------------------------+------------------------------------------+
| Accuracy                  | Overall correct predictions / total      |
| Precision (weighted)      | TP / (TP + FP) averaged across classes   |
| Recall (weighted)         | TP / (TP + FN) averaged across classes   |
| F1-Score (weighted)       | 2 * P * R / (P + R) averaged             |
| FPR                       | FP / (FP + TN) -- false alarm rate       |
| FNR                       | FN / (FN + TP) -- missed attack rate     |
| AUC-ROC                   | Area under ROC curve (OVR for multiclass)|
| Confusion Matrix          | Full N x N prediction breakdown          |
| Robust Accuracy           | Accuracy on adversarial test data        |
| Accuracy Drop             | Clean accuracy - Robust accuracy         |
| Robustness Ratio          | Robust accuracy / Clean accuracy         |
+---------------------------+------------------------------------------+
```

### Visualization Outputs

```
+------------------------------------------+
| Confusion Matrix Heatmap                 |  confusion_matrix.png
| - Per-class prediction counts            |
| - Color-coded with annotations           |
+------------------------------------------+
| ROC Curves                               |  roc_curves.png
| - One curve per class (OVR)              |
| - AUC value in legend                    |
+------------------------------------------+
| Epsilon Sensitivity Plot                 |  epsilon_sensitivity.png
| - X-axis: perturbation magnitude         |
| - Y-axis: model accuracy                 |
| - One line per attack type               |
+------------------------------------------+
| Tier Breakdown Pie Chart                 |  tier_breakdown.png
| - Fraction of detections per tier        |
+------------------------------------------+
| Training History                         |  training_history.png
| - Loss and accuracy vs. epoch            |
| - Train and validation curves            |
+------------------------------------------+
| Baseline Comparison                      |  baseline_comparison.png
| - Bar chart comparing system configs     |
| - Accuracy, Precision, Recall, F1       |
+------------------------------------------+
| Feature Importance                       |  feature_importance.png
| - Horizontal bar chart of top-20 features|
+------------------------------------------+
```

---

## 12. Dashboard Architecture

### Technology Stack

```
Frontend:  Streamlit (Python-based reactive web framework)
Charts:    Plotly (interactive charts)
Backend:   Direct Python imports from src/ modules
Hosting:   Local (default: http://localhost:8501)
```

### Page Layout

```
+================================================================+
|  HEADER: Adversarially Robust Intrusion Detection System       |
+================================================================+
|  SIDEBAR         |  MAIN CONTENT AREA                          |
|  +------------+  |                                              |
|  | Control    |  |  +------ TOP METRIC ROW ----------------+  |
|  | Panel      |  |  | Traffic | Attacks | Advers. | FPR    |  |
|  |            |  |  | 12,450  | 187     | 23      | 3.2%   |  |
|  | Mode:      |  |  +--------------------------------------+  |
|  | [dropdown] |  |                                              |
|  |            |  |  Content changes based on selected mode:    |
|  | - Realtime |  |                                              |
|  | - Batch    |  |  [MODE-SPECIFIC CONTENT]                    |
|  | - Attack   |  |                                              |
|  | - Model    |  |                                              |
|  +------------+  |                                              |
+================================================================+
```

### Mode-Specific Content

```
Real-time Monitor:
+---------------------------+  +------------------+
| Traffic Flow Chart        |  | Alert Feed       |
| (Plotly line chart)       |  | (scrollable list)|
| Benign / Attack / Adv     |  | severity + type  |
+---------------------------+  +------------------+
+--------------------------------------------------+
| Tier Breakdown (3-column layout)                 |
| Tier 1: 45 det, 0.2ms  |  Tier 2: 119 det, 12ms|
| Tier 3: 23 det, 35ms                             |
+--------------------------------------------------+

Batch Analysis:
+--------------------------------------------------+
| File Upload Widget (CSV)                         |
| Preview Table (first 10 rows)                    |
| [Run Detection] button                           |
| Results JSON                                     |
+--------------------------------------------------+

Attack Simulation:
+--------------------------------------------------+
| Attack Type: [FGSM | PGD | C&W | DeepFool]      |
| Epsilon: [-----slider-----] 0.01 - 0.5          |
| Samples: [100]                                    |
| [Launch Simulation]                               |
| Results: JSON + Bar Chart (success vs detected)  |
+--------------------------------------------------+

Model Performance:
+--------------------------------------------------+
| Comparison Table:                                |
|   Signature Only | ML Only | Dual-Tier | Ours   |
|   85.2% acc      | 93.5%   | 94.8%     | 95.2%  |
|   0% robust      | 45%     | 48%       | 88%    |
| Grouped Bar Chart: Clean vs Robust Accuracy      |
+--------------------------------------------------+
```

---

## Summary

The system follows a defense-in-depth principle: each tier adds a distinct layer of protection, and the combination is significantly more robust than any single approach. The modular design allows individual components to be upgraded, replaced, or disabled independently through the configuration file.

Key architectural decisions:
- **TensorFlow for Tier 2 models:** Mature, well-supported for production Keras models with callbacks.
- **PyTorch for attacks and Tier 3:** Provides the fine-grained gradient control needed for adversarial operations.
- **WGAN-GP over vanilla GAN:** Stable training without mode collapse.
- **Feature squeezing for adversarial detection:** Simple, effective, and fast at inference time.
- **Ensemble voting for robust classification:** Leverages model diversity to overcome adversarial transferability.
- **YAML configuration:** Single source of truth for all parameters, easy to experiment with different settings.
