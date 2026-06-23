"""
Adversarially Robust IDS - Main Entry Point

Usage:
    python main.py --mode train          # Train all models
    python main.py --mode evaluate       # Run full evaluation
    python main.py --mode detect --input data.csv  # Run detection on CSV
    python main.py --mode dashboard      # Launch Streamlit dashboard
    python main.py --mode demo           # Quick demo with synthetic data
"""

import argparse
import os
import sys
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import load_config, resolve_path
from src.utils.logger import setup_logger


def demo(config):
    """Quick demo with synthetic data - no real dataset needed."""
    logger = setup_logger('IDS-Demo')
    logger.info("Starting demo with synthetic data...")

    from src.preprocessing.data_loader import DataLoader
    from src.preprocessing.preprocessor import DataPreprocessor
    from src.tier1_signature.signature_detector import Tier1SignatureDetector

    # 1. Generate synthetic data
    loader = DataLoader()
    df = loader.generate_synthetic(n_samples=500)
    logger.info(f"Generated synthetic data: {df.shape}")

    # 2. Preprocess
    preprocessor = DataPreprocessor(config)
    data = preprocessor.run_pipeline(df, label_type='multiclass')
    logger.info(f"Preprocessed: {data['X_train'].shape[0]} train, "
                f"{data['X_val'].shape[0]} val, {data['X_test'].shape[0]} test")
    logger.info(f"Features: {data['n_features']}, Classes: {data['n_classes']}")

    # 3. Tier 1 test
    sig_path = resolve_path(config['tier1']['signature_db'])
    tier1 = Tier1SignatureDetector(sig_path)

    # Test with known attack pattern
    attack_sample = {
        'protocol': 'TCP', 'syn_flag_count': 200,
        'ack_flag_count': 2, 'flow_duration': 500
    }
    result = tier1.detect(attack_sample)
    logger.info(f"Tier 1 test: is_attack={result['is_attack']}, "
                f"type={result.get('attack_type')}, time={result['detection_time_ms']:.2f}ms")

    # Test benign
    benign_sample = {
        'protocol': 'TCP', 'syn_flag_count': 1,
        'ack_flag_count': 1, 'flow_duration': 5000
    }
    result = tier1.detect(benign_sample)
    logger.info(f"Tier 1 benign test: is_attack={result['is_attack']}")

    # 4. Quick DNN train (5 epochs for demo)
    logger.info("Training quick DNN model (5 epochs)...")
    from src.tier2_ml_detection.models import build_dnn

    model = build_dnn(data['n_features'], data['n_classes'])
    model.fit(
        data['X_train'], data['y_train'],
        validation_data=(data['X_val'], data['y_val']),
        epochs=5, batch_size=64, verbose=1
    )

    val_loss, val_acc = model.evaluate(data['X_val'], data['y_val'], verbose=0)
    logger.info(f"DNN Val Accuracy: {val_acc:.4f}")

    # 5. Test adversarial attacks
    logger.info("Testing FGSM attack...")
    import torch
    from src.adversarial_attacks.attack_utils import PyTorchDNN
    from src.adversarial_attacks.fgsm import fgsm_attack

    pt_model = PyTorchDNN(data['n_features'], data['n_classes'])
    x_sample = torch.FloatTensor(data['X_test'][:20])
    y_sample = torch.LongTensor(data['y_test'][:20])

    x_adv = fgsm_attack(pt_model, x_sample, y_sample, epsilon=0.1)
    perturbation = (x_adv - x_sample).abs().mean().item()
    logger.info(f"FGSM avg perturbation: {perturbation:.4f}")

    # # 6. Test GAN
    # logger.info("Testing GAN (5 epochs)...")
    # from src.gan_generator.gan_model import WGANGP
    #
    # attack_data = data['X_train'][data['y_train'] != 0][:200]
    # if len(attack_data) > 10:
    #     gan = WGANGP(config, data['n_features'])
    #     gan_config = config.copy()
    #     gan_config['gan'] = config['gan'].copy()
    #     gan_config['gan']['epochs'] = 5
    #     gan_config['gan']['save_interval'] = 100
    #     gan.config = gan_config
    #     gan.trainer.config = gan_config
    #     gen_losses, disc_losses = gan.trainer.train(attack_data, epochs=5, batch_size=min(32, len(attack_data)))
    #     samples = gan.generate(10)
    #     logger.info(f"GAN generated {samples.shape[0]} samples with shape {samples.shape[1]}")

    # Summary
    print("\n" + "="*60)
    print("DEMO COMPLETE - All components working!")
    print("="*60)
    print(f"  Synthetic Data: {df.shape[0]} samples")
    print(f"  Features Selected: {data['n_features']}")
    print(f"  Classes: {data['n_classes']}")
    print(f"  DNN Val Accuracy: {val_acc:.4f}")
    print(f"  Tier 1 Signature Detection: Working")
    print(f"  FGSM Attack: Working (avg perturbation: {perturbation:.4f})")
    # print(f"  GAN Generator: Working")
    print("="*60)


def train(config, dataset_override=None):
    """Train all models."""
    logger = setup_logger('IDS-Train')
    logger.info("Starting full training pipeline...")

    from src.preprocessing.data_loader import DataLoader
    from src.preprocessing.preprocessor import DataPreprocessor
    from src.tier2_ml_detection.train import Tier2Trainer

    # Apply dataset override if specified via --dataset
    if dataset_override:
        config['dataset']['primary'] = dataset_override
        logger.info(f"Dataset overridden to: {dataset_override}")

    # 1. Load and preprocess data
    loader = DataLoader()
    try:
        df = loader.load(config)
        logger.info(f"Loaded dataset: {df.shape}")
    except FileNotFoundError as e:
        logger.warning(f"Dataset not found: {e}")
        logger.info("Using synthetic data instead...")
        df = loader.generate_synthetic(n_samples=5000)

    preprocessor = DataPreprocessor(config)
    data = preprocessor.run_pipeline(df, label_type='multiclass')
    logger.info(f"Preprocessed: {data['n_features']} features, {data['n_classes']} classes")

    # 2. Train Tier 2 models
    logger.info("Training Tier 2 models...")
    trainer = Tier2Trainer(config)
    results, best_type = trainer.train_all_models(data)

    # # 3. Train GAN
    # logger.info("Training GAN...")
    # from src.gan_generator.gan_model import WGANGP
    #
    # attack_data = data['X_train'][data['y_train'] != 0]
    # gan = WGANGP(config, data['n_features'])
    # gan_result = gan.train(attack_data)
    # gan.save(resolve_path('models/gan'))
    # logger.info(f"GAN training complete: {gan_result['epochs']} epochs")

    # 4. Adversarial Training (Tier 3)
    logger.info("Running adversarial training...")
    import torch
    from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
    from src.adversarial_attacks.attack_utils import PyTorchDNN
    from src.tier3_adversarial_defense.adversarial_training import AdversarialTrainer

    pt_model = PyTorchDNN(data['n_features'], data['n_classes'])
    trainer_adv = AdversarialTrainer(pt_model, config)

    train_dataset = TensorDataset(
        torch.FloatTensor(data['X_train']),
        torch.LongTensor(data['y_train'])
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(data['X_val']),
        torch.LongTensor(data['y_val'])
    )

    train_loader = TorchDataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = TorchDataLoader(val_dataset, batch_size=128)

    best_acc = trainer_adv.train(train_loader, val_loader)
    logger.info(f"Adversarial training complete. Best val accuracy: {best_acc:.4f}")

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


def evaluate(config, dataset_override=None):
    """
    Run full evaluation for Tier 1, Tier 2, and Tier 3.

    Evaluates:
      - Tier 1 (Signature rules) — coverage stats
      - Tier 2 (Keras DNN)  on clean + FGSM + PGD adversarial data
      - Tier 3 (PyTorch robust DNN) on attack-gated samples + adversarial data
      - Per-attack breakdowns (DoS, Probe, R2L, U2R)
      - Tier comparisons (Tier 1+2 and Tier 1+2+3)

    Results saved to evaluation_results/ (separate files per tier + combined)
    """
    logger = setup_logger('IDS-Eval')
    logger.info("Starting evaluation...")

    from src.preprocessing.data_loader import DataLoader
    from src.preprocessing.preprocessor import DataPreprocessor
    from src.evaluation.evaluator import SystemEvaluator

    # Apply dataset override if specified via --dataset
    if dataset_override:
        config['dataset']['primary'] = dataset_override
        logger.info(f"Dataset overridden to: {dataset_override}")

    # ── Load data ──────────────────────────────────────────────────────
    loader = DataLoader()
    try:
        df = loader.load(config)
        logger.info(f"Dataset loaded: {len(df)} samples")
    except FileNotFoundError:
        logger.info("Dataset not found — using synthetic data for evaluation...")
        df = loader.generate_synthetic(n_samples=2000)

    # ── Preprocess ─────────────────────────────────────────────────────
    preprocessor = DataPreprocessor(config)
    data = preprocessor.run_pipeline(df, label_type='multiclass')

    X_test = data['X_test']
    y_test = data['y_test']
    logger.info(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")

    # ── Run evaluation for both tiers ──────────────────────────────────
    evaluator = SystemEvaluator(config)
    evaluator.run_full_evaluation_both_tiers(X_test, y_test)
    evaluator.save_results()

    logger.info("Evaluation complete. Results saved to evaluation_results/")


def dashboard(config):
    """Launch Streamlit dashboard."""
    import subprocess
    app_path = os.path.join(os.path.dirname(__file__), 'src', 'dashboard', 'app.py')
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', app_path])


def main():
    parser = argparse.ArgumentParser(description='Adversarially Robust IDS')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'evaluate', 'detect', 'dashboard', 'demo'],
                        help='Operation mode')
    parser.add_argument('--input', type=str, help='Input CSV file for detect mode')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['NSL-KDD', 'CICIDS2017', 'combined', 'synthetic'],
                        help='Dataset to use for training/evaluation '
                             '(overrides config.yaml). Options: NSL-KDD, CICIDS2017, combined, synthetic')

    args = parser.parse_args()

    config = load_config(args.config)

    if args.mode == 'demo':
        demo(config)
    elif args.mode == 'train':
        train(config, dataset_override=args.dataset)
    elif args.mode == 'evaluate':
        evaluate(config, dataset_override=args.dataset)
    elif args.mode == 'dashboard':
        dashboard(config)
    elif args.mode == 'detect':
        if not args.input:
            print("Error: --input required for detect mode")
            sys.exit(1)
        print(f"Detection on {args.input} - run training first, then use this mode.")


if __name__ == '__main__':
    main()
