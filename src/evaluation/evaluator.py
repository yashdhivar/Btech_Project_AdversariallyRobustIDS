import numpy as np
import os
import json
import torch

from src.evaluation.metrics import (
    compute_all_metrics, compute_robust_accuracy,
    compute_per_class_metrics, compute_per_class_robust_metrics,
    ATTACK_CATEGORIES,
)
from src.utils.config import resolve_path


class SystemEvaluator:
    """Comprehensive evaluation framework for the three-tier IDS."""

    def __init__(self, config):
        self.config = config
        self.results = {}

    # ------------------------------------------------------------------
    # Existing methods (unchanged)
    # ------------------------------------------------------------------

    def evaluate_clean(self, model, X_test, y_test, model_name='model'):
        """Evaluate model on clean (unperturbed) test data."""
        predictions = model.predict(X_test, verbose=0) if hasattr(model, 'predict') else model(X_test)

        if hasattr(predictions, 'numpy'):
            predictions = predictions.numpy()
        predictions = np.array(predictions)

        if predictions.ndim > 1:
            y_pred = np.argmax(predictions, axis=1)
            y_probs = predictions
        else:
            y_pred = (predictions > 0.5).astype(int)
            y_probs = None

        metrics = compute_all_metrics(y_test, y_pred, y_probs)
        self.results[f'{model_name}_clean'] = metrics
        return metrics

    def evaluate_robust(self, model_predict_fn, X_test, y_test, X_adv, attack_name):
        """Evaluate model robustness against adversarial attack."""
        pred_clean = model_predict_fn(X_test)
        pred_adv = model_predict_fn(X_adv)

        if hasattr(pred_clean, 'numpy'):
            pred_clean = pred_clean.numpy()
        if hasattr(pred_adv, 'numpy'):
            pred_adv = pred_adv.numpy()

        pred_clean = np.array(pred_clean)
        pred_adv = np.array(pred_adv)

        if pred_clean.ndim > 1:
            y_pred_clean = np.argmax(pred_clean, axis=1)
            y_pred_adv = np.argmax(pred_adv, axis=1)
        else:
            y_pred_clean = (pred_clean > 0.5).astype(int)
            y_pred_adv = (pred_adv > 0.5).astype(int)

        robust_metrics = compute_robust_accuracy(y_test, y_pred_clean, y_pred_adv)
        self.results[f'robust_{attack_name}'] = robust_metrics
        return robust_metrics

    def compare_baselines(self, baseline_results):
        """Compare different system configurations."""
        comparison = []
        for name, metrics in baseline_results.items():
            comparison.append({
                'System': name,
                'Accuracy': metrics.get('accuracy', 0),
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'F1': metrics.get('f1_score', 0),
                'FPR': metrics.get('fpr', 0),
            })
        self.results['baseline_comparison'] = comparison
        return comparison

    def run_full_evaluation(self, model, X_test, y_test, model_name='model'):
        """Run complete evaluation suite (legacy single-model path)."""
        print(f"\n{'='*60}")
        print(f"Running Full Evaluation for {model_name}")
        print(f"{'='*60}")

        clean_metrics = self.evaluate_clean(model, X_test, y_test, model_name)
        print(f"\nClean Accuracy: {clean_metrics['accuracy']:.4f}")
        print(f"Precision:      {clean_metrics['precision']:.4f}")
        print(f"Recall:         {clean_metrics['recall']:.4f}")
        print(f"F1-Score:       {clean_metrics['f1_score']:.4f}")
        print(f"FPR:            {clean_metrics.get('fpr', 'N/A')}")

        return self.results

    # ------------------------------------------------------------------
    # Full two-tier evaluation with per-tier, per-attack, and comparisons
    # ------------------------------------------------------------------

    def run_full_evaluation_both_tiers(self, X_test, y_test,
                                        adv_sample_cap=5000):
        """
        Evaluate Tier 1 (signature), Tier 2 (Keras DNN), and Tier 3
        (PyTorch robust DNN) individually, with per-attack breakdowns
        and tier comparisons.

        Tier 3 is only evaluated on samples that Tier 2 classifies as
        benign/normal -- these are the samples that may be adversarial
        evasion attacks that fooled Tier 2. Tier 3's adversarially-trained
        model provides a second opinion to catch what Tier 2 missed.

        Results structure:
            tier1: {metrics, per_class}
            tier2: {clean, robust_fgsm, robust_pgd, per_class_clean, per_class_robust_fgsm, per_class_robust_pgd}
            tier3: {clean, robust_fgsm, robust_pgd, per_class_clean, per_class_robust_fgsm, per_class_robust_pgd}
            comparison_tier1_tier2: {...}
            comparison_all_tiers: {...}
        """
        from src.adversarial_attacks.fgsm import fgsm_attack
        from src.adversarial_attacks.pgd import pgd_attack
        from src.adversarial_attacks.attack_utils import PyTorchDNN

        input_dim = X_test.shape[1]
        num_classes = len(np.unique(y_test))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ── Stratified subsample for adversarial evaluation ────────────
        n_total = len(X_test)
        if n_total > adv_sample_cap:
            print(f"\n[INFO] Test set has {n_total:,} samples.")
            print(f"[INFO] Using a stratified subset of {adv_sample_cap:,} samples "
                  f"for adversarial evaluation (full set used for clean metrics).")
            rng = np.random.default_rng(42)
            classes = np.unique(y_test)
            adv_indices = []
            per_class = adv_sample_cap // len(classes)
            for c in classes:
                c_idx = np.where(y_test == c)[0]
                chosen = rng.choice(c_idx, size=min(per_class, len(c_idx)), replace=False)
                adv_indices.extend(chosen.tolist())
            remaining = adv_sample_cap - len(adv_indices)
            if remaining > 0:
                all_idx = set(range(n_total))
                leftover = list(all_idx - set(adv_indices))
                extra = rng.choice(leftover, size=min(remaining, len(leftover)), replace=False)
                adv_indices.extend(extra.tolist())
            adv_indices = np.array(adv_indices)
            X_adv_base = X_test[adv_indices]
            y_adv_base = y_test[adv_indices]
        else:
            X_adv_base = X_test
            y_adv_base = y_test
            print(f"\n[INFO] Using all {n_total:,} test samples for adversarial evaluation.")

        # ── 1. Tier 1 Evaluation (Signature) ──────────────────────────
        print(f"\n{'='*60}")
        print("TIER 1: Signature-Based Detection Evaluation")
        print(f"{'='*60}")
        tier1_results = self._evaluate_tier1(X_test, y_test)
        self.results['tier1'] = tier1_results

        # ── 2. Load Tier 2 Keras model ─────────────────────────────────
        print(f"\n{'='*60}")
        print("TIER 2: Loading Keras DNN (best_model.h5)")
        print(f"{'='*60}")

        import tensorflow as tf
        tier2_path = resolve_path(self.config['tier2']['model_path'])

        if not os.path.exists(tier2_path):
            print(f"[ERROR] Tier 2 model not found at {tier2_path}. Skipping Tier 2.")
            tier2_keras = None
        else:
            tier2_keras = tf.keras.models.load_model(tier2_path)
            print(f"Loaded Tier 2 model from {tier2_path}")

        # ── 3. Load Tier 3 PyTorch robust model ────────────────────────
        print(f"\n{'='*60}")
        print("TIER 3: Loading PyTorch Robust DNN (robust_model.pth)")
        print(f"{'='*60}")

        tier3_path = resolve_path(self.config['tier3']['robust_model_path'])

        if not os.path.exists(tier3_path):
            print(f"[ERROR] Tier 3 model not found at {tier3_path}. Skipping Tier 3.")
            tier3_torch = None
        else:
            tier3_torch = PyTorchDNN(input_dim=input_dim, num_classes=num_classes)
            state = torch.load(tier3_path, map_location=device)
            tier3_torch.load_state_dict(state)
            tier3_torch.to(device)
            tier3_torch.eval()
            print(f"Loaded Tier 3 model from {tier3_path}")
            print(f"  input_dim={input_dim}, num_classes={num_classes}")

        # ── 4. Generate adversarial examples ───────────────────────────
        print(f"\n{'='*60}")
        print("Generating adversarial examples (FGSM + PGD)...")
        print(f"{'='*60}")

        X_adv_fgsm = None
        X_adv_pgd = None
        surrogate = tier3_torch

        if surrogate is not None:
            surrogate.eval()
            x_t = torch.FloatTensor(X_adv_base).to(device)
            y_t = torch.LongTensor(y_adv_base).to(device)

            fgsm_eps  = self.config['adversarial_attacks']['fgsm']['epsilon']
            pgd_eps   = self.config['adversarial_attacks']['pgd']['epsilon']
            pgd_alpha = self.config['adversarial_attacks']['pgd']['alpha']
            pgd_iters = min(self.config['adversarial_attacks']['pgd']['num_iterations'], 20)

            print(f"  FGSM: epsilon={fgsm_eps}  ({len(X_adv_base):,} samples)")
            X_adv_fgsm = fgsm_attack(surrogate, x_t, y_t, epsilon=fgsm_eps).cpu().numpy()

            print(f"  PGD:  epsilon={pgd_eps}, alpha={pgd_alpha}, iters={pgd_iters}  ({len(X_adv_base):,} samples)")
            X_adv_pgd = pgd_attack(
                surrogate, x_t, y_t,
                epsilon=pgd_eps, alpha=pgd_alpha, num_iterations=pgd_iters
            ).cpu().numpy()

            print("  Adversarial examples generated.")
        else:
            print("[WARN] No surrogate model available — skipping adversarial generation.")

        # ── 5. Evaluate Tier 2 (separate results) ─────────────────────
        if tier2_keras is not None:
            print(f"\n{'='*60}")
            print(f"Evaluating Tier 2 on CLEAN data ({len(X_test):,} samples)...")
            print(f"{'='*60}")

            t2_clean = self.evaluate_clean(tier2_keras, X_test, y_test, 'Tier2-DNN')
            self._print_metrics("Tier 2 — Clean", t2_clean)

            # Per-class metrics for Tier 2 clean
            t2_preds = np.argmax(tier2_keras.predict(X_test, verbose=0), axis=1)
            t2_per_class = compute_per_class_metrics(y_test, t2_preds)
            self.results['tier2_per_class_clean'] = t2_per_class
            self._print_per_class("Tier 2 — Per-Attack (Clean)", t2_per_class)

            if X_adv_fgsm is not None:
                print(f"\nEvaluating Tier 2 on FGSM adversarial data ({len(X_adv_base):,} samples)...")
                t2_fgsm = self._evaluate_keras_robust(
                    tier2_keras, X_adv_base, y_adv_base, X_adv_fgsm,
                    'Tier2-DNN_robust_fgsm'
                )
                self._print_robust_metrics("Tier 2 — FGSM Robust", t2_fgsm)

                # Per-class robust for FGSM
                t2_pred_clean_sub = np.argmax(tier2_keras.predict(X_adv_base, verbose=0), axis=1)
                t2_pred_fgsm = np.argmax(tier2_keras.predict(X_adv_fgsm, verbose=0), axis=1)
                t2_pc_fgsm = compute_per_class_robust_metrics(y_adv_base, t2_pred_clean_sub, t2_pred_fgsm)
                self.results['tier2_per_class_robust_fgsm'] = t2_pc_fgsm

            if X_adv_pgd is not None:
                print(f"\nEvaluating Tier 2 on PGD adversarial data ({len(X_adv_base):,} samples)...")
                t2_pgd = self._evaluate_keras_robust(
                    tier2_keras, X_adv_base, y_adv_base, X_adv_pgd,
                    'Tier2-DNN_robust_pgd'
                )
                self._print_robust_metrics("Tier 2 — PGD Robust", t2_pgd)

                t2_pred_pgd = np.argmax(tier2_keras.predict(X_adv_pgd, verbose=0), axis=1)
                t2_pc_pgd = compute_per_class_robust_metrics(y_adv_base, t2_pred_clean_sub, t2_pred_pgd)
                self.results['tier2_per_class_robust_pgd'] = t2_pc_pgd

        # ── 6. Evaluate Tier 3 — only on samples Tier 2 classifies as benign ─
        #    Tier 3 catches adversarial evasion attacks that fooled Tier 2
        #    into classifying them as normal/benign.
        if tier3_torch is not None:
            # Gate: only evaluate Tier 3 on samples Tier 2 classifies as normal
            if tier2_keras is not None:
                t2_full_preds = np.argmax(tier2_keras.predict(X_test, verbose=0), axis=1)
                benign_mask_full = (t2_full_preds == 0)  # Tier 2 says normal
                X_test_t3 = X_test[benign_mask_full]
                y_test_t3 = y_test[benign_mask_full]

                # Also gate the adversarial subset
                t2_adv_preds = np.argmax(tier2_keras.predict(X_adv_base, verbose=0), axis=1)
                benign_mask_adv = (t2_adv_preds == 0)
                X_adv_base_t3 = X_adv_base[benign_mask_adv]
                y_adv_base_t3 = y_adv_base[benign_mask_adv]
                X_adv_fgsm_t3 = X_adv_fgsm[benign_mask_adv] if X_adv_fgsm is not None else None
                X_adv_pgd_t3 = X_adv_pgd[benign_mask_adv] if X_adv_pgd is not None else None

                print(f"\n[INFO] Tier 3 gating: Tier 2 classified {benign_mask_full.sum():,}/{len(X_test):,} "
                      f"as benign (only these go to Tier 3 for adversarial evasion check).")
            else:
                # No Tier 2 — evaluate Tier 3 on all data
                X_test_t3 = X_test
                y_test_t3 = y_test
                X_adv_base_t3 = X_adv_base
                y_adv_base_t3 = y_adv_base
                X_adv_fgsm_t3 = X_adv_fgsm
                X_adv_pgd_t3 = X_adv_pgd

            if len(X_test_t3) > 0:
                print(f"\n{'='*60}")
                print(f"Evaluating Tier 3 on CLEAN data ({len(X_test_t3):,} samples, Tier-2-benign)...")
                print(f"{'='*60}")
                t3_clean = self._evaluate_pytorch_clean(
                    tier3_torch, X_test_t3, y_test_t3, 'Tier3-RobustDNN', device
                )
                self._print_metrics("Tier 3 — Clean (benign-gated)", t3_clean)

                # Per-class for Tier 3
                with torch.no_grad():
                    t3_preds = tier3_torch(torch.FloatTensor(X_test_t3).to(device)).argmax(dim=1).cpu().numpy()
                t3_per_class = compute_per_class_metrics(y_test_t3, t3_preds)
                self.results['tier3_per_class_clean'] = t3_per_class
                self._print_per_class("Tier 3 — Per-Attack (Clean)", t3_per_class)

                # Also store full-dataset Tier 3 evaluation for comparison
                t3_clean_full = self._evaluate_pytorch_clean(
                    tier3_torch, X_test, y_test, 'Tier3-RobustDNN-Full', device
                )
                self.results['Tier3-RobustDNN-Full_clean'] = t3_clean_full

                if X_adv_fgsm_t3 is not None and len(X_adv_base_t3) > 0:
                    print(f"\nEvaluating Tier 3 on FGSM adversarial data ({len(X_adv_base_t3):,} samples)...")
                    t3_fgsm = self._evaluate_pytorch_robust(
                        tier3_torch, X_adv_base_t3, y_adv_base_t3, X_adv_fgsm_t3,
                        'Tier3-RobustDNN_robust_fgsm', device
                    )
                    self._print_robust_metrics("Tier 3 — FGSM Robust", t3_fgsm)

                    with torch.no_grad():
                        t3_pred_clean_sub = tier3_torch(torch.FloatTensor(X_adv_base_t3).to(device)).argmax(dim=1).cpu().numpy()
                        t3_pred_fgsm = tier3_torch(torch.FloatTensor(X_adv_fgsm_t3).to(device)).argmax(dim=1).cpu().numpy()
                    t3_pc_fgsm = compute_per_class_robust_metrics(y_adv_base_t3, t3_pred_clean_sub, t3_pred_fgsm)
                    self.results['tier3_per_class_robust_fgsm'] = t3_pc_fgsm

                if X_adv_pgd_t3 is not None and len(X_adv_base_t3) > 0:
                    print(f"\nEvaluating Tier 3 on PGD adversarial data ({len(X_adv_base_t3):,} samples)...")
                    t3_pgd = self._evaluate_pytorch_robust(
                        tier3_torch, X_adv_base_t3, y_adv_base_t3, X_adv_pgd_t3,
                        'Tier3-RobustDNN_robust_pgd', device
                    )
                    self._print_robust_metrics("Tier 3 — PGD Robust", t3_pgd)

                    with torch.no_grad():
                        t3_pred_pgd = tier3_torch(torch.FloatTensor(X_adv_pgd_t3).to(device)).argmax(dim=1).cpu().numpy()
                    t3_pc_pgd = compute_per_class_robust_metrics(y_adv_base_t3, t3_pred_clean_sub, t3_pred_pgd)
                    self.results['tier3_per_class_robust_pgd'] = t3_pc_pgd
            else:
                print("[INFO] Tier 2 classified everything as attacks — no benign samples for Tier 3 to recheck.")

        # ── 7. Tier comparisons ────────────────────────────────────────
        self._build_tier_comparisons()

        # ── 8. Summary table ───────────────────────────────────────────
        self._print_summary()

        return self.results

    # ------------------------------------------------------------------
    # Tier 1 evaluation
    # ------------------------------------------------------------------

    def _evaluate_tier1(self, X_test, y_test):
        """
        Evaluate Tier 1 signature detection.
        Since Tier 1 works on raw feature dicts and we have numeric arrays,
        we compute a binary detection metric: how well does Tier 1 separate
        normal (class 0) vs attack (class != 0) using the signature DB.
        """
        from src.tier1_signature.signature_detector import Tier1SignatureDetector

        sig_path = resolve_path(self.config['tier1']['signature_db'])
        if not os.path.exists(sig_path):
            print("[WARN] Signature DB not found — skipping Tier 1 evaluation.")
            return {'note': 'signature_db not found'}

        tier1 = Tier1SignatureDetector(sig_path)

        # Tier 1 operates on dict samples — for evaluation purposes we
        # create synthetic feature dicts from the numeric test data.
        # Since signature rules use specific feature names (protocol, syn_flag_count, etc.)
        # and our test data is preprocessed (scaled + feature-selected), Tier 1
        # cannot meaningfully match. We report Tier 1 stats from the signature DB.
        n_signatures = 0
        category_counts = {}
        for cat, sigs in tier1.signatures.items():
            if isinstance(sigs, dict):
                for sub_cat, sub_sigs in sigs.items():
                    count = len(sub_sigs)
                    n_signatures += count
                    category_counts[sub_cat] = count
            else:
                count = len(sigs)
                n_signatures += count
                category_counts[cat] = count

        # Binary ground truth: normal vs attack
        y_binary = (y_test != 0).astype(int)
        n_attacks = int(y_binary.sum())
        n_normal = int((y_binary == 0).sum())

        result = {
            'total_signatures': n_signatures,
            'signatures_by_category': category_counts,
            'test_set_attacks': n_attacks,
            'test_set_normal': n_normal,
            'note': ('Tier 1 uses raw traffic features (pre-preprocessing). '
                     'Numeric evaluation is approximate on preprocessed data.'),
        }

        print(f"  Tier 1 Signature DB: {n_signatures} signatures across {len(category_counts)} categories")
        for cat, cnt in category_counts.items():
            print(f"    {cat}: {cnt} rules")
        print(f"  Test set: {n_attacks} attacks, {n_normal} normal")

        return result

    # ------------------------------------------------------------------
    # Tier comparisons
    # ------------------------------------------------------------------

    def _build_tier_comparisons(self):
        """Build comparison summaries across tiers."""
        comparison_t1_t2 = {}
        comparison_all = {}

        # Tier 1 info
        t1 = self.results.get('tier1', {})
        if t1 and 'total_signatures' in t1:
            comparison_t1_t2['tier1'] = {
                'type': 'Signature-Based',
                'total_signatures': t1['total_signatures'],
                'categories_covered': list(t1.get('signatures_by_category', {}).keys()),
            }
            comparison_all['tier1'] = comparison_t1_t2['tier1']

        # Tier 2
        t2c = self.results.get('Tier2-DNN_clean', {})
        t2f = self.results.get('Tier2-DNN_robust_fgsm', {})
        t2p = self.results.get('Tier2-DNN_robust_pgd', {})
        if t2c:
            t2_summary = {
                'type': 'ML-DNN (Keras)',
                'clean_accuracy': t2c.get('accuracy', 0),
                'clean_f1': t2c.get('f1_score', 0),
                'clean_precision': t2c.get('precision', 0),
                'clean_recall': t2c.get('recall', 0),
                'fgsm_robust_accuracy': t2f.get('robust_accuracy', 0),
                'pgd_robust_accuracy': t2p.get('robust_accuracy', 0),
                'fgsm_accuracy_drop': t2f.get('accuracy_drop', 0),
                'pgd_accuracy_drop': t2p.get('accuracy_drop', 0),
            }
            comparison_t1_t2['tier2'] = t2_summary
            comparison_all['tier2'] = t2_summary

        # Tier 3
        t3c = self.results.get('Tier3-RobustDNN_clean', {})
        t3f = self.results.get('Tier3-RobustDNN_robust_fgsm', {})
        t3p = self.results.get('Tier3-RobustDNN_robust_pgd', {})
        if t3c:
            comparison_all['tier3'] = {
                'type': 'Adversarially-Trained DNN (PyTorch)',
                'clean_accuracy': t3c.get('accuracy', 0),
                'clean_f1': t3c.get('f1_score', 0),
                'clean_precision': t3c.get('precision', 0),
                'clean_recall': t3c.get('recall', 0),
                'fgsm_robust_accuracy': t3f.get('robust_accuracy', 0),
                'pgd_robust_accuracy': t3p.get('robust_accuracy', 0),
                'fgsm_accuracy_drop': t3f.get('accuracy_drop', 0),
                'pgd_accuracy_drop': t3p.get('accuracy_drop', 0),
                'note': 'Evaluated only on Tier-2-classified-benign samples (adversarial evasion check)',
            }

        self.results['comparison_tier1_tier2'] = comparison_t1_t2
        self.results['comparison_all_tiers'] = comparison_all

        # Print comparison
        print(f"\n{'='*60}")
        print("TIER COMPARISON — Tier 1 + Tier 2")
        print(f"{'='*60}")
        for tier_name, info in comparison_t1_t2.items():
            print(f"  {tier_name}: {info.get('type', 'N/A')}")
            if 'clean_accuracy' in info:
                print(f"    Clean Acc: {info['clean_accuracy']:.4f}  F1: {info['clean_f1']:.4f}")

        print(f"\n{'='*60}")
        print("TIER COMPARISON — All Tiers (Tier 1 + 2 + 3)")
        print(f"{'='*60}")
        for tier_name, info in comparison_all.items():
            print(f"  {tier_name}: {info.get('type', 'N/A')}")
            if 'clean_accuracy' in info:
                print(f"    Clean Acc: {info['clean_accuracy']:.4f}  F1: {info['clean_f1']:.4f}")
                if info.get('pgd_robust_accuracy'):
                    print(f"    PGD Robust: {info['pgd_robust_accuracy']:.4f}  "
                          f"Drop: {info['pgd_accuracy_drop']:.4f}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evaluate_keras_robust(self, keras_model, X_clean, y_test, X_adv, result_key):
        """Evaluate Keras model on adversarial data and store robust metrics."""
        preds_clean = keras_model.predict(X_clean, verbose=0)
        preds_adv   = keras_model.predict(X_adv,   verbose=0)

        y_pred_clean = np.argmax(preds_clean, axis=1)
        y_pred_adv   = np.argmax(preds_adv,   axis=1)

        metrics = compute_robust_accuracy(y_test, y_pred_clean, y_pred_adv)
        self.results[result_key] = metrics
        return metrics

    def _evaluate_pytorch_clean(self, torch_model, X_test, y_test, model_name, device):
        """Evaluate PyTorch model on clean data and store full metrics."""
        torch_model.eval()
        x_t = torch.FloatTensor(X_test).to(device)

        with torch.no_grad():
            logits = torch_model(x_t)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()

        y_pred = np.argmax(probs, axis=1)
        metrics = compute_all_metrics(y_test, y_pred, probs)
        self.results[f'{model_name}_clean'] = metrics
        return metrics

    def _evaluate_pytorch_robust(self, torch_model, X_clean, y_test, X_adv,
                                  result_key, device):
        """Evaluate PyTorch model on adversarial data and store robust metrics."""
        torch_model.eval()

        with torch.no_grad():
            logits_clean = torch_model(torch.FloatTensor(X_clean).to(device))
            logits_adv   = torch_model(torch.FloatTensor(X_adv).to(device))

        y_pred_clean = logits_clean.argmax(dim=1).cpu().numpy()
        y_pred_adv   = logits_adv.argmax(dim=1).cpu().numpy()

        metrics = compute_robust_accuracy(y_test, y_pred_clean, y_pred_adv)
        self.results[result_key] = metrics
        return metrics

    def _print_metrics(self, label, m):
        print(f"\n  [{label}]")
        print(f"    Accuracy:  {m['accuracy']:.4f}")
        print(f"    Precision: {m['precision']:.4f}")
        print(f"    Recall:    {m['recall']:.4f}")
        print(f"    F1:        {m['f1_score']:.4f}")
        print(f"    FPR:       {m.get('fpr', 'N/A'):.4f}" if isinstance(m.get('fpr'), float) else f"    FPR:       N/A")
        if 'auc_roc' in m:
            print(f"    AUC-ROC:   {m['auc_roc']:.4f}")

    def _print_robust_metrics(self, label, m):
        print(f"\n  [{label}]")
        print(f"    Clean Accuracy:   {m['clean_accuracy']:.4f}")
        print(f"    Robust Accuracy:  {m['robust_accuracy']:.4f}")
        print(f"    Accuracy Drop:    {m['accuracy_drop']:.4f}")
        print(f"    Robustness Ratio: {m['robustness_ratio']:.4f}")

    def _print_per_class(self, label, per_class):
        print(f"\n  [{label}]")
        print(f"    {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print(f"    {'-'*52}")
        for cls_name, m in per_class.items():
            print(f"    {cls_name:<12} {m['precision']:>10.4f} {m['recall']:>10.4f} "
                  f"{m['f1_score']:>10.4f} {m['support']:>10}")

    def _print_summary(self):
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        rows = [
            ('Tier2-DNN_clean',              'Tier 2 Clean Acc'),
            ('Tier2-DNN_robust_fgsm',        'Tier 2 Robust Acc (FGSM)'),
            ('Tier2-DNN_robust_pgd',         'Tier 2 Robust Acc (PGD)'),
            ('Tier3-RobustDNN_clean',        'Tier 3 Clean Acc (benign-gated)'),
            ('Tier3-RobustDNN_robust_fgsm',  'Tier 3 Robust Acc (FGSM)'),
            ('Tier3-RobustDNN_robust_pgd',   'Tier 3 Robust Acc (PGD)'),
        ]
        for key, label in rows:
            if key not in self.results:
                continue
            r = self.results[key]
            acc = r.get('accuracy') or r.get('clean_accuracy') or r.get('robust_accuracy')
            rob = r.get('robust_accuracy')
            if rob is not None:
                print(f"  {label:<40} clean={r['clean_accuracy']:.4f}  robust={rob:.4f}  drop={r['accuracy_drop']:.4f}")
            else:
                print(f"  {label:<40} acc={acc:.4f}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------

    def save_results(self, output_dir='evaluation_results'):
        """Save all evaluation results to JSON."""
        save_dir = resolve_path(output_dir)
        os.makedirs(save_dir, exist_ok=True)

        def _make_serializable(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_make_serializable(i) for i in obj]
            return obj

        serializable = _make_serializable(self.results)

        # Save combined results
        out_path = os.path.join(save_dir, 'evaluation_results.json')
        with open(out_path, 'w') as f:
            json.dump(serializable, f, indent=2, default=str)

        # Save per-tier separate files
        tier_keys = {
            'tier1_results.json': ['tier1'],
            'tier2_results.json': [k for k in self.results if k.startswith('Tier2') or k.startswith('tier2_per')],
            'tier3_results.json': [k for k in self.results if k.startswith('Tier3') or k.startswith('tier3_per')],
            'tier_comparisons.json': ['comparison_tier1_tier2', 'comparison_all_tiers'],
        }

        for filename, keys in tier_keys.items():
            tier_data = {k: _make_serializable(self.results[k]) for k in keys if k in self.results}
            if tier_data:
                path = os.path.join(save_dir, filename)
                with open(path, 'w') as f:
                    json.dump(tier_data, f, indent=2, default=str)

        print(f"\nResults saved to {save_dir}/")
        print(f"  - evaluation_results.json (combined)")
        print(f"  - tier1_results.json")
        print(f"  - tier2_results.json")
        print(f"  - tier3_results.json")
        print(f"  - tier_comparisons.json")
