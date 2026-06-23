import json
import time
from typing import Dict, List


class Tier1SignatureDetector:
    """
    Signature-based intrusion detector.
    Matches network traffic against known attack signatures.
    """

    def __init__(self, signature_db_path: str):
        self.signatures = self._load_signatures(signature_db_path)
        self.detection_count = 0

    def _load_signatures(self, path: str) -> Dict:
        """Load signature database from JSON file."""
        with open(path, 'r') as f:
            return json.load(f)

    def _match_single_signature(self, sample: Dict, signature: Dict, threshold: float = 0.8) -> bool:
        """
        Check if a traffic sample matches a specific signature.
        Uses threshold-based matching for numerical features
        and exact matching for categorical features.
        """
        conditions = signature.get('conditions', {})
        match_count = 0
        total_conditions = len(conditions)

        for feature, rule in conditions.items():
            sample_value = sample.get(feature)
            if sample_value is None:
                continue

            if isinstance(rule, dict):
                # Range-based matching
                matched = True
                if 'min' in rule and sample_value < rule['min']:
                    matched = False
                if 'max' in rule and sample_value > rule['max']:
                    matched = False
                if matched:
                    match_count += 1
            elif isinstance(rule, list):
                # Value in list
                if sample_value in rule:
                    match_count += 1
            else:
                # Exact match
                if sample_value == rule:
                    match_count += 1

        # Match if threshold% conditions are satisfied
        match_ratio = match_count / total_conditions if total_conditions > 0 else 0
        return match_ratio >= threshold

    def detect(self, traffic_sample: Dict) -> Dict:
        """
        Run signature detection on a single traffic sample.

        Returns:
            {
                'is_attack': bool,
                'attack_type': str or None,
                'signature_id': str or None,
                'severity': str,
                'confidence': 'CERTAIN',
                'tier': 1,
                'detection_time_ms': float
            }
        """
        start_time = time.time()

        for category, sigs in self.signatures.items():
            # Handle nested nsl_kdd_rules structure
            if isinstance(sigs, dict):
                for sub_category, sub_sigs in sigs.items():
                    for sig in sub_sigs:
                        if self._match_single_signature(traffic_sample, sig):
                            self.detection_count += 1
                            detection_time = (time.time() - start_time) * 1000

                            return {
                                'is_attack': True,
                                'attack_type': sig.get('name', sub_category),
                                'attack_category': sub_category,
                                'signature_id': sig.get('id'),
                                'severity': sig.get('severity', 'MEDIUM'),
                                'confidence': 'CERTAIN',
                                'tier': 1,
                                'detection_time_ms': detection_time
                            }
            else:
                # Standard flat list of signatures
                for sig in sigs:
                    if self._match_single_signature(traffic_sample, sig):
                        self.detection_count += 1
                        detection_time = (time.time() - start_time) * 1000

                        return {
                            'is_attack': True,
                            'attack_type': sig.get('name', category),
                            'attack_category': category,
                            'signature_id': sig.get('id'),
                            'severity': sig.get('severity', 'MEDIUM'),
                            'confidence': 'CERTAIN',
                            'tier': 1,
                            'detection_time_ms': detection_time
                        }

        detection_time = (time.time() - start_time) * 1000
        return {
            'is_attack': False,
            'attack_type': None,
            'tier': 1,
            'detection_time_ms': detection_time
        }

    def detect_batch(self, traffic_batch: List[Dict]) -> List[Dict]:
        """Run signature detection on a batch of samples."""
        return [self.detect(sample) for sample in traffic_batch]
