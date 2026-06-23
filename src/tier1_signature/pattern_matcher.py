from typing import Dict, List, Tuple


class PatternMatcher:
    """Pattern matching algorithms for signature detection."""

    def __init__(self, match_threshold: float = 0.8):
        self.threshold = match_threshold

    def match_sample(self, sample: Dict, signature: Dict) -> Tuple[bool, float]:
        """
        Match a single sample against a signature.

        Returns:
            (is_match, match_ratio)
        """
        conditions = signature.get('conditions', {})
        if not conditions:
            return False, 0.0

        match_count = 0
        total = len(conditions)

        for feature, rule in conditions.items():
            value = sample.get(feature)
            if value is None:
                continue

            if self._check_condition(value, rule):
                match_count += 1

        ratio = match_count / total if total > 0 else 0.0
        return ratio >= self.threshold, ratio

    def _check_condition(self, value, rule) -> bool:
        """Check if a value satisfies a rule."""
        if isinstance(rule, dict):
            if 'min' in rule and value < rule['min']:
                return False
            if 'max' in rule and value > rule['max']:
                return False
            return True
        elif isinstance(rule, list):
            return value in rule
        else:
            return value == rule

    def match_batch(self, samples: List[Dict], signatures: Dict) -> List[Dict]:
        """Match a batch of samples against all signatures."""
        results = []
        for sample in samples:
            matched = False
            for category, sigs in signatures.items():
                if isinstance(sigs, dict):
                    for sub_cat, sub_sigs in sigs.items():
                        for sig in sub_sigs:
                            is_match, ratio = self.match_sample(sample, sig)
                            if is_match:
                                results.append({
                                    'matched': True,
                                    'category': sub_cat,
                                    'signature': sig,
                                    'ratio': ratio
                                })
                                matched = True
                                break
                        if matched:
                            break
                else:
                    for sig in sigs:
                        is_match, ratio = self.match_sample(sample, sig)
                        if is_match:
                            results.append({
                                'matched': True,
                                'category': category,
                                'signature': sig,
                                'ratio': ratio
                            })
                            matched = True
                            break
                if matched:
                    break
            if not matched:
                results.append({'matched': False, 'category': None, 'signature': None, 'ratio': 0.0})
        return results
