import json
import os
from typing import Dict, List


class SignatureDatabase:
    """Manage the signature database: load, add, validate, save."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.signatures = self._load(db_path) if os.path.exists(db_path) else {}

    def _load(self, path: str) -> Dict:
        with open(path, 'r') as f:
            return json.load(f)

    def get_all_categories(self) -> List[str]:
        """Return all signature categories."""
        return list(self.signatures.keys())

    def get_signatures_for_category(self, category: str) -> List[Dict]:
        """Get all signatures for a specific category."""
        sigs = self.signatures.get(category, [])
        if isinstance(sigs, dict):
            # Handle nested structure like nsl_kdd_rules
            all_sigs = []
            for sub_sigs in sigs.values():
                all_sigs.extend(sub_sigs)
            return all_sigs
        return sigs

    def add_signature(self, category: str, signature: Dict) -> None:
        """Add a new signature to a category."""
        if not self.validate_signature(signature):
            raise ValueError("Invalid signature format")
        if category not in self.signatures:
            self.signatures[category] = []
        self.signatures[category].append(signature)

    def validate_signature(self, signature: Dict) -> bool:
        """Validate that a signature has required fields."""
        required = ['id', 'name', 'severity', 'conditions']
        return all(key in signature for key in required)

    def get_total_count(self) -> int:
        """Total number of signatures across all categories."""
        count = 0
        for sigs in self.signatures.values():
            if isinstance(sigs, dict):
                for sub_sigs in sigs.values():
                    count += len(sub_sigs)
            else:
                count += len(sigs)
        return count

    def save(self) -> None:
        """Save the signature database to disk."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, 'w') as f:
            json.dump(self.signatures, f, indent=2)
