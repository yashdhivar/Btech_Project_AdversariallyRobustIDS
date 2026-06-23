import os
import yaml


def load_config(config_path='config/config.yaml'):
    """Load YAML configuration file."""
    # Resolve path relative to project root
    if not os.path.isabs(config_path):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(project_root, config_path)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_project_root():
    """Get the absolute path to the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve_path(relative_path):
    """Resolve a relative path to absolute using project root."""
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.join(get_project_root(), relative_path)
