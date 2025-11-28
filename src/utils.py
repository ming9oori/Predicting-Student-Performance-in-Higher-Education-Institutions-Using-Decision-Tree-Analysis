"""
Utility functions for the project.
"""

from pathlib import Path


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_data_dir():
    """Get the data directory, creating it if it doesn't exist."""
    data_dir = get_project_root() / 'data'
    data_dir.mkdir(exist_ok=True)
    return data_dir


def get_models_dir():
    """Get the models directory, creating it if it doesn't exist."""
    models_dir = get_project_root() / 'models'
    models_dir.mkdir(exist_ok=True)
    return models_dir


def get_results_dir():
    """Get the results directory, creating it if it doesn't exist."""
    results_dir = get_project_root() / 'results'
    results_dir.mkdir(exist_ok=True)
    return results_dir
