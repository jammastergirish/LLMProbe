import os
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import re
import torch


def sanitize_for_filesystem(name):
    """
    Sanitize a string to be safely used as a folder or file name
    by removing or replacing potentially problematic characters.
    """
    # Replace characters that are not allowed in most filesystems
    # Windows is most restrictive, so we'll use its rules
    # Disallowed: < > : " / \ | ? * and control characters

    # First, replace slashes (already being done for model_name)
    name = name.replace("/", "_").replace("\\", "_")

    # Replace other problematic characters
    name = re.sub(r'[<>:"|?*]', '_', name)

    # Remove control characters
    name = re.sub(r'[\x00-\x1f\x7f]', '', name)

    # Trim leading/trailing whitespace and periods
    # (periods at end of folder names can cause issues in Windows)
    name = name.strip().strip('.')

    # Maximum length consideration (255 bytes is common limit)
    if len(name.encode('utf-8')) > 255:
        # Truncate to fit within byte limit while preserving unicode characters
        while len(name.encode('utf-8')) > 255:
            name = name[:-1]

    # Ensure the name is not empty after sanitization
    if not name:
        name = "unnamed"

    return name

SAVED_DATA_DIR = "saved_data"


def create_run_folder(model_name, dataset):
    """Create a unique folder for the current run."""
    os.makedirs(SAVED_DATA_DIR, exist_ok=True)
    
    # Sanitize each component
    safe_model_name = sanitize_for_filesystem(model_name)
    safe_dataset = sanitize_for_filesystem(dataset)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create run ID from sanitized components
    run_id = f"{timestamp}_{safe_model_name}_{safe_dataset}"
    
    # Create folder
    run_folder = os.path.join(SAVED_DATA_DIR, run_id)
    os.makedirs(run_folder, exist_ok=True)
    return run_folder, run_id


def save_json(data, filepath):
    """Save data as a JSON file, ensuring all objects are JSON serializable."""
    def convert(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist()
        elif isinstance(obj, (set, tuple)):
            return list(obj)
        raise TypeError(
            f"Object of type {type(obj).__name__} is not JSON serializable")

    with open(filepath, "w") as f:
        json.dump(data, f, indent=4, default=convert)


def save_graph(fig, filepath):
    """Save a matplotlib figure as an image."""
    fig.savefig(filepath)
