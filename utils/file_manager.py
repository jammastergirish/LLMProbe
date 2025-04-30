import os
import uuid
import json
import matplotlib.pyplot as plt
import numpy as np

SAVED_DATA_DIR = "saved_data"


def create_run_folder():
    """Create a unique folder for the current run."""
    os.makedirs(SAVED_DATA_DIR, exist_ok=True)
    run_id = str(uuid.uuid4())
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
