import os
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
import re


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
    run_id = sanitize_for_filesystem(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_name}_{dataset}")
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


def save_representations(hidden_states, filepath):
    """
    Save the representations (hidden states) to a file.

    Args:
        hidden_states (torch.Tensor): The hidden states tensor from the model
        filepath (str): The file path to save the representations to
    """
    # Convert tensor to numpy for saving
    hidden_states_np = hidden_states.cpu().detach().numpy()

    # Save the representations as NPY format (efficient binary numpy format)
    np.save(filepath, hidden_states_np)

    # Save metadata in a small JSON file
    metadata_path = filepath.replace('.npy', '_metadata.json')
    metadata = {
        "shape": list(hidden_states_np.shape),
        "dtype": str(hidden_states_np.dtype),
        "min": float(hidden_states_np.min()),
        "max": float(hidden_states_np.max()),
        "mean": float(hidden_states_np.mean()),
        "std": float(hidden_states_np.std())
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)


def save_probe_weights(probes, filepath):
    """
    Save the linear probe weights to a file.

    Args:
        probes (list): List of trained probe models
        filepath (str): The file path to save the weights to
    """
    # Create a dict to store all probe weights
    probes_data = {}

    # Extract weights from each probe
    for i, probe in enumerate(probes):
        if hasattr(probe, 'linear') and hasattr(probe.linear, 'weight'):
            weights = probe.linear.weight.cpu().detach().numpy()
            bias = probe.linear.bias.cpu().detach().numpy() if hasattr(probe.linear, 'bias') else None

            # Save individual layer weights as NPY files for efficient access
            layer_dir = os.path.dirname(filepath)
            layer_filename = f"layer_{i}_weights.npy"
            layer_path = os.path.join(layer_dir, layer_filename)
            np.save(layer_path, weights)

            if bias is not None:
                bias_filename = f"layer_{i}_bias.npy"
                bias_path = os.path.join(layer_dir, bias_filename)
                np.save(bias_path, bias)

            # Store metadata for this layer
            probes_data[f"layer_{i}"] = {
                "weights_shape": list(weights.shape),
                "weights_file": layer_filename,
                "bias_file": bias_filename if bias is not None else None,
                "weights_stats": {
                    "min": float(weights.min()),
                    "max": float(weights.max()),
                    "mean": float(weights.mean()),
                    "std": float(weights.std())
                }
            }

            if bias is not None:
                probes_data[f"layer_{i}"]["bias_stats"] = {
                    "value": float(bias.item()) if bias.size == 1 else bias.tolist()
                }

    # Save the metadata as JSON
    with open(filepath, 'w') as f:
        json.dump(probes_data, f, indent=4)

    # Also save the entire set of probes as a PyTorch model file if possible
    try:
        if all(hasattr(probe, 'state_dict') for probe in probes):
            probes_state = [probe.state_dict() for probe in probes]
            torch_path = filepath.replace('.json', '.pt')
            torch.save(probes_state, torch_path)
    except Exception as e:
        # If saving as PyTorch model fails, we still have the NPY files
        print(f"Note: Could not save probes as PyTorch model: {e}")
        # This is not a critical error since we already saved the weights as NPY files


def save_graph(fig, filepath):
    """Save a matplotlib figure as an image."""
    fig.savefig(filepath)


def save_autoencoder_models(autoencoders, filepath):
    """
    Save the sparse autoencoder models to a file.

    Args:
        autoencoders (list): List of trained autoencoder models
        filepath (str): The base file path to save the models to
    """
    # Create a dict to store all autoencoder metadata
    autoencoders_data = {}

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Extract and save each autoencoder
    for i, autoencoder in enumerate(autoencoders):
        # Save the entire model as a PyTorch file
        model_filename = f"autoencoder_layer_{i}.pt"
        model_path = os.path.join(os.path.dirname(filepath), model_filename)
        torch.save(autoencoder.state_dict(), model_path)

        # Extract metadata about this autoencoder
        is_supervised = hasattr(autoencoder, 'classifier')

        # Get dimensions from encoder weights
        if hasattr(autoencoder, 'encoder') and hasattr(autoencoder.encoder, 'weight'):
            input_dim = autoencoder.encoder.weight.shape[1]
            bottleneck_dim = autoencoder.encoder.weight.shape[0]

            # Create metadata entry
            autoencoders_data[f"layer_{i}"] = {
                "model_file": model_filename,
                "model_type": autoencoder.__class__.__name__,
                "is_supervised": is_supervised,
                "input_dim": input_dim,
                "bottleneck_dim": bottleneck_dim,
                "bottleneck_ratio": float(bottleneck_dim) / float(input_dim),
                "tied_weights": not hasattr(autoencoder, 'decoder') or autoencoder.tied_weights
            }

    # Save the metadata as JSON
    metadata_path = f"{filepath}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(autoencoders_data, f, indent=4)
