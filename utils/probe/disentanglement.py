"""
Utilities for measuring disentanglement in sparse autoencoder representations.
This module analyzes how cleanly truth is encoded in sparse features.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.metrics import accuracy_score
import pandas as pd

from .linear_probe import LinearProbe


def extract_sparse_codes(sae_model, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Extract sparse codes (z) from the SAE encoder for given hidden states.
    
    Args:
        sae_model: Trained SparseAutoencoder model
        hidden_states: Hidden states from the model [n_samples, hidden_dim]
        
    Returns:
        Sparse codes z [n_samples, feature_dim]
    """
    with torch.no_grad():
        return sae_model.encode(hidden_states)


def train_z_probe(
    z_values: torch.Tensor, 
    labels: torch.Tensor,
    epochs: int = 100, 
    lr: float = 0.01,
    device: torch.device = torch.device("cpu")
) -> Tuple[LinearProbe, float]:
    """
    Train a linear probe on sparse codes (z).
    
    Args:
        z_values: Sparse codes [n_samples, feature_dim]
        labels: Truth labels [n_samples]
        epochs: Number of training epochs
        lr: Learning rate
        device: Compute device
        
    Returns:
        Tuple of (trained probe, accuracy)
    """
    # Move to device
    z_values = z_values.to(device)
    labels = labels.to(device)
    
    # Create and train probe
    input_dim = z_values.shape[1]
    probe = LinearProbe(input_dim).to(device)
    
    # Train the probe
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = probe(z_values)
        loss = criterion(outputs, labels.float().view(-1, 1))
        loss.backward()
        optimizer.step()
    
    # Evaluate accuracy
    with torch.no_grad():
        predictions = (probe(z_values) > 0.5).long()
        accuracy = (predictions == labels.view(-1, 1)).float().mean().item()
    
    return probe, accuracy


def calculate_gini_coefficient(weights: torch.Tensor) -> float:
    """
    Calculate Gini coefficient to measure inequality of feature importance.
    Higher values mean more concentrated distribution (more disentangled).
    
    Args:
        weights: Linear probe weights [feature_dim]
        
    Returns:
        Gini coefficient (0-1)
    """
    # Get absolute weights as measure of importance
    weights_abs = torch.abs(weights).flatten().detach().cpu().numpy()
    
    # Sort weights
    weights_sorted = np.sort(weights_abs)
    n = len(weights_sorted)
    
    # Calculate Gini coefficient
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * weights_sorted)) / (n * np.sum(weights_sorted))


def find_minimal_feature_set(
    probe: LinearProbe, 
    z_values: torch.Tensor, 
    labels: torch.Tensor,
    target_percent: float = 0.9
) -> Tuple[int, List[float]]:
    """
    Find the minimal number of features needed to reach X% of full accuracy.
    
    Args:
        probe: Trained linear probe
        z_values: Sparse codes [n_samples, feature_dim]
        labels: Truth labels [n_samples]
        target_percent: Target percentage of full accuracy (0-1)
        
    Returns:
        Tuple of (minimal feature count, accuracy curve)
    """
    device = z_values.device
    
    # Get feature importances
    weights = probe.linear.weight[0].detach().cpu().numpy()
    importance = np.abs(weights)
    
    # Sort features by importance
    sorted_indices = np.argsort(-importance)
    
    # Calculate full accuracy
    with torch.no_grad():
        full_preds = (probe(z_values) > 0.5).long().cpu().numpy()
        full_accuracy = accuracy_score(labels.cpu().numpy(), full_preds)
    
    # Calculate accuracy with increasing number of features
    accuracies = []
    min_features = z_values.shape[1]  # Default to all features
    
    for k in range(1, len(sorted_indices) + 1):
        # Create masked version with only top-k features
        top_k_indices = sorted_indices[:k]
        
        # Create a feature mask tensor
        mask = torch.zeros(len(weights), device=device)
        mask[top_k_indices] = 1.0
        
        # Apply mask to z values
        masked_z = z_values * mask
        
        # Get predictions
        with torch.no_grad():
            masked_preds = (probe(masked_z) > 0.5).long().cpu().numpy()
            accuracy = accuracy_score(labels.cpu().numpy(), masked_preds)
            accuracies.append(accuracy)
        
        # Check if we've reached target accuracy
        if accuracy >= target_percent * full_accuracy and k < min_features:
            min_features = k
    
    return min_features, accuracies


def get_feature_importance(probe: LinearProbe) -> np.ndarray:
    """
    Get feature importance scores based on probe weights.
    
    Args:
        probe: Trained linear probe
        
    Returns:
        Array of feature importance scores
    """
    return np.abs(probe.linear.weight[0].detach().cpu().numpy())


def plot_feature_importance_distribution(importance: np.ndarray, title: str = None) -> plt.Figure:
    """
    Create a bar plot showing the distribution of feature importance.
    
    Args:
        importance: Feature importance scores
        title: Optional plot title
        
    Returns:
        Matplotlib figure
    """
    # Sort importance
    sorted_importance = np.sort(importance)[::-1]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(sorted_importance)), sorted_importance)
    ax.set_xlabel('Feature Rank')
    ax.set_ylabel('Importance (|Weight|)')
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return fig


def plot_cumulative_accuracy(accuracies: List[float], min_features: int, title: str = None) -> plt.Figure:
    """
    Create a line plot showing cumulative accuracy as features are added.
    
    Args:
        accuracies: List of accuracies as features are added
        min_features: Minimal feature count needed for target accuracy
        title: Optional plot title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(accuracies) + 1), accuracies)
    ax.axvline(x=min_features, color='r', linestyle='--', 
               label=f'Min Features: {min_features}')
    ax.set_xlabel('Number of Features Used')
    ax.set_ylabel('Accuracy')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Cumulative Accuracy by Number of Features')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    return fig


def plot_disentanglement_metrics(metrics_df: pd.DataFrame) -> plt.Figure:
    """
    Create a bar plot comparing disentanglement metrics across layers.
    
    Args:
        metrics_df: DataFrame with metrics by layer
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get layers and metrics
    layers = metrics_df['Layer']
    gini = metrics_df['Gini Coefficient']
    min_features_pct = metrics_df['Min Features %']
    
    # Set width of bars
    bar_width = 0.35
    x = np.arange(len(layers))
    
    # Create bars
    ax.bar(x - bar_width/2, gini, bar_width, label='Gini Coefficient')
    ax.bar(x + bar_width/2, min_features_pct, bar_width, label='Min Features %')
    
    # Add labels and legend
    ax.set_xlabel('Layer')
    ax.set_ylabel('Metric Value')
    ax.set_title('Disentanglement Metrics by Layer')
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    
    return fig


def analyze_disentanglement(
    sae_models: Dict[int, Any],
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device = torch.device("cpu"),
    epochs: int = 100,
    lr: float = 0.01,
    target_percent: float = 0.9
) -> Dict[str, Any]:
    """
    Analyze disentanglement of truth in sparse representations across layers.
    
    Args:
        sae_models: Dictionary of {layer_idx: sae_model} pairs
        hidden_states: Hidden states [n_samples, n_layers, hidden_dim]
        labels: Truth labels [n_samples]
        device: Compute device
        epochs: Number of training epochs for probes
        lr: Learning rate for probes
        target_percent: Target percentage of accuracy for minimal feature set
        
    Returns:
        Dictionary with disentanglement analysis results
    """
    results = {}
    metrics = []
    
    for layer_idx, sae_model in sae_models.items():
        layer_hidden_states = hidden_states[:, layer_idx, :]
        
        # Extract sparse codes
        z_values = extract_sparse_codes(sae_model, layer_hidden_states)
        
        # Train probe on z
        z_probe, z_accuracy = train_z_probe(z_values, labels, epochs, lr, device)
        
        # Calculate Gini coefficient
        gini = calculate_gini_coefficient(z_probe.linear.weight)
        
        # Find minimal feature set
        min_features, accuracies = find_minimal_feature_set(
            z_probe, z_values, labels, target_percent)
        
        # Calculate feature importance
        importance = get_feature_importance(z_probe)
        
        # Store results
        results[layer_idx] = {
            'z_accuracy': z_accuracy,
            'gini_coefficient': gini,
            'min_features': min_features,
            'min_features_percent': min_features / z_values.shape[1],
            'accuracies': accuracies,
            'importance': importance,
            'z_probe': z_probe
        }
        
        # Add to metrics dataframe
        metrics.append({
            'Layer': layer_idx,
            'Z Accuracy': z_accuracy,
            'Gini Coefficient': gini,
            'Min Features': min_features,
            'Min Features %': min_features / z_values.shape[1],
            'Feature Dim': z_values.shape[1]
        })
    
    # Create metrics dataframe
    results['metrics_df'] = pd.DataFrame(metrics)
    
    return results