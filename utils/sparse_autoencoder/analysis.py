import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from utils.file_manager import save_graph

def calculate_sparsity_percentage(h_activated):
    """Calculate percentage of neurons that are zero (inactive) in the hidden representation
    
    Args:
        h_activated: Activated representation of shape [batch_size, bottleneck_dim]
        
    Returns:
        sparsity_percentage: Percentage of neurons that are zero (inactive)
        active_neurons: Number of neurons that are active (non-zero)
        total_neurons: Total number of neurons
    """
    active_neurons = torch.sum(h_activated > 0).item()
    total_neurons = h_activated.numel()
    sparsity_percentage = 100 * (1 - active_neurons / total_neurons)
    
    return sparsity_percentage, active_neurons, total_neurons

def calculate_average_activation(h_activated):
    """Calculate average activation value for active neurons
    
    Args:
        h_activated: Activated representation of shape [batch_size, bottleneck_dim]
        
    Returns:
        avg_activation: Average activation value for active neurons
    """
    # Get only non-zero values
    active_mask = h_activated > 0
    if torch.sum(active_mask) > 0:
        active_values = h_activated[active_mask]
        avg_activation = torch.mean(active_values).item()
    else:
        avg_activation = 0.0
    
    return avg_activation

def calculate_l1_sparsity(h_activated):
    """Calculate L1 sparsity measure (average absolute value of activations)
    
    Args:
        h_activated: Activated representation of shape [batch_size, bottleneck_dim]
        
    Returns:
        l1_sparsity: Average absolute value of activations
    """
    return torch.mean(torch.abs(h_activated)).item()

def get_sparsity_metrics_by_layer(autoencoders, test_hidden_states):
    """Calculate sparsity metrics across all layers
    
    Args:
        autoencoders: List of trained autoencoder models
        test_hidden_states: Hidden states from test set [batch_size, num_layers, hidden_dim]
        
    Returns:
        metrics_by_layer: List of dictionaries with sparsity metrics for each layer
    """
    metrics_by_layer = []
    num_layers = len(autoencoders)
    
    for layer_idx in range(num_layers):
        # Get test features for this layer
        test_feats = test_hidden_states[:, layer_idx, :]
        
        # Get autoencoder for this layer
        autoencoder = autoencoders[layer_idx]
        
        # Forward pass through autoencoder (no gradients needed)
        with torch.no_grad():
            _, h_activated, _ = autoencoder(test_feats)
            
            # Calculate sparsity metrics
            sparsity_percentage, active_neurons, total_neurons = calculate_sparsity_percentage(h_activated)
            avg_activation = calculate_average_activation(h_activated)
            l1_sparsity = calculate_l1_sparsity(h_activated)
            
            # Store metrics for this layer
            metrics_by_layer.append({
                "layer": layer_idx,
                "sparsity_percentage": sparsity_percentage,
                "active_neurons": active_neurons,
                "total_neurons": total_neurons,
                "avg_activation": avg_activation,
                "l1_sparsity": l1_sparsity
            })
    
    return metrics_by_layer

def create_sparsity_dataframe(metrics_by_layer):
    """Create a DataFrame for displaying sparsity metrics
    
    Args:
        metrics_by_layer: List of dictionaries with sparsity metrics for each layer
        
    Returns:
        df: DataFrame with sparsity metrics
    """
    df = pd.DataFrame(metrics_by_layer)
    # Format percentages
    df["sparsity_percentage"] = df["sparsity_percentage"].map("{:.2f}%".format)
    # Format average activation
    df["avg_activation"] = df["avg_activation"].map("{:.4f}".format)
    # Format L1 sparsity
    df["l1_sparsity"] = df["l1_sparsity"].map("{:.4f}".format)
    
    return df

def plot_sparsity_by_layer(metrics_by_layer, model_name, dataset_source, run_folder=None):
    """Plot sparsity percentage by layer
    
    Args:
        metrics_by_layer: List of dictionaries with sparsity metrics for each layer
        model_name: Name of the model
        dataset_source: Name of the dataset
        run_folder: Folder to save the plot to
        
    Returns:
        fig: Matplotlib figure object
    """
    # Extract layer indices and sparsity percentages
    layers = [metrics["layer"] for metrics in metrics_by_layer]
    sparsity_percentages = [metrics["sparsity_percentage"] for metrics in metrics_by_layer]
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the data
    ax.plot(layers, sparsity_percentages, marker="o", linewidth=2)
    
    # Add titles and labels
    ax.set_title(f"Activation Sparsity per Layer ({model_name})", fontsize=14)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Sparsity Percentage (%)", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add exact values as text labels
    for i, sparsity in enumerate(sparsity_percentages):
        ax.annotate(f"{sparsity:.2f}%", (i, sparsity), textcoords="offset points",
                    xytext=(0, 5), ha='center')
    
    plt.tight_layout()
    
    # Save the figure if run_folder is provided
    if run_folder:
        save_graph(fig, os.path.join(run_folder, "sparsity_plot.png"))
    
    return fig

def plot_neuron_activations(autoencoder, test_feats, layer_idx, top_k=50, run_folder=None):
    """Plot the most active neurons for a specific layer
    
    Args:
        autoencoder: Trained autoencoder model for the layer
        test_feats: Test features for the layer [batch_size, hidden_dim]
        layer_idx: Layer index
        top_k: Number of top neurons to visualize
        run_folder: Folder to save the plot to
        
    Returns:
        fig: Matplotlib figure object
    """
    # Forward pass through autoencoder (no gradients needed)
    with torch.no_grad():
        _, h_activated, _ = autoencoder(test_feats)
        
        # Calculate mean activation per neuron
        mean_activations = torch.mean(h_activated, dim=0).cpu().numpy()
        
        # Get top-k most active neurons
        top_indices = np.argsort(mean_activations)[::-1][:top_k]
        top_activations = mean_activations[top_indices]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot top-k activations
        ax.bar(range(len(top_activations)), top_activations)
        ax.set_title(f"Top {top_k} Active Neurons - Layer {layer_idx}", fontsize=14)
        ax.set_xlabel("Neuron Rank", fontsize=12)
        ax.set_ylabel("Mean Activation", fontsize=12)
        ax.set_xticks(range(len(top_activations)))
        ax.set_xticklabels([f"{idx}" for idx in top_indices], rotation=90, fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save the figure if run_folder is provided
        if run_folder:
            layer_save_dir = os.path.join(run_folder, "layers", str(layer_idx))
            os.makedirs(layer_save_dir, exist_ok=True)
            save_graph(fig, os.path.join(layer_save_dir, "top_neurons.png"))
            
        return fig

def plot_activation_distribution(autoencoder, test_feats, layer_idx, run_folder=None):
    """Plot distribution of neuron activations for a specific layer
    
    Args:
        autoencoder: Trained autoencoder model for the layer
        test_feats: Test features for the layer [batch_size, hidden_dim]
        layer_idx: Layer index
        run_folder: Folder to save the plot to
        
    Returns:
        fig: Matplotlib figure object
    """
    # Forward pass through autoencoder (no gradients needed)
    with torch.no_grad():
        _, h_activated, _ = autoencoder(test_feats)
        
        # Flatten activations to 1D array for histogram
        activations_flat = h_activated.flatten().cpu().numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram of activations, excluding zeros
        non_zero_activations = activations_flat[activations_flat > 0]
        if len(non_zero_activations) > 0:
            ax.hist(non_zero_activations, bins=50, alpha=0.7)
            ax.set_title(f"Neuron Activation Distribution (Non-Zero) - Layer {layer_idx}", fontsize=14)
            ax.set_xlabel("Activation Value", fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            
            # Add sparsity information
            sparsity_percentage = 100 * (1 - len(non_zero_activations) / len(activations_flat))
            ax.text(0.95, 0.95, f"Sparsity: {sparsity_percentage:.2f}%", 
                    transform=ax.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, "No non-zero activations found", 
                    ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        # Save the figure if run_folder is provided
        if run_folder:
            layer_save_dir = os.path.join(run_folder, "layers", str(layer_idx))
            os.makedirs(layer_save_dir, exist_ok=True)
            save_graph(fig, os.path.join(layer_save_dir, "activation_distribution.png"))
            
        return fig

def plot_l1_sparsity_by_layer(metrics_by_layer, model_name, dataset_source, run_folder=None):
    """Plot L1 sparsity measure by layer
    
    Args:
        metrics_by_layer: List of dictionaries with sparsity metrics for each layer
        model_name: Name of the model
        dataset_source: Name of the dataset
        run_folder: Folder to save the plot to
        
    Returns:
        fig: Matplotlib figure object
    """
    # Extract layer indices and L1 sparsity values
    layers = [metrics["layer"] for metrics in metrics_by_layer]
    l1_sparsity_values = [metrics["l1_sparsity"] for metrics in metrics_by_layer]
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the data
    ax.plot(layers, l1_sparsity_values, marker="o", linewidth=2, color="#1E88E5")
    
    # Add titles and labels
    ax.set_title(f"L1 Sparsity Measure per Layer ({model_name})", fontsize=14)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("L1 Sparsity (Mean Absolute Activation)", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add exact values as text labels
    for i, sparsity in enumerate(l1_sparsity_values):
        ax.annotate(f"{sparsity:.4f}", (i, sparsity), textcoords="offset points",
                    xytext=(0, 5), ha='center')
    
    plt.tight_layout()
    
    # Save the figure if run_folder is provided
    if run_folder:
        save_graph(fig, os.path.join(run_folder, "l1_sparsity_plot.png"))
    
    return fig

def plot_reconstruction_error_by_layer(reconstruction_errors, model_name, dataset_source, run_folder=None):
    """Plot reconstruction error by layer
    
    Args:
        reconstruction_errors: List of reconstruction errors for each layer
        model_name: Name of the model
        dataset_source: Name of the dataset
        run_folder: Folder to save the plot to
        
    Returns:
        fig: Matplotlib figure object
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the data
    ax.plot(range(len(reconstruction_errors)), reconstruction_errors, marker="o", linewidth=2, color="#4CAF50")
    
    # Add titles and labels
    ax.set_title(f"Reconstruction Error per Layer ({model_name})", fontsize=14)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("MSE Reconstruction Error", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add exact values as text labels
    for i, error in enumerate(reconstruction_errors):
        ax.annotate(f"{error:.4f}", (i, error), textcoords="offset points",
                    xytext=(0, 5), ha='center')
    
    plt.tight_layout()
    
    # Save the figure if run_folder is provided
    if run_folder:
        save_graph(fig, os.path.join(run_folder, "reconstruction_error_plot.png"))
    
    return fig