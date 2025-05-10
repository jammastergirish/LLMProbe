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

def calculate_gini_coefficient(h_activated):
    """Calculate Gini coefficient for the activations

    The Gini coefficient is a measure of inequality, where 0 represents perfect equality
    (all neurons equally active) and 1 represents perfect inequality (one neuron has all activation).
    It's a common measure of sparsity in neural network activations.

    Args:
        h_activated: Activated representation of shape [batch_size, bottleneck_dim]

    Returns:
        gini: Gini coefficient value (between 0 and 1)
    """
    # Get absolute values of activations and flatten
    values = torch.abs(h_activated).flatten().cpu().numpy()

    # Sort values in ascending order
    values = np.sort(values)

    # Skip calculation if all values are zero
    if np.sum(values) == 0:
        return 0.0

    # Calculate Gini coefficient
    n = len(values)
    cumulative_sum = np.cumsum(values)
    cumulative_proportion = cumulative_sum / cumulative_sum[-1]
    area_under_curve = np.sum(cumulative_proportion)

    # Calculate the Gini coefficient (area between Lorenz curve and equality line)
    gini = 1 - 2 * area_under_curve / n + 1 / n

    return gini

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
            gini_coefficient = calculate_gini_coefficient(h_activated)

            # Store metrics for this layer
            metrics_by_layer.append({
                "layer": layer_idx,
                "sparsity_percentage": sparsity_percentage,
                "active_neurons": active_neurons,
                "total_neurons": total_neurons,
                "avg_activation": avg_activation,
                "l1_sparsity": l1_sparsity,
                "gini_coefficient": gini_coefficient
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
    # Format Gini coefficient
    df["gini_coefficient"] = df["gini_coefficient"].map("{:.4f}".format)

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

def plot_gini_coefficient_by_layer(metrics_by_layer, model_name, dataset_source, run_folder=None):
    """Plot Gini coefficient by layer

    Args:
        metrics_by_layer: List of dictionaries with sparsity metrics for each layer
        model_name: Name of the model
        dataset_source: Name of the dataset
        run_folder: Folder to save the plot to

    Returns:
        fig: Matplotlib figure object
    """
    # Extract layer indices and Gini coefficients
    layers = [metrics["layer"] for metrics in metrics_by_layer]
    gini_coefficients = [metrics["gini_coefficient"] for metrics in metrics_by_layer]

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the data
    ax.plot(layers, gini_coefficients, marker="o", linewidth=2, color="#9C27B0")

    # Add titles and labels
    ax.set_title(f"Gini Coefficient per Layer ({model_name})", fontsize=14)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Gini Coefficient", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add exact values as text labels
    for i, gini in enumerate(gini_coefficients):
        ax.annotate(f"{gini:.4f}", (i, gini), textcoords="offset points",
                    xytext=(0, 5), ha='center')

    # Set y-axis limits (Gini coefficient is between 0 and 1)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    # Save the figure if run_folder is provided
    if run_folder:
        save_graph(fig, os.path.join(run_folder, "gini_coefficient_plot.png"))

    return fig

def get_top_activating_examples(autoencoder, test_feats, examples, feature_idx, top_k=5):
    """Get examples that most strongly activate a specific feature

    Args:
        autoencoder: Trained autoencoder model
        test_feats: Test features [batch_size, hidden_dim]
        examples: List of example dictionaries with 'text' field
        feature_idx: Index of the feature to analyze
        top_k: Number of top examples to return

    Returns:
        top_examples: List of dictionaries with example text and activation value
    """
    with torch.no_grad():
        # Forward pass through the autoencoder
        _, h_activated, _ = autoencoder(test_feats)
        
        # Get activations for the specific feature across all examples
        feature_activations = h_activated[:, feature_idx].cpu().numpy()
        
        # Get indices of examples with highest activations for this feature
        top_indices = np.argsort(feature_activations)[::-1][:top_k]
        
        # Create list of top examples with their activation values
        top_examples = []
        for idx in top_indices:
            if idx < len(examples):
                activation = feature_activations[idx]
                # Only include examples with positive activation
                if activation > 0:
                    top_examples.append({
                        'text': examples[idx]['text'],
                        'activation': float(activation)
                    })
        
        return top_examples

def get_feature_grid_data(autoencoder, test_feats, examples, layer_idx, num_features=25):
    """Get data for feature grid visualization of sparse autoencoder features

    Args:
        autoencoder: Trained autoencoder model
        test_feats: Test features [batch_size, hidden_dim]
        examples: List of example dictionaries with 'text' field
        layer_idx: Layer index
        num_features: Number of top features to display in the grid

    Returns:
        feature_data: List of dictionaries with feature information
    """
    with torch.no_grad():
        # Forward pass through the autoencoder
        _, h_activated, _ = autoencoder(test_feats)

        # Calculate mean activation per feature
        mean_activations = torch.mean(h_activated, dim=0).cpu().numpy()

        # Get indices of top activated features
        top_feature_indices = np.argsort(mean_activations)[::-1][:num_features]

        # Prepare feature data
        feature_data = []

        # For each feature, get the most activating examples
        for feature_idx in top_feature_indices:
            # Get examples that most activate this feature
            top_examples = get_top_activating_examples(
                autoencoder, test_feats, examples, feature_idx, top_k=3
            )

            # Store feature information
            feature_info = {
                'feature_idx': int(feature_idx),
                'mean_activation': float(mean_activations[feature_idx]),
                'top_examples': top_examples
            }

            feature_data.append(feature_info)

        return feature_data

# Keep the original plot_feature_grid for saving to disk but make it use the data function
def plot_feature_grid(autoencoder, test_feats, examples, layer_idx, num_features=25, run_folder=None):
    """Create a feature grid visualization for sparse autoencoder features

    Args:
        autoencoder: Trained autoencoder model
        test_feats: Test features [batch_size, hidden_dim]
        examples: List of example dictionaries with 'text' field
        layer_idx: Layer index
        num_features: Number of top features to display in the grid
        run_folder: Folder to save the visualization to

    Returns:
        fig: Matplotlib figure object with the feature grid
    """
    # Get feature data
    feature_data = get_feature_grid_data(autoencoder, test_feats, examples, layer_idx, num_features)

    # Create a grid figure
    fig_width = 15
    fig_height = max(15, num_features * 0.6)  # Adjust height based on number of features
    fig, axs = plt.subplots(len(feature_data), 1, figsize=(fig_width, fig_height))

    if len(feature_data) == 1:
        axs = [axs]  # Make axs always iterable

    # For each feature, show the most activating examples
    for i, feature_info in enumerate(feature_data):
        ax = axs[i]
        feature_idx = feature_info['feature_idx']
        mean_activation = feature_info['mean_activation']
        top_examples = feature_info['top_examples']

        # Display feature information
        feature_text = f"Feature {feature_idx} | Mean Activation: {mean_activation:.4f}\n"

        # Add top examples
        for j, example in enumerate(top_examples):
            # Truncate the text if it's too long
            text = example['text']
            if len(text) > 100:
                text = text[:97] + "..."

            feature_text += f"Example {j+1} (Act: {example['activation']:.4f}): {text}\n"

        # If no examples found
        if not top_examples:
            feature_text += "No examples with positive activation found for this feature."

        ax.text(0.01, 0.5, feature_text, wrap=True, fontsize=10, va='center')
        ax.axis('off')

    plt.tight_layout()

    # Save the figure if run_folder is provided
    if run_folder:
        layer_save_dir = os.path.join(run_folder, "layers", str(layer_idx))
        os.makedirs(layer_save_dir, exist_ok=True)
        save_graph(fig, os.path.join(layer_save_dir, "feature_grid.png"))

    return fig

def create_feature_activation_dataframe(autoencoder, test_feats, examples, feature_idx, top_k=10):
    """Create a DataFrame with examples that most strongly activate a specific feature
    
    Args:
        autoencoder: Trained autoencoder model
        test_feats: Test features [batch_size, hidden_dim]
        examples: List of example dictionaries
        feature_idx: Index of the feature to analyze
        top_k: Number of top examples to include
        
    Returns:
        df: DataFrame with top activating examples
    """
    top_examples = get_top_activating_examples(
        autoencoder, test_feats, examples, feature_idx, top_k=top_k
    )
    
    # Create DataFrame
    if top_examples:
        df = pd.DataFrame(top_examples)
        # Round activation values
        df['activation'] = df['activation'].map("{:.4f}".format)
        return df
    else:
        return pd.DataFrame(columns=['text', 'activation'])