"""
Sparse Autoencoder implementation for analyzing LLM internal representations.
Based on work from Anthropic research on sparse autoencoders in LLMs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union

class SparseAutoencoder(nn.Module):
    """
    A sparse autoencoder with an optional supervised component.
    
    This model encodes hidden states into a sparse representation and decodes them back.
    It can be trained in an unsupervised manner with L1 regularization or with supervision
    from dataset labels.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        feature_dim: int, 
        l1_coefficient: float = 0.01,
        tied_weights: bool = False,
        supervised: bool = False
    ):
        """
        Initialize the sparse autoencoder.
        
        Args:
            input_dim: Dimensionality of input vectors (hidden states)
            feature_dim: Dimensionality of the sparse feature space
            l1_coefficient: Coefficient for L1 regularization to encourage sparsity
            tied_weights: Whether to tie the encoder and decoder weights
            supervised: Whether to add a supervised component to the model
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.l1_coefficient = l1_coefficient
        self.tied_weights = tied_weights
        self.supervised = supervised
        
        # Encoder
        self.encoder = nn.Linear(input_dim, feature_dim, bias=True)
        
        # Decoder - shares weights with encoder if tied
        if tied_weights:
            self.decoder = nn.Linear(feature_dim, input_dim, bias=True)
            self.decoder.weight = nn.Parameter(self.encoder.weight.t())
        else:
            self.decoder = nn.Linear(feature_dim, input_dim, bias=True)
            
        # Initialize weights with small random values
        nn.init.kaiming_normal_(self.encoder.weight, nonlinearity='relu')
        if not tied_weights:
            nn.init.kaiming_normal_(self.decoder.weight, nonlinearity='relu')
        
        # Optional supervised component
        if supervised:
            self.classifier = nn.Linear(feature_dim, 1)
            nn.init.xavier_uniform_(self.classifier.weight)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input into sparse features"""
        return torch.relu(self.encoder(x))
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to input space"""
        return self.decoder(z)
    
    def classify(self, z: torch.Tensor) -> torch.Tensor:
        """Optional classification of sparse features"""
        if not self.supervised:
            raise ValueError("Model was not initialized with supervised=True")
        return torch.sigmoid(self.classifier(z))
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            If supervised:
                Tuple of (reconstruction, classification_logits)
            Otherwise:
                reconstruction
        """
        # Get sparse features
        z = self.encode(x)
        
        # Reconstruct input
        reconstruction = self.decode(z)
        
        if self.supervised:
            # Classification output
            classification = self.classify(z)
            return reconstruction, classification
        else:
            return reconstruction
    
    def get_l1_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate L1 loss on the activations to encourage sparsity"""
        z = self.encode(x)
        return self.l1_coefficient * torch.mean(torch.abs(z))
    
    def get_reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate MSE reconstruction loss"""
        z = self.encode(x)
        x_hat = self.decode(z)
        return F.mse_loss(x_hat, x)
    
    def get_classification_loss(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Calculate binary cross entropy loss for classification"""
        if not self.supervised:
            raise ValueError("Model was not initialized with supervised=True")
        
        z = self.encode(x)
        y_pred = self.classify(z)
        return F.binary_cross_entropy(y_pred, labels.float().view(-1, 1))
    
    def get_total_loss(
        self, 
        x: torch.Tensor, 
        labels: Optional[torch.Tensor] = None, 
        supervised_weight: float = 0.5
    ) -> torch.Tensor:
        """
        Calculate total loss including reconstruction, L1 regularization, 
        and optional supervised component.
        """
        reconstruction_loss = self.get_reconstruction_loss(x)
        l1_loss = self.get_l1_loss(x)
        
        if self.supervised and labels is not None:
            classification_loss = self.get_classification_loss(x, labels)
            return reconstruction_loss + l1_loss + (supervised_weight * classification_loss)
        else:
            return reconstruction_loss + l1_loss
    
    def get_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Get sparse features for input x"""
        with torch.no_grad():
            return self.encode(x)
    
    def get_sparsity_metrics(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Calculate sparsity metrics for the given input.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with various sparsity metrics
        """
        with torch.no_grad():
            z = self.encode(x)
            
            # Calculate L0 sparsity (percentage of exactly zero activations)
            l0_sparsity = (z == 0).float().mean().item()
            
            # Calculate L1 norm
            l1_norm = torch.abs(z).mean().item()
            
            # Calculate dead neurons (features that never activate)
            active_neurons = (z.sum(dim=0) > 0).float().mean().item()
            
            return {
                "l0_sparsity": l0_sparsity,
                "l1_norm": l1_norm,
                "active_neurons": active_neurons
            }


def train_sparse_autoencoder(
    model: SparseAutoencoder,
    train_data: torch.Tensor,
    train_labels: Optional[torch.Tensor] = None,
    val_data: Optional[torch.Tensor] = None,
    val_labels: Optional[torch.Tensor] = None,
    epochs: int = 50,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    supervised_weight: float = 0.5,
    progress_callback=None,
    device: torch.device = torch.device("cpu")
) -> Dict[str, List[float]]:
    """
    Train a sparse autoencoder model.
    
    Args:
        model: The SparseAutoencoder model to train
        train_data: Training data [n_samples, input_dim]
        train_labels: Optional training labels for supervised training
        val_data: Optional validation data
        val_labels: Optional validation labels
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        supervised_weight: Weight for supervised loss component
        progress_callback: Optional callback for reporting progress
        device: Device to train on
        
    Returns:
        Dictionary containing training history
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model = model.to(device)
    
    # Convert data to device
    train_data = train_data.to(device)
    if train_labels is not None:
        train_labels = train_labels.to(device)
    if val_data is not None:
        val_data = val_data.to(device)
    if val_labels is not None:
        val_labels = val_labels.to(device)
    
    # Training history
    history = {
        "reconstruction_loss": [],
        "l1_loss": [],
        "total_loss": []
    }
    
    if model.supervised:
        history["classification_loss"] = []
        history["accuracy"] = []
    
    if val_data is not None:
        history["val_reconstruction_loss"] = []
        history["val_total_loss"] = []
        if model.supervised:
            history["val_classification_loss"] = []
            history["val_accuracy"] = []
    
    # Training loop
    n_samples = train_data.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for epoch in range(epochs):
        model.train()
        
        # Shuffle data
        indices = torch.randperm(n_samples)
        
        epoch_reconstruction_loss = 0.0
        epoch_l1_loss = 0.0
        epoch_classification_loss = 0.0
        epoch_total_loss = 0.0
        correct = 0
        total = 0
        
        # Process mini-batches
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            x_batch = train_data[batch_indices]
            
            # Get labels if supervised
            if model.supervised and train_labels is not None:
                y_batch = train_labels[batch_indices]
            else:
                y_batch = None
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass and loss calculation
            reconstruction_loss = model.get_reconstruction_loss(x_batch)
            l1_loss = model.get_l1_loss(x_batch)
            
            if model.supervised and y_batch is not None:
                classification_loss = model.get_classification_loss(x_batch, y_batch)
                total_loss = reconstruction_loss + l1_loss + (supervised_weight * classification_loss)
                
                # Calculate accuracy
                with torch.no_grad():
                    predictions = model.classify(model.encode(x_batch))
                    predicted_labels = (predictions > 0.5).float()
                    correct += (predicted_labels == y_batch.view(-1, 1)).sum().item()
                    total += y_batch.size(0)
                
                epoch_classification_loss += classification_loss.item() * (end_idx - start_idx)
            else:
                total_loss = reconstruction_loss + l1_loss
            
            # Backpropagation
            total_loss.backward()
            optimizer.step()
            
            # Accumulate losses for epoch
            epoch_reconstruction_loss += reconstruction_loss.item() * (end_idx - start_idx)
            epoch_l1_loss += l1_loss.item() * (end_idx - start_idx)
            epoch_total_loss += total_loss.item() * (end_idx - start_idx)
        
        # Average losses
        epoch_reconstruction_loss /= n_samples
        epoch_l1_loss /= n_samples
        epoch_total_loss /= n_samples
        
        history["reconstruction_loss"].append(epoch_reconstruction_loss)
        history["l1_loss"].append(epoch_l1_loss)
        history["total_loss"].append(epoch_total_loss)
        
        if model.supervised:
            epoch_classification_loss /= n_samples
            history["classification_loss"].append(epoch_classification_loss)
            history["accuracy"].append(correct / total if total > 0 else 0.0)
        
        # Validation if provided
        if val_data is not None:
            model.eval()
            with torch.no_grad():
                val_reconstruction_loss = model.get_reconstruction_loss(val_data)
                
                if model.supervised and val_labels is not None:
                    val_classification_loss = model.get_classification_loss(val_data, val_labels)
                    val_total_loss = val_reconstruction_loss + model.get_l1_loss(val_data) + (supervised_weight * val_classification_loss)
                    
                    # Validation accuracy
                    val_predictions = model.classify(model.encode(val_data))
                    val_predicted_labels = (val_predictions > 0.5).float()
                    val_accuracy = (val_predicted_labels == val_labels.view(-1, 1)).float().mean().item()
                    
                    history["val_classification_loss"].append(val_classification_loss.item())
                    history["val_accuracy"].append(val_accuracy)
                else:
                    val_total_loss = val_reconstruction_loss + model.get_l1_loss(val_data)
                
                history["val_reconstruction_loss"].append(val_reconstruction_loss.item())
                history["val_total_loss"].append(val_total_loss.item())
        
        # Report progress
        if progress_callback is not None:
            progress = (epoch + 1) / epochs

            # Create status message
            status_msg = f"Epoch {epoch+1}/{epochs}"
            details = f"loss: {epoch_total_loss:.4f}, recon: {epoch_reconstruction_loss:.4f}"

            if model.supervised:
                details += f", cls_loss: {epoch_classification_loss:.4f}, acc: {(correct/total):.4f}"

            # Try to handle the callback with different parameter passing styles
            try:
                progress_callback(progress, status_msg, details)
            except TypeError:
                # If the standard call fails, try with the first parameter only
                try:
                    progress_callback(progress)
                except:
                    pass
    
    return history


def visualize_feature_grid(model: SparseAutoencoder, top_k: int = 100, n_cols: int = 10) -> plt.Figure:
    """
    Visualize the features learned by the sparse autoencoder decoder.
    
    Args:
        model: Trained SparseAutoencoder model
        top_k: Number of features to visualize
        n_cols: Number of columns in the grid
        
    Returns:
        Matplotlib figure with the feature grid
    """
    with torch.no_grad():
        # Get the decoder weights
        decoder_weights = model.decoder.weight.cpu().t().numpy()
        
        # Get feature norms
        feature_norms = np.linalg.norm(decoder_weights, axis=0)
        
        # Get indices of top-k features by norm
        if top_k > decoder_weights.shape[1]:
            top_k = decoder_weights.shape[1]
        
        top_indices = np.argsort(-feature_norms)[:top_k]
        
        # Create a grid of features
        n_rows = (top_k + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*1.5, n_rows*1.5))
        axes = axes.flatten()
        
        for i, idx in enumerate(top_indices):
            # Get the feature
            feature = decoder_weights[:, idx]
            
            # Reshape feature to make it visually interpretable if needed
            # This depends on your data - for now, just show as a heatmap
            
            # Normalize for visualization
            vmax = np.max(np.abs(feature))
            vmin = -vmax
            
            # Display the feature
            im = axes[i].imshow(feature.reshape(1, -1), cmap='RdBu_r', vmin=vmin, vmax=vmax)
            axes[i].set_title(f"F{idx}")
            axes[i].axis('off')
        
        # Hide any unused subplots
        for i in range(top_k, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        return fig


def visualize_feature_activations(
    model: SparseAutoencoder, 
    data: torch.Tensor,
    labels: Optional[torch.Tensor] = None
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Visualize feature activations from the sparse autoencoder.
    
    Args:
        model: Trained SparseAutoencoder model
        data: Input data to analyze
        labels: Optional labels to differentiate true/false examples
        
    Returns:
        Tuple of (sparsity_fig, activation_dist_fig)
    """
    with torch.no_grad():
        # Get activations
        activations = model.get_activations(data).cpu().numpy()
        
        # Sparsity plot: how often each feature activates
        fig_sparsity, ax_sparsity = plt.subplots(figsize=(12, 6))
        
        # Calculate activation frequency
        activation_freq = (activations > 0).mean(axis=0)
        
        # Sort frequencies
        sorted_indices = np.argsort(-activation_freq)
        sorted_freq = activation_freq[sorted_indices]
        
        ax_sparsity.bar(np.arange(len(sorted_freq)), sorted_freq)
        ax_sparsity.set_title("Feature Activation Frequency")
        ax_sparsity.set_xlabel("Feature Index (sorted)")
        ax_sparsity.set_ylabel("Activation Frequency")
        ax_sparsity.set_ylim(0, 1)
        ax_sparsity.grid(True, linestyle='--', alpha=0.7)
        
        # Activation distribution plot
        fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
        
        # Filter to only positive activations for histogram
        positive_acts = activations[activations > 0]
        
        if len(positive_acts) > 0:
            ax_dist.hist(positive_acts, bins=50, alpha=0.7)
            ax_dist.set_title("Distribution of Non-zero Activations")
            ax_dist.set_xlabel("Activation Value")
            ax_dist.set_ylabel("Frequency")
            ax_dist.grid(True, linestyle='--', alpha=0.7)
        else:
            ax_dist.text(0.5, 0.5, "No positive activations found", 
                         horizontalalignment='center', verticalalignment='center')
        
        return fig_sparsity, fig_dist


def visualize_feature_attribution(
    model: SparseAutoencoder,
    data: torch.Tensor,
    labels: torch.Tensor
) -> plt.Figure:
    """
    Visualize how features correlate with true/false values.
    
    Args:
        model: Trained SparseAutoencoder model
        data: Input data
        labels: Binary labels (1=true, 0=false)
        
    Returns:
        Matplotlib figure with feature attribution
    """
    with torch.no_grad():
        # Get activations
        activations = model.get_activations(data).cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        # Calculate correlation of each feature with labels
        correlations = []
        
        for i in range(activations.shape[1]):
            feature_activations = activations[:, i]
            corr = np.corrcoef(feature_activations, labels_np)[0, 1]
            correlations.append(corr)
        
        # Sort correlations
        sorted_indices = np.argsort(correlations)
        sorted_corrs = np.array(correlations)[sorted_indices]
        
        # Create a figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create color mask: positive correlations are green, negative are red
        colors = ['green' if c >= 0 else 'red' for c in sorted_corrs]
        
        ax.bar(np.arange(len(sorted_corrs)), sorted_corrs, color=colors)
        ax.set_title("Feature Correlation with Truth Values")
        ax.set_xlabel("Feature Index (sorted)")
        ax.set_ylabel("Correlation Coefficient")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='More active for TRUE'),
            Patch(facecolor='red', label='More active for FALSE')
        ]
        ax.legend(handles=legend_elements, loc='best')
        
        return fig


def visualize_neuron_feature_connections(
    model: SparseAutoencoder,
    top_k: int = 20
) -> plt.Figure:
    """
    Visualize connections between original neurons and learned features.
    
    Args:
        model: Trained SparseAutoencoder model
        top_k: Number of strongest connections to visualize
        
    Returns:
        Matplotlib figure with neuron-feature connections
    """
    with torch.no_grad():
        # Get encoder and decoder weights
        encoder_weights = model.encoder.weight.cpu().numpy()
        decoder_weights = model.decoder.weight.cpu().numpy()
        
        # Create a figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Calculate absolute importance of connections
        connection_strengths = np.abs(encoder_weights) @ np.abs(decoder_weights)
        
        # Show only the top connections for visibility
        vmax = np.percentile(connection_strengths, 95)
        
        im = ax.imshow(connection_strengths, cmap='viridis', vmax=vmax)
        ax.set_title("Neuron-Feature Connection Strengths")
        ax.set_xlabel("Original Neuron Index")
        ax.set_ylabel("Sparse Feature Index")
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Connection Strength")
        
        # Show gridlines
        ax.grid(False)
        
        # Limit the number of ticks for readability
        max_ticks = 20
        if encoder_weights.shape[0] > max_ticks:
            ax.set_yticks(np.linspace(0, encoder_weights.shape[0]-1, max_ticks, dtype=int))
        if encoder_weights.shape[1] > max_ticks:
            ax.set_xticks(np.linspace(0, encoder_weights.shape[1]-1, max_ticks, dtype=int))
        
        return fig