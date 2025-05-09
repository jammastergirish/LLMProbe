import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Optional, Tuple, List, Dict, Any, Union


class SparseAutoencoder(nn.Module):
    """
    A sparse autoencoder implementation that can be trained on model activations.
    
    This autoencoder includes L1 regularization to enforce sparsity in the latent
    representation.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        latent_dim: int,
        tied_weights: bool = False,
        bias: bool = True,
        activation: nn.Module = nn.ReLU(),
        device: torch.device = None
    ):
        """
        Initialize the sparse autoencoder.
        
        Args:
            input_dim: Dimension of the input data (hidden state size)
            latent_dim: Dimension of the latent space representation
            tied_weights: Whether to tie the encoder and decoder weights
            bias: Whether to include bias terms
            activation: Activation function to use (default: ReLU)
            device: Device to use for computation (default: None)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.tied_weights = tied_weights
        self.activation = activation
        
        # If no device is provided, use CUDA if available, else CPU
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Encoder layer
        self.encoder = nn.Linear(input_dim, latent_dim, bias=bias)
        
        # Decoder layer
        if tied_weights:
            # We'll use the transpose of the encoder weights during forward pass
            self.decoder = None
        else:
            self.decoder = nn.Linear(latent_dim, input_dim, bias=bias)
            
        # Move model to the specified device
        self.to(self.device)
        
        # Initialize weights using Xavier (Glorot) initialization
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights for better training convergence."""
        # Xavier (Glorot) initialization for encoder
        nn.init.xavier_uniform_(self.encoder.weight)
        
        if self.encoder.bias is not None:
            nn.init.zeros_(self.encoder.bias)
            
        # Initialize decoder if not using tied weights
        if not self.tied_weights and self.decoder is not None:
            nn.init.xavier_uniform_(self.decoder.weight)
            
            if self.decoder.bias is not None:
                nn.init.zeros_(self.decoder.bias)
                
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input to the latent space.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Latent representation of shape [batch_size, latent_dim]
        """
        latent = self.encoder(x)
        return self.activation(latent)
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode the latent representation back to the input space.
        
        Args:
            z: Latent tensor of shape [batch_size, latent_dim]
            
        Returns:
            Reconstructed tensor of shape [batch_size, input_dim]
        """
        if self.tied_weights:
            # Use the transpose of the encoder weight matrix
            return F.linear(z, self.encoder.weight.t(), self.encoder.bias)
        else:
            return self.decoder(z)
            
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Tuple of (reconstructed_input, latent_representation)
        """
        # Encode
        z = self.encode(x)
        
        # Decode
        x_reconstructed = self.decode(z)
        
        return x_reconstructed, z
        
    @staticmethod
    def l1_loss(latent: torch.Tensor, l1_weight: float = 1.0) -> torch.Tensor:
        """
        Compute L1 regularization loss for sparsity.
        
        Args:
            latent: Latent representation tensor
            l1_weight: Weight of the L1 regularization (default: 1.0)
            
        Returns:
            L1 loss (weighted sum of absolute values)
        """
        return l1_weight * torch.mean(torch.abs(latent))
    
    @staticmethod
    def reconstruction_loss(x: torch.Tensor, x_reconstructed: torch.Tensor) -> torch.Tensor:
        """
        Compute Mean Squared Error (MSE) reconstruction loss.
        
        Args:
            x: Original input tensor
            x_reconstructed: Reconstructed tensor from the decoder
            
        Returns:
            MSE loss
        """
        return F.mse_loss(x_reconstructed, x, reduction='mean')


class SupervisedSparseAutoencoder(SparseAutoencoder):
    """
    A supervised sparse autoencoder variant that can leverage labels during training.
    
    This extends the base SparseAutoencoder by adding an additional supervised component
    that can be used to make the latent features more aligned with specific tasks.
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_classes: int = 2,
        tied_weights: bool = False,
        bias: bool = True,
        activation: nn.Module = nn.ReLU(),
        device: torch.device = None
    ):
        """
        Initialize the supervised sparse autoencoder.
        
        Args:
            input_dim: Dimension of the input data (hidden state size)
            latent_dim: Dimension of the latent space representation
            num_classes: Number of classes for classification task
            tied_weights: Whether to tie the encoder and decoder weights
            bias: Whether to include bias terms
            activation: Activation function to use (default: ReLU)
            device: Device to use for computation (default: None)
        """
        super().__init__(input_dim, latent_dim, tied_weights, bias, activation, device)
        
        # Add a classification head from the latent space
        self.classifier = nn.Linear(latent_dim, num_classes)
        
        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)
            
        # Move to device
        self.classifier.to(self.device)
        
    def classify(self, z: torch.Tensor) -> torch.Tensor:
        """
        Apply the classification head to the latent representation.
        
        Args:
            z: Latent tensor of shape [batch_size, latent_dim]
            
        Returns:
            Classification logits of shape [batch_size, num_classes]
        """
        return self.classifier(z)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the supervised autoencoder.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Tuple of (reconstructed_input, latent_representation, classification_logits)
        """
        # Get reconstruction and latent from parent class
        x_reconstructed, z = super().forward(x)
        
        # Apply classification head
        logits = self.classify(z)
        
        return x_reconstructed, z, logits
    
    @staticmethod
    def classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute classification loss using cross-entropy.
        
        Args:
            logits: Output from classifier of shape [batch_size, num_classes]
            labels: Ground truth labels of shape [batch_size]
            
        Returns:
            Cross entropy loss
        """
        return F.cross_entropy(logits, labels, reduction='mean')


def train_sparse_autoencoder(
    model: Union[SparseAutoencoder, SupervisedSparseAutoencoder],
    train_data: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    val_data: Optional[torch.Tensor] = None,
    val_labels: Optional[torch.Tensor] = None,
    batch_size: int = 64,
    lr: float = 0.001,
    l1_weight: float = 0.1,
    num_epochs: int = 50,
    supervised: bool = False,
    lambda_supervised: float = 1.0,
    patience: int = 5,
    progress_callback: Optional[callable] = None
) -> Dict[str, List[float]]:
    """
    Train a sparse autoencoder on the provided data.
    
    Args:
        model: The autoencoder model to train
        train_data: Training data tensor of shape [n_samples, input_dim]
        labels: Optional labels for supervised training
        val_data: Optional validation data for early stopping
        val_labels: Optional validation labels if supervised
        batch_size: Training batch size
        lr: Learning rate
        l1_weight: Weight for L1 regularization (sparsity)
        num_epochs: Maximum number of training epochs
        supervised: Whether to use supervised training
        lambda_supervised: Weight for supervised loss component
        patience: Number of epochs to wait for improvement before early stopping
        progress_callback: Optional callback function for showing training progress
        
    Returns:
        Dictionary containing training history
    """
    # Ensure data is on the correct device
    train_data = train_data.to(model.device)
    if labels is not None:
        labels = labels.to(model.device)
    
    if val_data is not None:
        val_data = val_data.to(model.device)
        if val_labels is not None:
            val_labels = val_labels.to(model.device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training history
    history = {
        'train_loss': [],
        'train_recon_loss': [],
        'train_l1_loss': [],
        'val_loss': [],
    }
    
    if supervised:
        history['train_class_loss'] = []
        history['train_class_acc'] = []
        if val_labels is not None:
            history['val_class_loss'] = []
            history['val_class_acc'] = []
    
    best_val_loss = float('inf')
    counter = 0  # For early stopping
    
    # Create data loader
    dataset = torch.utils.data.TensorDataset(
        train_data, 
        labels if labels is not None else torch.zeros(len(train_data))
    )
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True
    )
    
    # Train the model
    for epoch in range(num_epochs):
        model.train()
        train_loss_epoch = 0
        train_recon_loss_epoch = 0
        train_l1_loss_epoch = 0
        train_class_loss_epoch = 0
        train_correct = 0
        total_samples = 0
        
        for i, (batch_x, batch_y) in enumerate(loader):
            # Training progress
            if progress_callback:
                progress = (epoch / num_epochs) + ((i / len(loader)) / num_epochs)
                progress_callback(
                    progress, 
                    f"Training epoch {epoch+1}/{num_epochs}, batch {i+1}/{len(loader)}",
                    f"Loss: {train_loss_epoch/(i+1):.4f}"
                )
            
            optimizer.zero_grad()
            
            # Forward pass
            if supervised and isinstance(model, SupervisedSparseAutoencoder):
                x_reconstructed, z, logits = model(batch_x)
                recon_loss = model.reconstruction_loss(batch_x, x_reconstructed)
                l1_loss = model.l1_loss(z, l1_weight)
                
                # Only add classification loss if we have labels
                if labels is not None:
                    class_loss = model.classification_loss(logits, batch_y)
                    total_loss = recon_loss + l1_loss + lambda_supervised * class_loss
                    train_class_loss_epoch += class_loss.item() * batch_x.size(0)
                    
                    # Calculate accuracy
                    _, predicted = torch.max(logits, 1)
                    train_correct += (predicted == batch_y).sum().item()
                else:
                    total_loss = recon_loss + l1_loss
            else:
                x_reconstructed, z = model(batch_x)
                recon_loss = model.reconstruction_loss(batch_x, x_reconstructed)
                l1_loss = model.l1_loss(z, l1_weight)
                total_loss = recon_loss + l1_loss
            
            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()
            
            # Update accumulators
            train_loss_epoch += total_loss.item() * batch_x.size(0)
            train_recon_loss_epoch += recon_loss.item() * batch_x.size(0)
            train_l1_loss_epoch += l1_loss.item() * batch_x.size(0)
            total_samples += batch_x.size(0)
        
        # Calculate epoch losses
        train_loss = train_loss_epoch / total_samples
        train_recon_loss = train_recon_loss_epoch / total_samples
        train_l1_loss = train_l1_loss_epoch / total_samples
        
        # Save to history
        history['train_loss'].append(train_loss)
        history['train_recon_loss'].append(train_recon_loss)
        history['train_l1_loss'].append(train_l1_loss)
        
        if supervised and labels is not None:
            train_class_loss = train_class_loss_epoch / total_samples
            train_class_acc = train_correct / total_samples
            history['train_class_loss'].append(train_class_loss)
            history['train_class_acc'].append(train_class_acc)
        
        # Validation
        if val_data is not None:
            model.eval()
            with torch.no_grad():
                if supervised and isinstance(model, SupervisedSparseAutoencoder) and val_labels is not None:
                    x_reconstructed, z, logits = model(val_data)
                    val_recon_loss = model.reconstruction_loss(val_data, x_reconstructed)
                    val_l1_loss = model.l1_loss(z, l1_weight)
                    val_class_loss = model.classification_loss(logits, val_labels)
                    val_loss = val_recon_loss + val_l1_loss + lambda_supervised * val_class_loss
                    
                    # Calculate validation accuracy
                    _, predicted = torch.max(logits, 1)
                    val_correct = (predicted == val_labels).sum().item()
                    val_acc = val_correct / len(val_labels)
                    
                    history['val_class_loss'].append(val_class_loss.item())
                    history['val_class_acc'].append(val_acc)
                else:
                    x_reconstructed, z = model(val_data)
                    val_recon_loss = model.reconstruction_loss(val_data, x_reconstructed)
                    val_l1_loss = model.l1_loss(z, l1_weight)
                    val_loss = val_recon_loss + val_l1_loss
                
                history['val_loss'].append(val_loss.item())
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss.item()
                    counter = 0
                else:
                    counter += 1
                    
                if counter >= patience:
                    if progress_callback:
                        progress_callback(
                            1.0,
                            f"Early stopping at epoch {epoch+1}",
                            f"No improvement for {patience} epochs"
                        )
                    break
        
        # Final epoch progress
        if progress_callback:
            val_loss_str = f", Val Loss: {val_loss.item():.4f}" if val_data is not None else ""
            supervised_str = ""
            if supervised and labels is not None:
                supervised_str = f", Class Acc: {train_class_acc:.4f}"
                if val_data is not None and val_labels is not None:
                    supervised_str += f", Val Acc: {val_acc:.4f}"
                    
            progress_callback(
                (epoch + 1) / num_epochs,
                f"Completed epoch {epoch+1}/{num_epochs}",
                f"Loss: {train_loss:.4f}, Recon: {train_recon_loss:.4f}, L1: {train_l1_loss:.4f}{supervised_str}{val_loss_str}"
            )
    
    return history


def plot_training_history(history: Dict[str, List[float]]) -> Tuple[plt.Figure, ...]:
    """
    Plot the training history of the sparse autoencoder.
    
    Args:
        history: Dictionary containing training metrics
        
    Returns:
        Tuple of matplotlib figures
    """
    figures = []
    
    # Loss plot
    fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
    ax_loss.plot(history['train_loss'], label='Training Loss')
    ax_loss.plot(history['train_recon_loss'], label='Reconstruction Loss')
    ax_loss.plot(history['train_l1_loss'], label='L1 Loss')
    
    if 'val_loss' in history and history['val_loss']:
        ax_loss.plot(history['val_loss'], label='Validation Loss')
        
    ax_loss.set_xlabel('Epochs')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Training and Validation Loss')
    ax_loss.legend()
    ax_loss.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    figures.append(fig_loss)
    
    # Classification metrics if supervised
    if 'train_class_acc' in history and history['train_class_acc']:
        fig_acc, ax_acc = plt.subplots(figsize=(10, 6))
        ax_acc.plot(history['train_class_acc'], label='Training Accuracy')
        
        if 'val_class_acc' in history and history['val_class_acc']:
            ax_acc.plot(history['val_class_acc'], label='Validation Accuracy')
            
        ax_acc.set_xlabel('Epochs')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.set_title('Classification Accuracy')
        ax_acc.legend()
        ax_acc.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        figures.append(fig_acc)
        
        fig_class_loss, ax_class_loss = plt.subplots(figsize=(10, 6))
        ax_class_loss.plot(history['train_class_loss'], label='Training Classification Loss')
        
        if 'val_class_loss' in history and history['val_class_loss']:
            ax_class_loss.plot(history['val_class_loss'], label='Validation Classification Loss')
            
        ax_class_loss.set_xlabel('Epochs')
        ax_class_loss.set_ylabel('Loss')
        ax_class_loss.set_title('Classification Loss')
        ax_class_loss.legend()
        ax_class_loss.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        figures.append(fig_class_loss)
    
    return tuple(figures)


def visualize_feature_activations(
    model: Union[SparseAutoencoder, SupervisedSparseAutoencoder],
    data: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    top_k: int = 10
) -> Tuple[plt.Figure, ...]:
    """
    Visualize the activation patterns of features in the sparse autoencoder.
    
    Args:
        model: Trained sparse autoencoder model
        data: Input data to visualize activations for
        labels: Optional labels for color-coding (binary labels expected)
        top_k: Number of top activating neurons to highlight
        
    Returns:
        Tuple of matplotlib figures
    """
    model.eval()
    figures = []
    
    # Ensure data is on the correct device
    data = data.to(model.device)
    
    with torch.no_grad():
        # Get latent activations
        if isinstance(model, SupervisedSparseAutoencoder):
            _, latent, _ = model(data)
        else:
            _, latent = model(data)
    
    latent_np = latent.cpu().numpy()
    
    # Compute average activations
    avg_activations = np.mean(latent_np, axis=0)
    
    # Plot average activation per feature
    fig_activations, ax_activations = plt.subplots(figsize=(12, 6))
    ax_activations.bar(range(len(avg_activations)), avg_activations)
    ax_activations.set_xlabel('Feature Index')
    ax_activations.set_ylabel('Average Activation')
    ax_activations.set_title('Average Activation Per Latent Feature')
    ax_activations.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    figures.append(fig_activations)
    
    # Compute sparsity metrics
    activation_frequency = np.mean(latent_np > 0, axis=0)
    
    # Plot activation frequency
    fig_frequency, ax_frequency = plt.subplots(figsize=(12, 6))
    ax_frequency.bar(range(len(activation_frequency)), activation_frequency)
    ax_frequency.set_xlabel('Feature Index')
    ax_frequency.set_ylabel('Activation Frequency')
    ax_frequency.set_title('Frequency of Feature Activation (Proportion of Samples Where Feature is Active)')
    ax_frequency.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    figures.append(fig_frequency)
    
    # If we have labels, plot feature importance for binary classification
    if labels is not None:
        labels_np = labels.cpu().numpy()
        
        # Compute feature importance for each class
        class_0_mean = np.mean(latent_np[labels_np == 0], axis=0)
        class_1_mean = np.mean(latent_np[labels_np == 1], axis=0)
        
        # Compute class difference
        class_diff = class_1_mean - class_0_mean
        
        # Plot class difference
        fig_class_diff, ax_class_diff = plt.subplots(figsize=(12, 6))
        ax_class_diff.bar(range(len(class_diff)), class_diff)
        ax_class_diff.set_xlabel('Feature Index')
        ax_class_diff.set_ylabel('Activation Difference (Class 1 - Class 0)')
        ax_class_diff.set_title('Feature Importance for Class Discrimination')
        ax_class_diff.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        figures.append(fig_class_diff)
        
        # Find top-k most discriminative features
        top_k_features = np.argsort(np.abs(class_diff))[-top_k:]
        
        # Plot activations of top-k features for each sample
        fig_top_k, ax_top_k = plt.subplots(figsize=(14, 8))
        
        # For each top feature
        for i, feat_idx in enumerate(top_k_features):
            # Class 0 samples
            class_0_activations = latent_np[labels_np == 0, feat_idx]
            ax_top_k.scatter(
                [i] * len(class_0_activations),
                class_0_activations,
                color='blue',
                alpha=0.5,
                label='Class 0' if i == 0 else None
            )
            
            # Class 1 samples
            class_1_activations = latent_np[labels_np == 1, feat_idx]
            ax_top_k.scatter(
                [i] * len(class_1_activations),
                class_1_activations,
                color='red',
                alpha=0.5,
                label='Class 1' if i == 0 else None
            )
            
            # Mean activations
            ax_top_k.scatter(i, np.mean(class_0_activations), color='darkblue', marker='x', s=100)
            ax_top_k.scatter(i, np.mean(class_1_activations), color='darkred', marker='x', s=100)
        
        ax_top_k.set_xticks(range(len(top_k_features)))
        ax_top_k.set_xticklabels([f'Feature {idx}' for idx in top_k_features])
        ax_top_k.set_ylabel('Activation Value')
        ax_top_k.set_title(f'Top {top_k} Most Discriminative Features')
        ax_top_k.legend()
        ax_top_k.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        figures.append(fig_top_k)
    
    # PCA on latent space for overall distribution
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_np)
    
    fig_pca, ax_pca = plt.subplots(figsize=(10, 8))
    
    if labels is not None:
        # Color points by label
        class_0_mask = labels_np == 0
        class_1_mask = labels_np == 1
        
        ax_pca.scatter(
            latent_2d[class_0_mask, 0],
            latent_2d[class_0_mask, 1],
            color='blue',
            alpha=0.6,
            label='Class 0'
        )
        
        ax_pca.scatter(
            latent_2d[class_1_mask, 0],
            latent_2d[class_1_mask, 1],
            color='red',
            alpha=0.6,
            label='Class 1'
        )
        
        ax_pca.legend()
    else:
        # Single color for all points
        ax_pca.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.6)
    
    ax_pca.set_xlabel('PCA Component 1')
    ax_pca.set_ylabel('PCA Component 2')
    ax_pca.set_title('PCA of Latent Space Representations')
    ax_pca.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    figures.append(fig_pca)
    
    return tuple(figures)


def visualize_feature_weights(
    model: Union[SparseAutoencoder, SupervisedSparseAutoencoder],
    input_dim: int,
    feature_indices: Optional[List[int]] = None,
    n_features_to_show: int = 5
) -> plt.Figure:
    """
    Visualize the weights connecting input dimensions to latent features.
    
    Args:
        model: Trained sparse autoencoder model
        input_dim: Dimension of the input data (for reshaping)
        feature_indices: Specific latent feature indices to visualize
        n_features_to_show: Number of top features to show if feature_indices is None
        
    Returns:
        Matplotlib figure showing feature weight patterns
    """
    model.eval()
    
    # Get encoder weights
    weights = model.encoder.weight.cpu().detach().numpy()
    
    # If feature indices not provided, select the n most active features
    if feature_indices is None:
        # Use average activation as a proxy for importance
        feature_importance = np.sum(np.abs(weights), axis=1)
        feature_indices = np.argsort(feature_importance)[-n_features_to_show:]
    
    # Number of features to visualize
    n_features = len(feature_indices)
    
    # Create the plot
    fig, axes = plt.subplots(1, n_features, figsize=(15, 3))
    
    # Handle case where only one feature is selected
    if n_features == 1:
        axes = [axes]
    
    for i, feature_idx in enumerate(feature_indices):
        feature_weights = weights[feature_idx]
        
        # Display the weights as a bar chart
        axes[i].bar(range(len(feature_weights)), feature_weights)
        axes[i].set_title(f'Feature {feature_idx}')
        axes[i].set_xlabel('Input Dimension')
        
        # Only add y-label for the first subplot
        if i == 0:
            axes[i].set_ylabel('Weight')
            
        # Add horizontal line at zero
        axes[i].axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        # Add grid
        axes[i].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig


def compare_layers_sparsity(
    autoencoders: List[Union[SparseAutoencoder, SupervisedSparseAutoencoder]],
    hidden_states: torch.Tensor,
    labels: Optional[torch.Tensor] = None
) -> Tuple[plt.Figure, ...]:
    """
    Compare sparsity and feature distribution across all layers' autoencoders
    
    Args:
        autoencoders: List of trained sparse autoencoder models (one per layer)
        hidden_states: Hidden states tensor of shape [batch_size, n_layers, hidden_dim]
        labels: Optional labels tensor for supervised metrics
        
    Returns:
        Tuple of matplotlib figures showing layer comparisons
    """
    figures = []
    num_layers = len(autoencoders)
    
    # Prepare data for metrics
    avg_activations_per_layer = []
    sparsity_per_layer = []
    reconstruction_losses = []
    classification_accuracy = []
    
    # Compute metrics for each layer
    for layer_idx, model in enumerate(autoencoders):
        model.eval()
        
        # Get layer hidden states
        layer_states = hidden_states[:, layer_idx, :]
        
        # Pass through autoencoder
        with torch.no_grad():
            if isinstance(model, SupervisedSparseAutoencoder) and labels is not None:
                x_reconstructed, latent, logits = model(layer_states)
                
                # Calculate classification accuracy if supervised
                _, preds = torch.max(logits, 1)
                correct = (preds == labels).sum().item()
                accuracy = correct / labels.size(0)
                classification_accuracy.append(accuracy)
            else:
                x_reconstructed, latent = model(layer_states)
                if labels is not None:
                    classification_accuracy.append(0.0)  # Placeholder for unsupervised models
            
            # Calculate reconstruction loss
            recon_loss = model.reconstruction_loss(layer_states, x_reconstructed).item()
            reconstruction_losses.append(recon_loss)
            
            # Calculate average activation
            avg_act = torch.mean(latent).item()
            avg_activations_per_layer.append(avg_act)
            
            # Calculate sparsity (percentage of zeros)
            sparsity = torch.mean((latent == 0.0).float()).item()
            sparsity_per_layer.append(sparsity)
    
    # Create figure for reconstruction loss
    fig_recon, ax_recon = plt.subplots(figsize=(10, 6))
    ax_recon.plot(range(num_layers), reconstruction_losses, marker='o', linestyle='-')
    ax_recon.set_xlabel('Layer')
    ax_recon.set_ylabel('Reconstruction Loss')
    ax_recon.set_title('Reconstruction Loss by Layer')
    ax_recon.grid(True, linestyle='--', alpha=0.7)
    
    # Add value annotations
    for i, loss in enumerate(reconstruction_losses):
        ax_recon.annotate(f"{loss:.4f}", (i, loss), 
                         textcoords="offset points", xytext=(0, 5), 
                         ha='center')
    
    plt.tight_layout()
    figures.append(fig_recon)
    
    # Create figure for sparsity
    fig_sparsity, ax_sparsity = plt.subplots(figsize=(10, 6))
    ax_sparsity.plot(range(num_layers), sparsity_per_layer, marker='o', linestyle='-')
    ax_sparsity.set_xlabel('Layer')
    ax_sparsity.set_ylabel('Sparsity (% zero activations)')
    ax_sparsity.set_title('Feature Sparsity by Layer')
    ax_sparsity.grid(True, linestyle='--', alpha=0.7)
    
    # Add value annotations
    for i, sparsity in enumerate(sparsity_per_layer):
        ax_sparsity.annotate(f"{sparsity:.4f}", (i, sparsity), 
                            textcoords="offset points", xytext=(0, 5), 
                            ha='center')
    
    plt.tight_layout()
    figures.append(fig_sparsity)
    
    # Create figure for average activation
    fig_act, ax_act = plt.subplots(figsize=(10, 6))
    ax_act.plot(range(num_layers), avg_activations_per_layer, marker='o', linestyle='-')
    ax_act.set_xlabel('Layer')
    ax_act.set_ylabel('Average Activation')
    ax_act.set_title('Feature Activation by Layer')
    ax_act.grid(True, linestyle='--', alpha=0.7)
    
    # Add value annotations
    for i, act in enumerate(avg_activations_per_layer):
        ax_act.annotate(f"{act:.4f}", (i, act), 
                       textcoords="offset points", xytext=(0, 5), 
                       ha='center')
    
    plt.tight_layout()
    figures.append(fig_act)
    
    # If supervised, create figure for classification accuracy
    if labels is not None and any(isinstance(m, SupervisedSparseAutoencoder) for m in autoencoders):
        fig_acc, ax_acc = plt.subplots(figsize=(10, 6))
        ax_acc.plot(range(num_layers), classification_accuracy, marker='o', linestyle='-')
        ax_acc.set_xlabel('Layer')
        ax_acc.set_ylabel('Classification Accuracy')
        ax_acc.set_title('Truth Classification Accuracy by Layer')
        ax_acc.grid(True, linestyle='--', alpha=0.7)
        
        # Add value annotations
        for i, acc in enumerate(classification_accuracy):
            ax_acc.annotate(f"{acc:.4f}", (i, acc), 
                           textcoords="offset points", xytext=(0, 5), 
                           ha='center')
        
        plt.tight_layout()
        figures.append(fig_acc)
    
    return tuple(figures)


def compare_feature_correlation(
    autoencoders: List[Union[SparseAutoencoder, SupervisedSparseAutoencoder]],
    hidden_states: torch.Tensor
) -> plt.Figure:
    """
    Compare the correlation between features across layers
    
    Args:
        autoencoders: List of trained sparse autoencoder models (one per layer)
        hidden_states: Hidden states tensor of shape [batch_size, n_layers, hidden_dim]
        
    Returns:
        Matplotlib figure showing feature correlation matrix
    """
    num_layers = len(autoencoders)
    
    # Extract latent representations for each layer
    all_latents = []
    
    for layer_idx, model in enumerate(autoencoders):
        model.eval()
        
        # Get layer hidden states
        layer_states = hidden_states[:, layer_idx, :]
        
        # Get latent representations
        with torch.no_grad():
            if isinstance(model, SupervisedSparseAutoencoder):
                _, latent, _ = model(layer_states)
            else:
                _, latent = model(layer_states)
            
            # Flatten to 1D vector per sample (mean across feature dimension)
            feature_means = torch.mean(latent, dim=1).cpu().numpy()
            all_latents.append(feature_means)
    
    # Calculate correlation matrix between layers
    correlation_matrix = np.zeros((num_layers, num_layers))
    
    for i in range(num_layers):
        for j in range(num_layers):
            # Calculate correlation between layer i and layer j
            if i == j:
                correlation_matrix[i, j] = 1.0
            else:
                correlation = np.corrcoef(all_latents[i], all_latents[j])[0, 1]
                correlation_matrix[i, j] = correlation
    
    # Create heatmap figure
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Correlation", rotation=-90, va="bottom")
    
    # Add labels
    ax.set_xticks(np.arange(num_layers))
    ax.set_yticks(np.arange(num_layers))
    ax.set_xticklabels([f"Layer {i}" for i in range(num_layers)])
    ax.set_yticklabels([f"Layer {i}" for i in range(num_layers)])
    
    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add title and labels
    ax.set_title("Correlation of Autoencoder Features Between Layers")
    
    # Loop over data dimensions and create text annotations
    for i in range(num_layers):
        for j in range(num_layers):
            text = ax.text(j, i, f"{correlation_matrix[i, j]:.2f}",
                          ha="center", va="center", 
                          color="black" if abs(correlation_matrix[i, j]) < 0.5 else "white")
    
    plt.tight_layout()
    return fig