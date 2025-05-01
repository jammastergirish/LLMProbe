import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        z = torch.relu(self.encoder(x))
        x_recon = self.decoder(z)
        return x_recon, z
        
    def calculate_sparsity(self, z, threshold=0.01):
        """Calculate percentage of activations below threshold"""
        return torch.mean((z < threshold).float()).item() * 100
        
    def classify(self, z):
        """Dummy classify function for compatibility"""
        # This is just a placeholder for compatibility
        return torch.zeros(z.shape[0], 1, device=z.device)

def train_unsupervised_autoencoder(hidden_states, hidden_dim=512, epochs=300, lr=1e-3, sparsity_weight=1e-5, 
                                  progress_callback=None):
    """Train a sparse autoencoder without using label information"""
    input_dim = hidden_states.shape[1]
    model = SparseAutoencoder(input_dim, hidden_dim).to(hidden_states.device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    total_loss_history = []
    mse_loss_history = []
    l1_loss_history = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        x_recon, z = model(hidden_states)
        mse_loss = criterion(x_recon, hidden_states)
        l1_loss = sparsity_weight * torch.mean(torch.abs(z))
        loss = mse_loss + l1_loss
        loss.backward()
        optimizer.step()
        
        total_loss_history.append(loss.item())
        mse_loss_history.append(mse_loss.item())
        l1_loss_history.append(l1_loss.item())
        
        # Update progress if callback is provided
        if progress_callback is not None:
            progress = (epoch + 1) / epochs
            progress_callback(progress, 
                             f"Epoch {epoch+1}/{epochs}", 
                             f"MSE: {mse_loss.item():.6f} | L1: {l1_loss.item():.6f} | Total: {loss.item():.6f}")
            
    return model, {
        'total_losses': total_loss_history,
        'recon_losses': mse_loss_history,
        'l1_losses': l1_loss_history,
        'sparsity_losses': l1_loss_history  # For compatibility with existing code
    }

def train_supervised_autoencoder(hidden_states, labels, hidden_dim=512, epochs=300, lr=1e-3, 
                               sparsity_weight=1e-5, supervision_weight=0.1, progress_callback=None):
    """Train a sparse autoencoder with label supervision"""
    input_dim = hidden_states.shape[1]
    model = SparseAutoencoder(input_dim, hidden_dim).to(hidden_states.device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    reconstruction_criterion = nn.MSELoss()
    supervision_criterion = nn.BCEWithLogitsLoss()
    
    # Create supervision layer to predict labels from latent space
    supervision_layer = nn.Linear(hidden_dim, 1).to(hidden_states.device)
    
    # Include supervision layer in optimization
    optimizer = optim.Adam(list(model.parameters()) + list(supervision_layer.parameters()), lr=lr)
    
    total_loss_history = []
    mse_loss_history = []
    l1_loss_history = []
    supervision_loss_history = []
    
    labels_float = labels.float().unsqueeze(1)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass through autoencoder
        x_recon, z = model(hidden_states)
        
        # Reconstruction loss
        mse_loss = reconstruction_criterion(x_recon, hidden_states)
        
        # Sparsity loss
        l1_loss = sparsity_weight * torch.mean(torch.abs(z))
        
        # Supervised loss component - predict labels from latent representation
        label_predictions = supervision_layer(z)
        supervision_loss = supervision_criterion(label_predictions, labels_float)
        
        # Total loss
        loss = mse_loss + l1_loss + supervision_weight * supervision_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss_history.append(loss.item())
        mse_loss_history.append(mse_loss.item())
        l1_loss_history.append(l1_loss.item())
        supervision_loss_history.append(supervision_loss.item())
        
        # Update progress if callback is provided
        if progress_callback is not None:
            progress = (epoch + 1) / epochs
            progress_callback(progress, 
                             f"Epoch {epoch+1}/{epochs}", 
                             f"MSE: {mse_loss.item():.6f} | L1: {l1_loss.item():.6f} | Supervision: {supervision_loss.item():.6f} | Total: {loss.item():.6f}")
            
    return model, {
        'total_losses': total_loss_history,
        'recon_losses': mse_loss_history,
        'l1_losses': l1_loss_history,
        'sparsity_losses': l1_loss_history,  # For compatibility with existing code
        'classification_losses': supervision_loss_history  # For compatibility with existing code
    }

def analyze_latents(autoencoder, hidden_states, labels):
    """Analyze which latent features correlate most with labels"""
    with torch.no_grad():
        _, latents = autoencoder(hidden_states)
    
    labels_float = labels.float().unsqueeze(1)
    correlations = torch.abs(torch.corrcoef(torch.cat([latents.T, labels_float.T]))[-1, :-1])
    top_indices = torch.topk(correlations, k=10).indices
    return top_indices, correlations

def get_latent_features(autoencoder, hidden_states):
    """Extract latent features from the autoencoder"""
    with torch.no_grad():
        _, latents = autoencoder(hidden_states)
    return latents