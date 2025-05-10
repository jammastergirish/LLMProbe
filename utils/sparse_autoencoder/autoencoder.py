import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=0, tied_weights=True):
        """
        Basic sparse autoencoder with optional tied weights

        Args:
            input_dim (int): Dimension of input features
            bottleneck_dim (int): Dimension of bottleneck layer (latent space)
                                If 0, uses same dimension as input
                                          If > input_dim, autoencoder is overcomplete
                                          If < input_dim, autoencoder is undercomplete
            tied_weights (bool): If True, decoder weights are tied to encoder weights
        """
        super().__init__()

        # Set dimensions
        self.input_dim = input_dim
        self.bottleneck_dim = input_dim if bottleneck_dim == 0 else bottleneck_dim
        self.tied_weights = tied_weights
        
        # Encoder
        self.encoder = nn.Linear(input_dim, self.bottleneck_dim)
        
        # Decoder - always create even if tied weights to avoid errors
        self.decoder = nn.Linear(self.bottleneck_dim, input_dim)

        # If using tied weights, we'll just tie them during forward pass
    
    def encode(self, x):
        """
        Encode input to bottleneck representation
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            h: Encoded representation of shape [batch_size, bottleneck_dim]
            h_activated: Activated representation (ReLU) of shape [batch_size, bottleneck_dim]
        """
        h = self.encoder(x)
        h_activated = F.relu(h)  # ReLU activation for sparsity
        return h, h_activated
    
    def decode(self, h):
        """
        Decode bottleneck representation back to input space

        Args:
            h: Encoded representation of shape [batch_size, bottleneck_dim]

        Returns:
            x_reconstructed: Reconstruction of shape [batch_size, input_dim]
        """
        if self.tied_weights:
            # For tied weights, manually use transposed weights before decoding
            with torch.no_grad():
                self.decoder.weight.copy_(self.encoder.weight.t())

        # Always use decoder for consistency
        return self.decoder(h)
    
    def forward(self, x):
        """
        Forward pass through the autoencoder
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            x_reconstructed: Reconstruction of shape [batch_size, input_dim]
            h_activated: Activated representation of shape [batch_size, bottleneck_dim]
            h: Raw encoded representation of shape [batch_size, bottleneck_dim]
        """
        # Encode
        h, h_activated = self.encode(x)
        
        # Decode
        x_reconstructed = self.decode(h_activated)
        
        return x_reconstructed, h_activated, h
    
    def get_sparsity_loss(self, h_activated, l1_coeff=0.01):
        """
        Calculate L1 sparsity penalty on activations
        
        Args:
            h_activated: Activated representation of shape [batch_size, bottleneck_dim]
            l1_coeff: L1 penalty coefficient
            
        Returns:
            l1_loss: L1 sparsity penalty
        """
        return l1_coeff * torch.mean(torch.abs(h_activated))
    
    def get_reconstruction_loss(self, x, x_reconstructed):
        """
        Calculate reconstruction loss (MSE)
        
        Args:
            x: Original input of shape [batch_size, input_dim]
            x_reconstructed: Reconstruction of shape [batch_size, input_dim]
            
        Returns:
            mse_loss: Mean squared error reconstruction loss
        """
        return F.mse_loss(x_reconstructed, x)