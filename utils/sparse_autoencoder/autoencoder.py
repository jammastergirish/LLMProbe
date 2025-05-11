import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=0, tied_weights=True, activation_type="ReLU", topk_percent=10):
        """
        Basic sparse autoencoder with optional tied weights

        Args:
            input_dim (int): Dimension of input features
            bottleneck_dim (int): Dimension of bottleneck layer (latent space)
                                If 0, uses same dimension as input
                                          If > input_dim, autoencoder is overcomplete
                                          If < input_dim, autoencoder is undercomplete
            tied_weights (bool): If True, decoder weights are tied to encoder weights
            activation_type (str): Type of activation function to use ('ReLU' or 'BatchTopK')
            topk_percent (int): If using BatchTopK, percentage of activations to keep active per batch
        """
        super().__init__()

        # Set dimensions
        self.input_dim = input_dim
        self.bottleneck_dim = input_dim if bottleneck_dim == 0 else bottleneck_dim
        self.tied_weights = tied_weights
        self.activation_type = activation_type
        self.topk_percent = topk_percent

        # Encoder
        self.encoder = nn.Linear(input_dim, self.bottleneck_dim)

        # Decoder - always create even if tied weights to avoid errors
        self.decoder = nn.Linear(self.bottleneck_dim, input_dim)

        # If using tied weights, we'll just tie them during forward pass


    def batch_topk_activation(self, x, percent=10):
        batch_size, hidden_dim = x.shape
        k = max(1, int(hidden_dim * percent / 100))

        # Get the top-k values and indices for each example in the batch at once
        _, indices = torch.topk(x.abs(), k, dim=1)

        # Create a mask of zeros with the same shape as x
        mask = torch.zeros_like(x)

        # Create batch indices that repeat for each of the k indices
        batch_indices = torch.arange(
            batch_size, device=x.device).view(-1, 1).expand(-1, k)

        # Set the mask to 1 at the top-k positions
        mask.scatter_(1, indices, 1.0)

        # Apply the mask
        return x * mask

    # def batch_topk_activation(self, x, percent=10):
    #     """
    #     Implements batch-wise top-k activation function

    #     Args:
    #         x: Input tensor of shape [batch_size, hidden_dim]
    #         percent: Percentage of activations to keep per batch (e.g., 10 means top 10%)

    #     Returns:
    #         x_activated: Tensor with only the top-k elements kept, rest set to zero
    #     """
    #     batch_size, hidden_dim = x.shape

    #     # Clone the input to avoid modifying it
    #     x_activated = x.clone()

    #     # Calculate how many elements to keep per batch example
    #     k = max(1, int(hidden_dim * percent / 100))

    #     # For each example in the batch
    #     for i in range(batch_size):
    #         # Get the values and indices of the top-k elements
    #         _, indices = torch.topk(x_activated[i].abs(), k)

    #         # Create a mask of zeros
    #         mask = torch.zeros_like(x_activated[i])

    #         # Set the mask to 1 at the indices of the top-k elements
    #         mask[indices] = 1

    #         # Apply the mask to keep only the top-k elements
    #         x_activated[i] = x_activated[i] * mask

    #     return x_activated

    def encode(self, x):
        """
        Encode input to bottleneck representation

        Args:
            x: Input tensor of shape [batch_size, input_dim]

        Returns:
            h: Encoded representation of shape [batch_size, bottleneck_dim]
            h_activated: Activated representation of shape [batch_size, bottleneck_dim]
        """
        h = self.encoder(x)

        # Apply the appropriate activation function
        if self.activation_type == "ReLU":
            h_activated = F.relu(h)  # ReLU activation for sparsity
        elif self.activation_type == "BatchTopK":
            h_activated = self.batch_topk_activation(h, self.topk_percent)
        else:
            # Default to ReLU if an unknown activation type is specified
            h_activated = F.relu(h)

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