import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.sparse_autoencoder.autoencoder import SparseAutoencoder

class SupervisedSparseAutoencoder(SparseAutoencoder):
    def __init__(self, input_dim, bottleneck_dim=0, tied_weights=True, activation_type="ReLU", topk_percent=10):
        """
        Supervised sparse autoencoder that adds a classification head for label prediction

        Args:
            input_dim (int): Dimension of input features
            bottleneck_dim (int): Dimension of bottleneck layer (latent space)
                                If 0, uses same dimension as input
            tied_weights (bool): If True, decoder weights are tied to encoder weights
            activation_type (str): Type of activation function ('ReLU' or 'BatchTopK')
            topk_percent (int): Percentage of activations to keep if using BatchTopK
        """
        super().__init__(input_dim, bottleneck_dim, tied_weights, activation_type, topk_percent)
        
        # Add classification head
        self.classifier = nn.Linear(self.bottleneck_dim, 1)
    
    def classify(self, h_activated):
        """
        Classify the encoded representation
        
        Args:
            h_activated: Activated representation from encoder
            
        Returns:
            logits: Raw logits from classifier
            probs: Sigmoid probabilities for binary classification
        """
        logits = self.classifier(h_activated)
        probs = torch.sigmoid(logits).squeeze(-1)
        return logits, probs
    
    def forward(self, x):
        """
        Forward pass through the supervised autoencoder
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            x_reconstructed: Reconstruction of shape [batch_size, input_dim]
            h_activated: Activated representation
            classification_probs: Classification probabilities
        """
        # Get encoded representation and reconstruction from parent class
        x_reconstructed, h_activated, h = super().forward(x)
        
        # Add classification
        _, classification_probs = self.classify(h_activated)
        
        return x_reconstructed, h_activated, classification_probs
    
    def get_classification_loss(self, classification_probs, labels):
        """
        Calculate binary cross-entropy loss for classification
        
        Args:
            classification_probs: Predicted probabilities
            labels: Ground truth labels
            
        Returns:
            bce_loss: Binary cross-entropy loss
        """
        return F.binary_cross_entropy(classification_probs, labels.float())