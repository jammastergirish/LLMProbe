import torch
from utils.sparse_autoencoder.autoencoder import SparseAutoencoder
from utils.sparse_autoencoder.supervised_autoencoder import SupervisedSparseAutoencoder

def train_autoencoder(features, epochs=100, lr=1e-3, l1_coeff=0.01,
                      bottleneck_dim=0, tied_weights=True, device=torch.device("cpu")):
    """
    Train an unsupervised sparse autoencoder on the given features

    Args:
        features (torch.Tensor): Input features of shape [batch_size, input_dim]
        epochs (int): Number of training epochs
        lr (float): Learning rate
        l1_coeff (float): Coefficient for L1 sparsity penalty
        bottleneck_dim (int): Dimension of bottleneck layer
                                 If 0, uses same dimension as input
        tied_weights (bool): Whether to use tied weights
        device (torch.device): Device to train on

    Returns:
        autoencoder (SparseAutoencoder): Trained autoencoder model
        losses (dict): Dictionary of training losses
    """
    input_dim = features.shape[1]
    autoencoder = SparseAutoencoder(input_dim, bottleneck_dim, tied_weights).to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    
    # Track losses
    losses = {
        'total': [],
        'reconstruction': [],
        'sparsity': []
    }
    
    for epoch in range(epochs):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        reconstructed, h_activated, _ = autoencoder(features)
        
        # Calculate losses
        reconstruction_loss = autoencoder.get_reconstruction_loss(features, reconstructed)
        sparsity_loss = autoencoder.get_sparsity_loss(h_activated, l1_coeff)
        total_loss = reconstruction_loss + sparsity_loss
        
        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()
        
        # Record losses
        losses['total'].append(total_loss.item())
        losses['reconstruction'].append(reconstruction_loss.item())
        losses['sparsity'].append(sparsity_loss.item())
    
    return autoencoder, losses

def train_supervised_autoencoder(features, labels, epochs=100, lr=1e-3, l1_coeff=0.01,
                                bottleneck_dim=0, tied_weights=True,
                                lambda_classify=1.0, device=torch.device("cpu")):
    """
    Train a supervised sparse autoencoder on the given features and labels

    Args:
        features (torch.Tensor): Input features of shape [batch_size, input_dim]
        labels (torch.Tensor): Binary labels of shape [batch_size]
        epochs (int): Number of training epochs
        lr (float): Learning rate
        l1_coeff (float): Coefficient for L1 sparsity penalty
        bottleneck_dim (int): Dimension of bottleneck layer
                                 If 0, uses same dimension as input
        tied_weights (bool): Whether to use tied weights
        lambda_classify (float): Weight for classification loss
        device (torch.device): Device to train on

    Returns:
        autoencoder (SupervisedSparseAutoencoder): Trained autoencoder model
        losses (dict): Dictionary of training losses
    """
    input_dim = features.shape[1]
    autoencoder = SupervisedSparseAutoencoder(input_dim, bottleneck_dim, tied_weights).to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    
    # Track losses
    losses = {
        'total': [],
        'reconstruction': [],
        'sparsity': [],
        'classification': []
    }
    
    for epoch in range(epochs):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        reconstructed, h_activated, classification_probs = autoencoder(features)
        
        # Calculate losses
        reconstruction_loss = autoencoder.get_reconstruction_loss(features, reconstructed)
        sparsity_loss = autoencoder.get_sparsity_loss(h_activated, l1_coeff)
        classification_loss = autoencoder.get_classification_loss(classification_probs, labels)
        
        # Total loss with weighting
        total_loss = reconstruction_loss + sparsity_loss + lambda_classify * classification_loss
        
        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()
        
        # Record losses
        losses['total'].append(total_loss.item())
        losses['reconstruction'].append(reconstruction_loss.item())
        losses['sparsity'].append(sparsity_loss.item())
        losses['classification'].append(classification_loss.item())
    
    return autoencoder, losses

def train_and_evaluate_autoencoders(train_hidden_states, train_labels, test_hidden_states, test_labels,
                                    num_layers, use_supervised, progress_callback=None,
                                    epochs=100, lr=0.001, l1_coeff=0.01, bottleneck_dim=0,
                                    tied_weights=True, lambda_classify=1.0, device=torch.device("cpu")):
    """
    Train sparse autoencoders across all layers and evaluate performance

    Args:
        train_hidden_states (torch.Tensor): Hidden states from training set [batch_size, num_layers, hidden_dim]
        train_labels (torch.Tensor): Labels from training set [batch_size]
        test_hidden_states (torch.Tensor): Hidden states from test set [batch_size, num_layers, hidden_dim]
        test_labels (torch.Tensor): Labels from test set [batch_size]
        num_layers (int): Number of model layers
        use_supervised (bool): Whether to use supervised autoencoders
        progress_callback (callable): Callback for progress updates
        epochs (int): Number of training epochs
        lr (float): Learning rate
        l1_coeff (float): Coefficient for L1 sparsity penalty
        bottleneck_dim (int): Dimension of bottleneck layer
                                 If 0, uses same dimension as input
        tied_weights (bool): Whether to use tied weights
        lambda_classify (float): Weight for classification loss (only used if use_supervised=True)
        device (torch.device): Device to train on

    Returns:
        results (dict): Dictionary of results including trained autoencoders
    """
    autoencoders = []
    reconstruction_errors = []
    sparsity_values = []
    
    # For supervised models
    classification_accuracies = []
    
    for layer in range(num_layers):
        # Update main progress
        main_progress = layer / num_layers
        if progress_callback:
            progress_callback(main_progress, f"Training autoencoder for layer {layer+1}/{num_layers}",
                             f"Working on layer {layer+1} of {num_layers}")
        
        # Extract features for this layer
        train_feats = train_hidden_states[:, layer, :]
        test_feats = test_hidden_states[:, layer, :]

        # Determine appropriate bottleneck dimension for this layer
        current_layer_dim = train_feats.shape[1]  # Get feature dimension for this layer

        # If bottleneck_dim is 0, use the layer's dimension, otherwise use the specified value
        layer_bottleneck_dim = current_layer_dim if bottleneck_dim == 0 else bottleneck_dim

        # Choose training function based on supervision type
        if use_supervised:
            # Train supervised autoencoder
            autoencoder_fn = train_supervised_autoencoder
            autoencoder, losses = autoencoder_fn(
                train_feats, train_labels,
                epochs=epochs, lr=lr, l1_coeff=l1_coeff,
                bottleneck_dim=layer_bottleneck_dim,
                tied_weights=tied_weights,
                lambda_classify=lambda_classify,
                device=device
            )
            
            # Evaluate on test set
            with torch.no_grad():
                _, _, classification_probs = autoencoder(test_feats)
                preds = (classification_probs > 0.5).long()
                accuracy = (preds == test_labels).float().mean().item()
                classification_accuracies.append(accuracy)
                
                if progress_callback:
                    progress_callback(
                        main_progress + 0.5/num_layers,
                        f"Layer {layer+1}/{num_layers}: Classification Accuracy: {accuracy:.4f}",
                        f"Evaluating classification performance"
                    )
        else:
            # Train unsupervised autoencoder
            autoencoder_fn = train_autoencoder
            autoencoder, losses = autoencoder_fn(
                train_feats,
                epochs=epochs, lr=lr, l1_coeff=l1_coeff,
                bottleneck_dim=layer_bottleneck_dim,
                tied_weights=tied_weights,
                device=device
            )
        
        # Compute reconstruction error on test set
        with torch.no_grad():
            reconstructed, h_activated, _ = autoencoder(test_feats)
            recon_error = ((reconstructed - test_feats) ** 2).mean().item()
            sparsity = torch.mean(torch.abs(h_activated)).item()
            
            if progress_callback:
                progress_callback(
                    main_progress + 0.8/num_layers,
                    f"Layer {layer+1}/{num_layers}: Reconstruction Error: {recon_error:.4f}",
                    f"Evaluating reconstruction performance"
                )
        
        # Store results
        autoencoders.append(autoencoder)
        reconstruction_errors.append(recon_error)
        sparsity_values.append(sparsity)
    
    # Final update to 100%
    if progress_callback:
        progress_callback(
            1.0, 
            "Completed training all autoencoders",
            f"Trained autoencoders for {num_layers} layers"
        )
    
    # Compile results
    results = {
        'autoencoders': autoencoders,
        'reconstruction_errors': reconstruction_errors,
        'sparsity_values': sparsity_values
    }
    
    if use_supervised:
        results['classification_accuracies'] = classification_accuracies
    
    return results