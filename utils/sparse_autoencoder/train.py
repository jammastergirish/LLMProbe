import torch
from utils.sparse_autoencoder.autoencoder import SparseAutoencoder
from utils.sparse_autoencoder.supervised_autoencoder import SupervisedSparseAutoencoder

def train_autoencoder(features, epochs=100, lr=1e-3, l1_coeff=0.01,
                      bottleneck_dim=0, tied_weights=True, device=torch.device("cpu"),
                      progress_callback=None, activation_type="ReLU", topk_percent=10):
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
        activation_type (str): Type of activation function ('ReLU' or 'BatchTopK')
        topk_percent (int): Percentage of activations to keep if using BatchTopK

    Returns:
        autoencoder (SparseAutoencoder): Trained autoencoder model
        losses (dict): Dictionary of training losses
    """
    input_dim = features.shape[1]
    autoencoder = SparseAutoencoder(input_dim, bottleneck_dim, tied_weights,
                                   activation_type=activation_type,
                                   topk_percent=topk_percent).to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)

    # Track losses
    losses = {
        'total': [],
        'reconstruction': [],
        'sparsity': []
    }

    # Get initial loss values to track improvement
    with torch.no_grad():
        initial_reconstructed, initial_h_activated, _ = autoencoder(features)
        initial_recon_loss = autoencoder.get_reconstruction_loss(features, initial_reconstructed).item()
        initial_sparsity_loss = autoencoder.get_sparsity_loss(initial_h_activated, l1_coeff).item()
        initial_total_loss = initial_recon_loss + initial_sparsity_loss

        # Calculate activation sparsity (percentage of neurons that are zero)
        active_neurons = torch.sum(initial_h_activated > 0).item()
        total_neurons = initial_h_activated.numel()
        sparsity_percentage = 100 * (1 - active_neurons / total_neurons)

        output_msg = f"AUTOENCODER - Initial: total_loss={initial_total_loss:.4f}, recon={initial_recon_loss:.4f}, " \
                 f"sparsity={initial_sparsity_loss:.4f}, zeros={sparsity_percentage:.1f}%"
        print(output_msg)

        # Update UI if progress_callback is provided
        if progress_callback and hasattr(progress_callback, 'add_training_output'):
            progress_callback.add_training_output(output_msg)

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
        current_total = total_loss.item()
        current_recon = reconstruction_loss.item()
        current_sparsity = sparsity_loss.item()

        losses['total'].append(current_total)
        losses['reconstruction'].append(current_recon)
        losses['sparsity'].append(current_sparsity)

        # Print progress at intervals
        if epoch % 10 == 0 or epoch == epochs - 1:
            # Calculate activation sparsity
            with torch.no_grad():
                active_neurons = torch.sum(h_activated > 0).item()
                total_neurons = h_activated.numel()
                sparsity_percentage = 100 * (1 - active_neurons / total_neurons)

            output_msg = f"AUTOENCODER - Epoch {epoch+1}/{epochs}: total={current_total:.4f}, recon={current_recon:.4f}, " \
                     f"sparsity={current_sparsity:.4f}, zeros={sparsity_percentage:.1f}%"
            print(output_msg)

            # Update UI if progress_callback is provided
            if progress_callback and hasattr(progress_callback, 'add_training_output'):
                progress_callback.add_training_output(output_msg)

    # Print final summary
    output_msg = f"AUTOENCODER - Finished: initial_loss={initial_total_loss:.4f}, final_loss={current_total:.4f}, " \
              f"improvement={initial_total_loss - current_total:.4f}"
    print(output_msg)

    # Update UI if progress_callback is provided
    if progress_callback and hasattr(progress_callback, 'add_training_output'):
        progress_callback.add_training_output(output_msg)

    return autoencoder, losses

def train_supervised_autoencoder(features, labels, epochs=100, lr=1e-3, l1_coeff=0.01,
                                bottleneck_dim=0, tied_weights=True,
                                lambda_classify=1.0, device=torch.device("cpu"),
                                progress_callback=None, activation_type="ReLU", topk_percent=10):
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
        activation_type (str): Type of activation function ('ReLU' or 'BatchTopK')
        topk_percent (int): Percentage of activations to keep if using BatchTopK

    Returns:
        autoencoder (SupervisedSparseAutoencoder): Trained autoencoder model
        losses (dict): Dictionary of training losses
    """
    input_dim = features.shape[1]
    autoencoder = SupervisedSparseAutoencoder(input_dim, bottleneck_dim, tied_weights,
                                             activation_type=activation_type,
                                             topk_percent=topk_percent).to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)

    # Track losses
    losses = {
        'total': [],
        'reconstruction': [],
        'sparsity': [],
        'classification': []
    }

    # Get initial loss values to track improvement
    with torch.no_grad():
        initial_reconstructed, initial_h_activated, initial_probs = autoencoder(features)
        initial_recon_loss = autoencoder.get_reconstruction_loss(features, initial_reconstructed).item()
        initial_sparsity_loss = autoencoder.get_sparsity_loss(initial_h_activated, l1_coeff).item()
        initial_class_loss = autoencoder.get_classification_loss(initial_probs, labels).item()
        initial_total_loss = initial_recon_loss + initial_sparsity_loss + lambda_classify * initial_class_loss

        # Calculate initial accuracy
        initial_preds = (initial_probs > 0.5).long()
        initial_acc = (initial_preds == labels).float().mean().item()

        # Calculate activation sparsity (percentage of neurons that are zero)
        active_neurons = torch.sum(initial_h_activated > 0).item()
        total_neurons = initial_h_activated.numel()
        sparsity_percentage = 100 * (1 - active_neurons / total_neurons)

        output_msg = f"SUPERVISED AUTOENCODER - Initial: total_loss={initial_total_loss:.4f}, recon={initial_recon_loss:.4f}, " \
                 f"sparsity={initial_sparsity_loss:.4f}, classify={initial_class_loss:.4f}, " \
                 f"acc={initial_acc:.4f}, zeros={sparsity_percentage:.1f}%"
        print(output_msg)

        # Update UI if progress_callback is provided
        if progress_callback and hasattr(progress_callback, 'add_training_output'):
            progress_callback.add_training_output(output_msg)

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
        current_total = total_loss.item()
        current_recon = reconstruction_loss.item()
        current_sparsity = sparsity_loss.item()
        current_class = classification_loss.item()

        losses['total'].append(current_total)
        losses['reconstruction'].append(current_recon)
        losses['sparsity'].append(current_sparsity)
        losses['classification'].append(current_class)

        # Print progress at intervals
        if epoch % 10 == 0 or epoch == epochs - 1:
            # Calculate metrics
            with torch.no_grad():
                preds = (classification_probs > 0.5).long()
                acc = (preds == labels).float().mean().item()

                # Calculate activation sparsity
                active_neurons = torch.sum(h_activated > 0).item()
                total_neurons = h_activated.numel()
                sparsity_percentage = 100 * (1 - active_neurons / total_neurons)

            output_msg = f"SUPERVISED AUTOENCODER - Epoch {epoch+1}/{epochs}: total={current_total:.4f}, recon={current_recon:.4f}, " \
                     f"sparsity={current_sparsity:.4f}, classify={current_class:.4f}, acc={acc:.4f}, zeros={sparsity_percentage:.1f}%"
            print(output_msg)

            # Update UI if progress_callback is provided
            if progress_callback and hasattr(progress_callback, 'add_training_output'):
                progress_callback.add_training_output(output_msg)

    # Print final summary
    with torch.no_grad():
        final_preds = (classification_probs > 0.5).long()
        final_acc = (final_preds == labels).float().mean().item()
        output_msg = f"SUPERVISED AUTOENCODER - Finished: initial_loss={initial_total_loss:.4f}, final_loss={current_total:.4f}, " \
                 f"improvement={initial_total_loss - current_total:.4f}, final_acc={final_acc:.4f}"
        print(output_msg)

        # Update UI if progress_callback is provided
        if progress_callback and hasattr(progress_callback, 'add_training_output'):
            progress_callback.add_training_output(output_msg)

    return autoencoder, losses

def train_and_evaluate_autoencoders(train_hidden_states, train_labels, test_hidden_states, test_labels,
                                    num_layers, use_supervised, progress_callback=None,
                                    epochs=100, lr=0.001, l1_coeff=0.01, bottleneck_dim=0,
                                    tied_weights=True, lambda_classify=1.0, device=torch.device("cpu"),
                                    print_function=print, activation_type="ReLU", topk_percent=10):
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
        activation_type (str): Type of activation function ('ReLU' or 'BatchTopK')
        topk_percent (int): Percentage of activations to keep if using BatchTopK

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
                device=device,
                progress_callback=progress_callback,
                activation_type=activation_type,
                topk_percent=topk_percent
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
                device=device,
                progress_callback=progress_callback,
                activation_type=activation_type,
                topk_percent=topk_percent
            )
        
        # Compute reconstruction error on test set
        with torch.no_grad():
            reconstructed, h_activated, test_output = autoencoder(test_feats)
            recon_error = ((reconstructed - test_feats) ** 2).mean().item()
            sparsity = torch.mean(torch.abs(h_activated)).item()

            # Calculate activation sparsity on test set
            active_neurons = torch.sum(h_activated > 0).item()
            total_neurons = h_activated.numel()
            sparsity_percentage = 100 * (1 - active_neurons / total_neurons)

            # Print test evaluation metrics
            output_msg = f"Layer {layer+1}/{num_layers} TEST - Reconstruction Error: {recon_error:.4f}, " \
                        f"L1 Sparsity: {sparsity:.4f}, Zero Activations: {sparsity_percentage:.1f}%"
            print(output_msg)

            # Update UI if progress_callback is provided
            if progress_callback and hasattr(progress_callback, 'add_training_output'):
                progress_callback.add_training_output(output_msg)

            # Add classification metrics for supervised autoencoders
            if use_supervised and isinstance(test_output, torch.Tensor):
                test_preds = (test_output > 0.5).long()
                test_acc = (test_preds == test_labels).float().mean().item()
                class_output_msg = f"Layer {layer+1}/{num_layers} TEST - Classification Accuracy: {test_acc:.4f}"
                print(class_output_msg)

                # Update UI if progress_callback is provided
                if progress_callback and hasattr(progress_callback, 'add_training_output'):
                    progress_callback.add_training_output(class_output_msg)

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