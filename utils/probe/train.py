import torch
from utils.probe.linear_probe import LinearProbe

def train_probe(features, labels, epochs=100, lr=1e-2, device=torch.device("cpu")):
    """Train a linear probe on the given features and labels"""
    probe = LinearProbe(features.shape[1]).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = probe(features)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

    return probe, loss.item()


def train_and_evaluate_model(train_hidden_states, train_labels, test_hidden_states, test_labels,
                             num_layers, use_control_tasks, progress_callback=None, epochs=100, lr=0.01,
                             device=torch.device("cpu")):
    """Train probes across all layers and evaluate performance"""
    probes = []
    accuracies = []
    control_accuracies = []
    selectivities = []
    losses = []
    test_losses = []

    for layer in range(num_layers):
        # Update main progress
        main_progress = (layer) / num_layers
        progress_callback(main_progress, f"Training probe for layer {layer+1}/{num_layers}",
                          f"Working on layer {layer+1} of {num_layers}")

        train_feats = train_hidden_states[:, layer, :]
        test_feats = test_hidden_states[:, layer, :]

        # Train probe with epoch progress
        probe = LinearProbe(train_feats.shape[1]).to(device)
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

        # Print initial loss and accuracy
        with torch.no_grad():
            outputs = probe(train_feats)
            loss = criterion(outputs, train_labels.float()).item()
            preds = (outputs > 0.5).long()
            acc = (preds == train_labels).float().mean().item()
            output_msg = f"Layer {layer+1}/{num_layers} - Initial: loss={loss:.4f}, acc={acc:.4f}"
            print(output_msg)

            # Update UI if progress_callback is provided
            if progress_callback and hasattr(progress_callback, 'add_training_output'):
                progress_callback.add_training_output(output_msg)

        for epoch in range(epochs):
            # Progress tracker update
            if epoch % 10 == 0 or epoch == epochs - 1:
                epoch_progress = main_progress + (epoch / epochs) / num_layers
                progress_callback(epoch_progress,
                                 f"Layer {layer+1}/{num_layers}: Epoch {epoch+1}/{epochs}",
                                 f"Training linear probe for truth detection")

            # Training step
            optimizer.zero_grad()
            outputs = probe(train_feats)
            loss = criterion(outputs, train_labels.float())
            loss.backward()
            optimizer.step()

            # Print training progress every 10 epochs
            if epoch % 10 == 0 or epoch == epochs - 1:
                with torch.no_grad():
                    current_loss = loss.item()
                    preds = (outputs > 0.5).long()
                    acc = (preds == train_labels).float().mean().item()
                    output_msg = f"Layer {layer+1}/{num_layers} - Epoch {epoch+1}/{epochs}: loss={current_loss:.4f}, acc={acc:.4f}"
                    print(output_msg)

                    # Update UI if progress_callback is provided
                    if progress_callback and hasattr(progress_callback, 'add_training_output'):
                        progress_callback.add_training_output(output_msg)

        # Save trained probe
        probes.append(probe)
        losses.append(loss.item())

        # Evaluate on test set
        with torch.no_grad():
            test_outputs = probe(test_feats)
            test_loss = criterion(test_outputs, test_labels.float())
            test_losses.append(test_loss.item())

            preds = (test_outputs > 0.5).long()
            acc = (preds == test_labels).float().mean().item()
            accuracies.append(acc)

        # Log evaluation results
        progress_callback(main_progress + 0.5/num_layers,
                          f"Layer {layer+1}/{num_layers}: Evaluation",
                          f"Layer {layer} accuracy: {acc:.4f}, loss: {test_loss.item():.4f}")

        # Control task (if enabled)
        if use_control_tasks:
            progress_callback(main_progress + 0.6/num_layers,
                              f"Layer {layer+1}/{num_layers}: Control task",
                              f"Training with shuffled labels to measure selectivity")

            shuffled_labels = train_labels[torch.randperm(
                train_labels.size(0))]
            ctrl_probe, _ = train_probe(
                train_feats, shuffled_labels, epochs=epochs, lr=lr, device=device)

            with torch.no_grad():
                ctrl_outputs = ctrl_probe(test_feats)
                ctrl_preds = (ctrl_outputs > 0.5).long()
                ctrl_acc = (ctrl_preds == test_labels).float().mean().item()
                control_accuracies.append(ctrl_acc)

                selectivity = acc - ctrl_acc
                selectivities.append(selectivity)

            progress_callback(main_progress + 0.9/num_layers,
                              f"Layer {layer+1}/{num_layers}: Control accuracy: {ctrl_acc:.4f}",
                              f"Selectivity: {selectivity:.4f} (Acc={acc:.4f} - Control={ctrl_acc:.4f})")

    # Update to 100%
    progress_callback(1.0, "Completed training all probes",
                      f"Trained probes for {num_layers} layers with best accuracy: {max(accuracies):.4f}")

    results = {
        'probes': probes,
        'accuracies': accuracies,
        'control_accuracies': control_accuracies if use_control_tasks else None,
        'selectivities': selectivities if use_control_tasks else None,
        'losses': losses,
        'test_losses': test_losses
    }

    return results