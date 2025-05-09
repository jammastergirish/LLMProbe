import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from utils.file_manager import save_graph

def calculate_mean_activation_difference(test_hidden_states, test_labels, layer):
    """Calculate mean activation difference between true and false examples for a specific layer"""
    layer_feats = test_hidden_states[:, layer, :]
    
    true_indices = (test_labels == 1).nonzero(as_tuple=True)[0]
    false_indices = (test_labels == 0).nonzero(as_tuple=True)[0]
    
    if len(true_indices) > 0 and len(false_indices) > 0:
        mean_activations_true = layer_feats[true_indices].mean(dim=0).cpu().numpy()
        mean_activations_false = layer_feats[false_indices].mean(dim=0).cpu().numpy()
        diff_activations = mean_activations_true - mean_activations_false
        return diff_activations, mean_activations_true, mean_activations_false
    else:
        return None, None, None

def calculate_alignment_strengths(test_hidden_states, test_labels, results, num_layers):
    """Calculate alignment strength (correlation coefficient) between mean activation differences and probe weights"""
    all_layers_mean_diff_activations = []
    probe_weights = []
    alignment_strengths = []
    
    if test_hidden_states.nelement() > 0 and test_labels.nelement() > 0:
        for layer_idx in range(num_layers):
            diff_act, _, _ = calculate_mean_activation_difference(test_hidden_states, test_labels, layer_idx)
            
            if diff_act is not None:
                all_layers_mean_diff_activations.append(diff_act)
                
                current_probe_weights = results['probes'][layer_idx].linear.weight[0].cpu().detach().numpy()
                probe_weights.append(current_probe_weights)
                
                # Ensure both arrays are 1D and have the same length
                if diff_act.ndim == 1 and current_probe_weights.ndim == 1 and len(diff_act) == len(current_probe_weights) and len(diff_act) > 1:
                    correlation = np.corrcoef(diff_act, current_probe_weights)[0, 1]
                    alignment_strengths.append(correlation)
                else:
                    # Append NaN if correlation cannot be computed
                    alignment_strengths.append(np.nan)
            else:
                all_layers_mean_diff_activations.append(np.array([]))
                probe_weights.append(np.array([]))
                alignment_strengths.append(np.nan)
                
    return alignment_strengths, all_layers_mean_diff_activations, probe_weights

def get_top_k_neurons(diff_activations, probe_weights, k=10):
    """Identify the top-k most influential neurons based on contribution score"""
    if diff_activations is None or probe_weights is None:
        return []
        
    contribution_scores = np.abs(diff_activations * probe_weights)
    top_k_indices = np.argsort(contribution_scores)[::-1][:k]
    
    top_k_data = []
    for rank, neuron_idx in enumerate(top_k_indices):
        top_k_data.append({
            "Rank": rank + 1,
            "Neuron Index": neuron_idx,
            "Contribution Score (abs(Diff*Weight))": contribution_scores[neuron_idx],
            "Mean Activation Difference": diff_activations[neuron_idx],
            "Probe Weight": probe_weights[neuron_idx]
        })
    
    return top_k_data

def calculate_confusion_matrix(test_hidden_states, test_labels, probe, layer):
    """Calculate confusion matrix components and metrics for a specific layer"""
    test_feats = test_hidden_states[:, layer, :]
    
    with torch.no_grad():
        # Get predictions
        test_outputs = probe(test_feats)
        test_preds = (test_outputs > 0.5).long()
        
        # Make sure tensors are on the same device
        device = test_preds.device
        test_labels_device = test_labels.to(device)
        
        # Confusion matrix components
        TP = ((test_preds == 1) & (test_labels_device == 1)).sum().item()
        FP = ((test_preds == 1) & (test_labels_device == 0)).sum().item()
        TN = ((test_preds == 0) & (test_labels_device == 0)).sum().item()
        FN = ((test_preds == 0) & (test_labels_device == 1)).sum().item()
        
        # Calculate metrics
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
    metrics = {
        'TP': TP,
        'FP': FP,
        'TN': TN,
        'FN': FN,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics

def create_metrics_dataframe(metrics):
    """Create a DataFrame for metrics display"""
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'True Positives',
                   'False Positives', 'True Negatives', 'False Negatives'],
        'Value': [f"{metrics['accuracy']:.4f}", f"{metrics['precision']:.4f}", 
                  f"{metrics['recall']:.4f}", f"{metrics['f1']:.4f}",
                  str(metrics['TP']), str(metrics['FP']), 
                  str(metrics['TN']), str(metrics['FN'])]
    })
    return metrics_df

def plot_truth_direction_projection(test_hidden_states, test_labels, probe, layer, run_folder=None):
    """Plot the projection of test examples onto the truth direction for a specific layer"""
    test_feats = test_hidden_states[:, layer, :]
    
    with torch.no_grad():
        projection = torch.matmul(test_feats, probe.linear.weight[0])
        
        # Get projection values for true and false examples
        true_proj = projection[test_labels == 1].cpu().numpy()
        false_proj = projection[test_labels == 0].cpu().numpy()
        
        # Create histogram
        fig, ax = plt.subplots(figsize=(8, 3))
        bins = np.linspace(
            min(projection.min().item(), -3),
            max(projection.max().item(), 3),
            30
        )
        
        # Plot histograms
        ax.hist(true_proj, bins=bins, alpha=0.7, label="True", color="#4CAF50")
        ax.hist(false_proj, bins=bins, alpha=0.7, label="False", color="#F44336")
        
        # Add a vertical line at the decision boundary (0.0)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel("Projection onto Truth Direction")
        ax.set_ylabel("Count")
        ax.legend()
        
        # Save the figure if run_folder is provided
        if run_folder:
            layer_save_dir = os.path.join(run_folder, "layers", str(layer))
            os.makedirs(layer_save_dir, exist_ok=True)
            save_graph(fig, os.path.join(layer_save_dir, "truth_projection.png"))
            
    return fig

def plot_confusion_matrix(metrics, layer, run_folder=None):
    """Plot confusion matrix for a specific layer"""
    fig, ax = plt.subplots(figsize=(4, 3))
    cm = np.array([[metrics['TN'], metrics['FP']], [metrics['FN'], metrics['TP']]])
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f"Layer {layer} Confusion Matrix")
    
    # Show all ticks and label them
    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(['Predicted False', 'Predicted True'])
    ax.set_yticklabels(['Actual False', 'Actual True'])
    
    # Rotate tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="w" if cm[i, j] > cm.max()/2 else "black")
                    
    plt.tight_layout()
    
    # Save the figure if run_folder is provided
    if run_folder:
        layer_save_dir = os.path.join(run_folder, "layers", str(layer))
        os.makedirs(layer_save_dir, exist_ok=True)
        save_graph(fig, os.path.join(layer_save_dir, "confusion_matrix.png"))
        
    return fig

def plot_probe_weights(probe, layer, run_folder=None):
    """Plot the weights of a probe for a specific layer"""
    probe_weights = probe.linear.weight[0].cpu().detach().numpy()
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(probe_weights)), probe_weights)
    ax.set_title(f"Probe Neuron Weights - Layer {layer}")
    ax.set_xlabel("Neuron Index in Hidden Dimension")
    ax.set_ylabel("Weight Value")
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure if run_folder is provided
    if run_folder:
        layer_save_dir = os.path.join(run_folder, "layers", str(layer))
        os.makedirs(layer_save_dir, exist_ok=True)
        save_graph(fig, os.path.join(layer_save_dir, "probe_weights.png"))
        
    return fig