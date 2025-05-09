import matplotlib.pyplot as plt
import numpy as np
import math
import os
import torch
from sklearn.decomposition import PCA
from utils.file_manager import save_graph

def plot_accuracy_by_layer(accuracies, model_name, dataset_source):
    """Plot accuracy by layer"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(accuracies)), accuracies, marker="o", linewidth=2)
    ax.set_title(
        f"Truth Detection Accuracy per Layer ({model_name})", fontsize=14)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.grid(True, alpha=0.3)
    # Add exact values as text labels
    for i, acc in enumerate(accuracies):
        ax.annotate(f"{acc:.3f}", (i, acc), textcoords="offset points",
                    xytext=(0, 5), ha='center')
    plt.tight_layout()
    return fig


def plot_selectivity_by_layer(selectivities, accuracies, control_accuracies, model_name, dataset_source):
    """Plot selectivity by layer with accuracy and control accuracy"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot all three metrics
    ax.plot(range(len(accuracies)), accuracies, marker="o", linewidth=2,
            label="Accuracy", color="#1E88E5")
    ax.plot(range(len(control_accuracies)), control_accuracies, marker="s", linewidth=2,
            linestyle='--', label="Control Accuracy", color="#FFC107")
    ax.plot(range(len(selectivities)), selectivities, marker="^", linewidth=2,
            label="Selectivity", color="#4CAF50")

    ax.set_title(f"Selectivity per Layer ({model_name})", fontsize=14)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add exact values for selectivity
    for i, sel in enumerate(selectivities):
        ax.annotate(f"{sel:.3f}", (i, sel), textcoords="offset points",
                    xytext=(0, 5), ha='center', color="#4CAF50")

    plt.tight_layout()
    return fig


def plot_pca_grid(test_hidden_states, test_labels, probes, model_name, dataset_source):
    """Generate PCA grid visualization"""
    num_layers = test_hidden_states.shape[1]
    cols = math.ceil(math.sqrt(num_layers))
    rows = math.ceil(num_layers / cols)

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    if rows * cols > 1:
        axs = axs.flatten()
    else:
        axs = np.array([axs])

    for layer in range(min(num_layers, rows * cols)):
        feats = test_hidden_states[:, layer, :].cpu().numpy()
        lbls = test_labels.cpu().numpy()

        # PCA
        pca = PCA(n_components=2)
        feats_2d = pca.fit_transform(feats)

        # Probing predictions
        probe = probes[layer]
        with torch.no_grad():
            preds = (probe(torch.tensor(feats).to(probe.linear.weight.device))
                     > 0.5).long().cpu().numpy()

        acc = (preds == lbls).mean()

        # Calculate explained variance
        expl_var = sum(pca.explained_variance_ratio_) * 100

        # Get correct subplot
        ax = axs[layer]

        # Plot PCA
        true_points = ax.scatter(
            feats_2d[lbls == 1][:, 0],
            feats_2d[lbls == 1][:, 1],
            color="#4CAF50",  # Green
            alpha=0.7,
            label="True",
            s=20,
            edgecolors='w',
            linewidths=0.5
        )
        false_points = ax.scatter(
            feats_2d[lbls == 0][:, 0],
            feats_2d[lbls == 0][:, 1],
            color="#F44336",  # Red
            alpha=0.7,
            label="False",
            s=20,
            edgecolors='w',
            linewidths=0.5
        )

        # Highlight misclassified points
        misclassified = preds != lbls
        if np.any(misclassified):
            ax.scatter(
                feats_2d[misclassified][:, 0],
                feats_2d[misclassified][:, 1],
                s=100,
                facecolors='none',
                edgecolors='#2196F3',  # Blue
                linewidths=1.5,
                alpha=0.8,
                label="Misclassified"
            )

        ax.set_title(f"Layer {layer} (Acc={acc:.3f}, Var={expl_var:.1f}%)")
        ax.set_xticks([])
        ax.set_yticks([])

        # Add decision boundary if possible
        try:
            # Create a mesh grid
            x_min, x_max = feats_2d[:, 0].min(
            ) - 0.5, feats_2d[:, 0].max() + 0.5
            y_min, y_max = feats_2d[:, 1].min(
            ) - 0.5, feats_2d[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                 np.linspace(y_min, y_max, 100))

            # Transform back to high-dimensional space (approximate)
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            high_dim_grid = pca.inverse_transform(grid_points)

            # Apply the probe
            with torch.no_grad():
                Z = probe(torch.tensor(high_dim_grid).float().to(
                    probe.linear.weight.device)).cpu().numpy()
            Z = Z.reshape(xx.shape)

            # Plot the decision boundary
            ax.contour(xx, yy, Z, levels=[0.5],
                       colors='k', alpha=0.5, linestyles='--')
        except Exception as e:
            # Skip decision boundary if it fails
            pass

    # Add legend to the first subplot with room
    if num_layers > 0:
        if rows * cols > num_layers:
            # Find an empty subplot
            empty_ax = axs[num_layers]
            empty_ax.axis('off')
            empty_ax.legend([true_points, false_points],
                            ['True', 'False'],
                            fontsize=12, loc='center')
        else:
            # Add legend to the first subplot
            axs[0].legend(fontsize=8, loc='best')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle(f"PCA Visualization of Representations by Layer ({model_name})",
                 fontsize=16, y=0.98)
    return fig


def plot_truth_projections(test_hidden_states, test_labels, probes):
    """Plot truth direction projection histograms"""
    num_layers = test_hidden_states.shape[1]
    rows = cols = math.ceil(num_layers**0.5)

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axs = axs.flatten()

    for layer in range(num_layers):
        feats = test_hidden_states[:, layer, :]
        lbls = test_labels

        probe = probes[layer]
        with torch.no_grad():
            projection = torch.matmul(
                feats, probe.linear.weight[0])  # shape: [N]
            probs = torch.sigmoid(projection)
            preds = (probs > 0.5).long()
            acc = (preds == lbls).float().mean().item()

        ax = axs[layer]

        # Get projection values for true and false examples
        true_proj = projection[lbls == 1].cpu().numpy()
        false_proj = projection[lbls == 0].cpu().numpy()

        # Calculate histogram stats for visualization
        bins = np.linspace(
            min(projection.min().item(), -3),
            max(projection.max().item(), 3),
            30
        )

        # Plot histograms
        ax.hist(true_proj, bins=bins, alpha=0.7, label="True", color="#4CAF50")
        ax.hist(false_proj, bins=bins, alpha=0.7,
                label="False", color="#F44336")

        # Add a vertical line at the decision boundary (0.0)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)

        # Calculate overlap
        hist_true, _ = np.histogram(true_proj, bins=bins)
        hist_false, _ = np.histogram(false_proj, bins=bins)
        overlap = np.minimum(hist_true, hist_false).sum(
        ) / max(1, max(hist_true.sum(), hist_false.sum()))

        ax.set_title(f"Layer {layer} (Acc={acc:.3f}, Overlap={overlap:.2f})")
        ax.set_xticks([])
        ax.set_yticks([])

    # Only add legend to the first subplot
    if num_layers > 0:
        axs[0].legend(fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle("Projection onto Truth Direction per Layer",
                 fontsize=20, y=0.98)
    return fig


def plot_neuron_alignment(mean_diff, weights, layer_index, run_folder):
    """
    Plots probe weight vs. mean activation difference for neurons.
    Size of points indicates combined importance.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Calculate product for size emphasis (ensure non-negative sizes)
    # Adding a small epsilon to avoid zero size for points on axes but still important
    sizes = np.abs(mean_diff * weights) * 1000 + \
        5  # Scaled for visibility + base size
    sizes = np.clip(sizes, 5, 500)  # Clip sizes to a reasonable range

    scatter = ax.scatter(mean_diff, weights, s=sizes,
                         alpha=0.7, cmap="viridis", c=sizes)

    ax.axhline(0, color='grey', lw=0.8, linestyle='--')
    ax.axvline(0, color='grey', lw=0.8, linestyle='--')

    ax.set_xlabel("Mean Activation Difference (True - False)", fontsize=12)
    ax.set_ylabel("Probe Weight", fontsize=12)
    ax.set_title(
        f"Neuron Alignment: Weight vs. Activation Diff - Layer {layer_index}", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Add quadrant labels for interpretation
    ax.text(0.95, 0.95, "High Diff, High Weight (Aligned True)", transform=ax.transAxes, ha='right', va='top',
            fontsize=9, color='green', bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='green', alpha=0.7))
    ax.text(0.05, 0.05, "Low Diff, Low Weight (Aligned False)", transform=ax.transAxes, ha='left', va='bottom',
            fontsize=9, color='red', bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='red', alpha=0.7))
    ax.text(0.05, 0.95, "Low Diff, High Weight (Probe relies, low natural signal for True)", transform=ax.transAxes,
            ha='left', va='top', fontsize=9, color='blue', bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='blue', alpha=0.7))
    ax.text(0.95, 0.05, "High Diff, Low Weight (Probe relies, low natural signal for False)", transform=ax.transAxes, ha='right',
            va='bottom', fontsize=9, color='purple', bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='purple', alpha=0.7))

    plt.tight_layout()

    # Save the figure
    layer_save_dir = os.path.join(run_folder, "layers", str(layer_index))
    os.makedirs(layer_save_dir, exist_ok=True)
    save_graph(fig, os.path.join(layer_save_dir, "neuron_alignment.png"))

    return fig


def plot_alignment_strength_by_layer(alignment_strengths, model_name, dataset_source, run_folder):
    """Plot alignment strength (correlation coefficient) by layer."""
    fig, ax = plt.subplots(figsize=(10, 6))
    layers = range(len(alignment_strengths))
    ax.plot(layers, alignment_strengths, marker="o",
            linewidth=2, color="#6A0DAD")  # Purple
    ax.set_title(
        f"Alignment Strength (Probe Weight vs. Activation Diff Correlation) - {model_name} on {dataset_source}", fontsize=12)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Pearson Correlation Coefficient", fontsize=12)
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color='grey', lw=0.8, linestyle='--')
    ax.grid(True, alpha=0.3)
    for i, corr in enumerate(alignment_strengths):
        ax.annotate(f"{corr:.3f}", (i, corr), textcoords="offset points",
                    xytext=(0, 5), ha='center')
    plt.tight_layout()
    save_graph(fig, os.path.join(run_folder, "alignment_strength_plot.png"))
    return fig