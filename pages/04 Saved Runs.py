import os
import json
import streamlit as st
from utils.file_manager import SAVED_DATA_DIR
import pandas as pd

st.set_page_config(page_title="Saved Runs", layout="wide")

st.title("📊 Saved Runs")

# List all saved runs
if os.path.exists(SAVED_DATA_DIR):
    run_folders = sorted(
        [f for f in os.listdir(SAVED_DATA_DIR) if os.path.isdir(
            os.path.join(SAVED_DATA_DIR, f))],
        key=lambda x: os.path.getctime(os.path.join(SAVED_DATA_DIR, x)),
        reverse=True  # Descending order
    )
    if run_folders:
        for run_id in run_folders:
            run_folder = os.path.join(SAVED_DATA_DIR, run_id)

            # Load parameters and results
            with open(os.path.join(run_folder, "parameters.json")) as f:
                parameters = json.load(f)
            with open(os.path.join(run_folder, "results.json")) as f:
                results = json.load(f)

            with st.expander(f"📅 {parameters['datetime']} | 🤖 {parameters['model_name']} | 📊 {parameters['dataset']} | 🔍 {parameters['output_activation']}"):

                # Create tabs for different sections
                run_tabs = st.tabs(
                    ["📋 Overview", "⚙️ Parameters", "📈 Visualizations"])

                # Overview tab
                with run_tabs[0]:
                    st.subheader("Run Information")
                    st.json(parameters)

                # Parameters tab
                with run_tabs[1]:
                    st.subheader("Configuration Parameters")

                    # Create columns for parameters
                    param_cols = st.columns(2)

                    # Model parameters
                    with param_cols[0]:
                        st.caption("🤖 MODEL CONFIGURATION")
                        st.markdown(
                            f"**Model Name:** {parameters['model_name']}")
                        st.markdown(
                            f"**Output Activation:** {parameters['output_activation']}")
                        st.markdown(
                            f"**Device:** {parameters.get('device', 'Not specified')}")

                    # Training parameters
                    with param_cols[1]:
                        st.caption("🧠 PROBE CONFIGURATION")
                        st.markdown(f"**Dataset:** {parameters['dataset']}")
                        st.markdown(
                            f"**Batch Size:** {parameters['batch_size']}")

                        # Handle potential parameter naming differences
                        epochs_key = 'train_epochs' if 'train_epochs' in parameters else 'epochs'
                        st.markdown(f"**Epochs:** {parameters[epochs_key]}")

                        st.markdown(
                            f"**Learning Rate:** {parameters['learning_rate']}")
                        st.markdown(
                            f"**Control Tasks:** {'Yes' if parameters['use_control_tasks'] else 'No'}")

                    # # Show raw parameters as JSON
                    # with st.expander("View Raw Parameters"):
                    #     st.json(parameters)

                # Visualizations tab
                with run_tabs[2]:
                    st.subheader("Analysis Visualizations")

                    # Accuracy plot
                    accuracy_plot_path = os.path.join(
                        run_folder, "accuracy_plot.png")
                    if os.path.exists(accuracy_plot_path):
                        st.caption("📈 ACCURACY PLOT")
                        st.image(accuracy_plot_path, use_container_width=True)

                    # Alignment Strength plot
                    alignment_strength_plot_path = os.path.join(
                        run_folder, "alignment_strength_plot.png")
                    if os.path.exists(alignment_strength_plot_path):
                        st.caption("🔗 ALIGNMENT STRENGTH BY LAYER")
                        st.image(alignment_strength_plot_path,
                                 use_container_width=True)

                    # PCA plot
                    pca_plot_path = os.path.join(run_folder, "pca_plot.png")
                    if os.path.exists(pca_plot_path):
                        st.caption("🔍 PCA VISUALIZATION")
                        st.image(pca_plot_path, use_container_width=True)

                    # Truth direction plot
                    truth_direction_plot_path = os.path.join(
                        run_folder, "proj_plot.png")
                    if os.path.exists(truth_direction_plot_path):
                        st.caption("🧭 TRUTH DIRECTION PLOT")
                        st.image(truth_direction_plot_path,
                                 use_container_width=True)

                    # --- Add Per-Layer Visualizations ---
                    st.markdown("--- ")  # Separator
                    st.subheader("Per-Layer Visualizations")

                    layers_dir = os.path.join(run_folder, "layers")
                    if os.path.exists(layers_dir) and os.path.isdir(layers_dir):
                        layer_subdirs = sorted(
                            [d for d in os.listdir(layers_dir) if os.path.isdir(
                                os.path.join(layers_dir, d)) and d.isdigit()],
                            key=int  # Sort numerically
                        )

                        if layer_subdirs:
                            layer_viz_tabs = st.tabs(
                                [f"Layer {d}" for d in layer_subdirs])

                            for idx, layer_num_str in enumerate(layer_subdirs):
                                with layer_viz_tabs[idx]:
                                    layer_viz_dir = os.path.join(
                                        layers_dir, layer_num_str)

                                    # Define expected paths
                                    probe_weights_path = os.path.join(
                                        layer_viz_dir, "probe_weights.png")
                                    activation_diff_path = os.path.join(
                                        layer_viz_dir, "activation_diff.png")
                                    truth_proj_path = os.path.join(
                                        layer_viz_dir, "truth_projection.png")
                                    conf_matrix_path = os.path.join(
                                        layer_viz_dir, "confusion_matrix.png")
                                    neuron_alignment_path = os.path.join(
                                        layer_viz_dir, "neuron_alignment.png")

                                    # Display if exists
                                    if os.path.exists(probe_weights_path):
                                        st.image(
                                            probe_weights_path, caption="Probe Neuron Weights", use_container_width=True)
                                    if os.path.exists(activation_diff_path):
                                        st.image(
                                            activation_diff_path, caption="Mean Activation Difference (True-False)", use_container_width=True)
                                    if os.path.exists(truth_proj_path):
                                        st.image(
                                            truth_proj_path, caption="Truth Direction Projection", use_container_width=True)
                                    if os.path.exists(conf_matrix_path):
                                        st.image(
                                            conf_matrix_path, caption="Confusion Matrix", use_container_width=True)
                                    if os.path.exists(neuron_alignment_path):
                                        st.image(
                                            neuron_alignment_path, caption="Neuron Alignment (Weight vs. Activation Diff)", use_container_width=True)

                        else:
                            st.info(
                                "No per-layer visualization subdirectories found.")
                    else:
                        st.info(
                            "No per-layer visualizations were saved for this run.")
                    # --- End Per-Layer Visualizations ---
    else:
        st.info("No saved runs found.")
else:
    st.info("No saved runs found.")
