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
                    ["📋 Overview", "⚙️ Parameters", "📈 Visualizations", "📝 Log", "💾 Data Files"])

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
                    # Create sub-tabs for different types of visualizations
                    viz_tabs = st.tabs(["Linear Probe Results", "Sparse Autoencoder Results"])

                    # Linear Probe Visualizations
                    with viz_tabs[0]:
                        st.subheader("Linear Probe Analysis")

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

                    # Sparse Autoencoder Visualizations
                    with viz_tabs[1]:
                        st.subheader("Sparse Autoencoder Analysis")

                        # Check if this run has autoencoder results
                        autoencoder_stats_path = os.path.join(run_folder, "autoencoder_stats.json")
                        sparsity_plot_path = os.path.join(run_folder, "sparsity_plot.png")
                        l1_sparsity_plot_path = os.path.join(run_folder, "l1_sparsity_plot.png")
                        reconstruction_error_plot_path = os.path.join(run_folder, "reconstruction_error_plot.png")

                        if os.path.exists(autoencoder_stats_path):
                            # Load and display autoencoder stats
                            with open(autoencoder_stats_path) as f:
                                autoencoder_stats = json.load(f)

                            # Display basic info
                            st.caption("🔧 AUTOENCODER CONFIGURATION")
                            config_cols = st.columns(3)
                            with config_cols[0]:
                                st.metric("L1 Coefficient", autoencoder_stats.get("l1_coefficient", "N/A"))
                                st.metric("Type", autoencoder_stats.get("autoencoder_type", "N/A"))
                            with config_cols[1]:
                                bottleneck = autoencoder_stats.get("bottleneck_dim", "N/A")
                                bottleneck_display = "Same as input" if bottleneck == 0 else bottleneck
                                st.metric("Hidden Dimension", bottleneck_display)
                                st.metric("Tied Weights", "Yes" if autoencoder_stats.get("tied_weights", False) else "No")
                            with config_cols[2]:
                                st.metric("Epochs", autoencoder_stats.get("training_epochs", "N/A"))
                                st.metric("Learning Rate", autoencoder_stats.get("learning_rate", "N/A"))

                            # Display sparsity plot
                            if os.path.exists(sparsity_plot_path):
                                st.caption("📊 SPARSITY PERCENTAGE BY LAYER")
                                st.image(sparsity_plot_path, use_container_width=True)

                            # Display L1 sparsity plot
                            if os.path.exists(l1_sparsity_plot_path):
                                st.caption("📉 L1 SPARSITY MEASURE BY LAYER")
                                st.image(l1_sparsity_plot_path, use_container_width=True)

                            # Display reconstruction error plot
                            if os.path.exists(reconstruction_error_plot_path):
                                st.caption("🔄 RECONSTRUCTION ERROR BY LAYER")
                                st.image(reconstruction_error_plot_path, use_container_width=True)

                            # Create expandable section to show raw data
                            with st.expander("View Raw Sparsity and Reconstruction Data"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.subheader("Sparsity Values")
                                    if "sparsity_values" in autoencoder_stats:
                                        sparsity_df = pd.DataFrame({
                                            "Layer": range(len(autoencoder_stats["sparsity_values"])),
                                            "L1 Sparsity": autoencoder_stats["sparsity_values"]
                                        })
                                        st.dataframe(sparsity_df)
                                    else:
                                        st.info("No sparsity values available")

                                with col2:
                                    st.subheader("Reconstruction Errors")
                                    if "reconstruction_errors" in autoencoder_stats:
                                        recon_df = pd.DataFrame({
                                            "Layer": range(len(autoencoder_stats["reconstruction_errors"])),
                                            "MSE": autoencoder_stats["reconstruction_errors"]
                                        })
                                        st.dataframe(recon_df)
                                    else:
                                        st.info("No reconstruction error values available")
                        else:
                            st.info("No sparse autoencoder analysis was performed for this run.")

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

                # Log tab
                with run_tabs[3]:
                    st.subheader("Log")
                    log_file_path = os.path.join(run_folder, "log.txt")

                    if os.path.exists(log_file_path):
                        with open(log_file_path, "rb") as f:
                            st.download_button(
                                label="📥 Download Log File",
                                data=f,
                                file_name="log.txt",
                                mime="text/plain"
                            )
                    else:
                        st.info("No log file found for this run.")

                # Data Files tab
                with run_tabs[4]:
                    st.subheader("Download Data Files")

                    # Check for representations
                    train_representations_path = os.path.join(run_folder, "train_representations.npy")
                    test_representations_path = os.path.join(run_folder, "test_representations.npy")
                    probe_weights_path = os.path.join(run_folder, "probe_weights.json")
                    probe_weights_pt_path = os.path.join(run_folder, "probe_weights.pt")

                    # Create columns for better layout
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### Representations")
                        files_found = False

                        # Train representations
                        if os.path.exists(train_representations_path):
                            files_found = True
                            with open(train_representations_path, "rb") as f:
                                st.download_button(
                                    label="📥 Download Train Representations (NPY)",
                                    data=f,
                                    file_name="train_representations.npy",
                                    mime="application/octet-stream"
                                )

                            # Check for metadata JSON
                            train_meta_path = train_representations_path.replace('.npy', '_metadata.json')
                            if os.path.exists(train_meta_path):
                                with open(train_meta_path, "rb") as f:
                                    st.download_button(
                                        label="📥 Download Train Representations Metadata (JSON)",
                                        data=f,
                                        file_name="train_representations_metadata.json",
                                        mime="application/json"
                                    )

                        # Test representations
                        if os.path.exists(test_representations_path):
                            files_found = True
                            with open(test_representations_path, "rb") as f:
                                st.download_button(
                                    label="📥 Download Test Representations (NPY)",
                                    data=f,
                                    file_name="test_representations.npy",
                                    mime="application/octet-stream"
                                )

                            # Check for metadata JSON
                            test_meta_path = test_representations_path.replace('.npy', '_metadata.json')
                            if os.path.exists(test_meta_path):
                                with open(test_meta_path, "rb") as f:
                                    st.download_button(
                                        label="📥 Download Test Representations Metadata (JSON)",
                                        data=f,
                                        file_name="test_representations_metadata.json",
                                        mime="application/json"
                                    )

                        if not files_found:
                            st.info("No representation files found for this run.")

                    with col2:
                        st.markdown("### Linear Probe Weights")
                        files_found = False

                        # Probe weights JSON metadata
                        if os.path.exists(probe_weights_path):
                            files_found = True
                            with open(probe_weights_path, "rb") as f:
                                st.download_button(
                                    label="📥 Download Probe Weights Metadata (JSON)",
                                    data=f,
                                    file_name="probe_weights.json",
                                    mime="application/json"
                                )

                        # Probe weights PyTorch model
                        if os.path.exists(probe_weights_pt_path):
                            files_found = True
                            with open(probe_weights_pt_path, "rb") as f:
                                st.download_button(
                                    label="📥 Download Probe Models (PyTorch)",
                                    data=f,
                                    file_name="probe_weights.pt",
                                    mime="application/octet-stream"
                                )

                        # Look for individual layer weight files
                        layer_weight_files = []
                        for file in os.listdir(run_folder):
                            if file.startswith("layer_") and file.endswith(".npy"):
                                layer_weight_files.append(file)

                        if layer_weight_files:
                            files_found = True
                            st.markdown("##### Layer-specific Weight Files")

                            # Create a zip file of all layer weight files if there are many
                            if len(layer_weight_files) > 5:
                                import zipfile
                                import io

                                # Create in-memory zip file
                                zip_buffer = io.BytesIO()
                                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                                    for file in layer_weight_files:
                                        file_path = os.path.join(run_folder, file)
                                        zipf.write(file_path, arcname=file)

                                # Reset buffer position
                                zip_buffer.seek(0)

                                # Create download button for zip
                                st.download_button(
                                    label=f"📥 Download All Layer Weights ({len(layer_weight_files)} files)",
                                    data=zip_buffer,
                                    file_name="layer_weights.zip",
                                    mime="application/zip"
                                )
                            else:
                                # If only a few files, provide individual download buttons
                                for file in sorted(layer_weight_files):
                                    file_path = os.path.join(run_folder, file)
                                    with open(file_path, "rb") as f:
                                        st.download_button(
                                            label=f"📥 {file}",
                                            data=f,
                                            file_name=file,
                                            mime="application/octet-stream",
                                            key=f"download_{file}"  # Unique key for each button
                                        )

                        if not files_found:
                            st.info("No probe weight files found for this run.")

                    # Display file information and help text
                    with st.expander("ℹ️ About these data files"):
                        st.markdown("""
                        ### Understanding the Data Files

                        #### Representations (Hidden States)

                        The **representations** are the hidden states from each layer of the model for each input example. These are stored as NumPy arrays (.npy) format:

                        - **Shape**: [num_examples, num_layers, hidden_dimension]
                        - **Usage**: Can be used for further analysis, visualization, or to train new probes

                        #### Linear Probe Weights

                        The **linear probe weights** are the weights learned during the probe training to classify true/false statements:

                        - **JSON file**: Contains metadata about weights and pointers to NPY files
                        - **NPY files**: Each layer's weights as a NumPy array
                        - **PyTorch file (.pt)**: Contains the full probe models in PyTorch format

                        #### How to Use These Files

                        ```python
                        import numpy as np
                        import torch
                        import json

                        # Load representations
                        representations = np.load('test_representations.npy')

                        # Load metadata
                        with open('test_representations_metadata.json', 'r') as f:
                            metadata = json.load(f)

                        # Load individual layer weights
                        layer_0_weights = np.load('layer_0_weights.npy')

                        # Load all probe models (if available)
                        probe_models = torch.load('probe_weights.pt')
                        ```
                        """)

                    # Option to download all data as a single zip
                    st.markdown("### Download Everything")

                    # Check if there are any data files to download
                    data_files = []
                    for file in os.listdir(run_folder):
                        if file.endswith(('.npy', '.json', '.pt')) and not file == "parameters.json" and not file == "results.json":
                            data_files.append(file)

                    if data_files:
                        import zipfile
                        import io

                        # Create in-memory zip file
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                            for file in data_files:
                                file_path = os.path.join(run_folder, file)
                                zipf.write(file_path, arcname=file)

                            # Also include parameters and results
                            if os.path.exists(os.path.join(run_folder, "parameters.json")):
                                zipf.write(os.path.join(run_folder, "parameters.json"), arcname="parameters.json")
                            if os.path.exists(os.path.join(run_folder, "results.json")):
                                zipf.write(os.path.join(run_folder, "results.json"), arcname="results.json")

                        # Reset buffer position
                        zip_buffer.seek(0)

                        # Create download button for zip
                        st.download_button(
                            label=f"📥 Download All Data Files ({len(data_files)+2} files)",
                            data=zip_buffer,
                            file_name=f"{run_id}_all_data.zip",
                            mime="application/zip"
                        )
                    else:
                        st.info("No data files found for this run.")
    else:
        st.info("No saved runs found.")
else:
    st.info("No saved runs found.")
