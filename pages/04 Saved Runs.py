import os
import json
import streamlit as st
from utils.file_manager import SAVED_DATA_DIR
import pandas as pd
import zipfile
import io
import shutil

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

            # Check for essential files
            parameters_path = os.path.join(run_folder, "parameters.json")
            results_path = os.path.join(run_folder, "results.json")

            # Skip this run if essential files don't exist
            if not os.path.exists(parameters_path) or not os.path.exists(results_path):
                st.warning(
                    f"Skipping run {run_id} due to missing essential files")
                continue

            # Load parameters and results
            try:
                with open(parameters_path) as f:
                    parameters = json.load(f)
                with open(results_path) as f:
                    results = json.load(f)
            except Exception as e:
                st.warning(
                    f"Skipping run {run_id} due to error loading files: {str(e)}")
                continue

            # Build a safe display title
            model_name = parameters.get('model_name', 'Unknown Model')
            dataset = parameters.get('dataset', 'Unknown Dataset')
            output_activation = parameters.get(
                'output_activation', 'Unknown Activation')
            datetime = parameters.get('datetime', 'Unknown Date')

            with st.expander(f"📅 {datetime} | 🤖 {model_name} | 📊 {dataset} | 🔍 {output_activation}"):

                # Create tabs for different sections
                run_tabs = st.tabs(
                    ["📋 Overview", "⚙️ Parameters", "📈 Visualizations", "📝 Log", "💾 Data Files"])

                # Overview tab
                with run_tabs[0]:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.subheader("Run Information")
                    with col2:
                        # Create a zip file of the entire run folder
                        try:
                            # Create in-memory zip file of the entire directory
                            zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                                for root, dirs, files in os.walk(run_folder):
                                    for file in files:
                                        file_path = os.path.join(root, file)
                                        # Calculate relative path from run_folder
                                        relative_path = os.path.relpath(file_path, run_folder)
                                        zipf.write(file_path, arcname=relative_path)

                            # Reset buffer position
                            zip_buffer.seek(0)

                            # Create download button for the entire folder
                            st.download_button(
                                label="📥 Download Full Run",
                                data=zip_buffer,
                                file_name=f"{run_id}.zip",
                                mime="application/zip"
                            )
                        except Exception as e:
                            st.warning(f"Error creating download: {str(e)}")

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
                            f"**Model Name:** {parameters.get('model_name', 'Not specified')}")
                        st.markdown(
                            f"**Output Activation:** {parameters.get('output_activation', 'Not specified')}")
                        st.markdown(
                            f"**Device:** {parameters.get('device', 'Not specified')}")

                    # Training parameters
                    with param_cols[1]:
                        st.caption("🧠 PROBE CONFIGURATION")
                        st.markdown(
                            f"**Dataset:** {parameters.get('dataset', 'Not specified')}")
                        st.markdown(
                            f"**Batch Size:** {parameters.get('batch_size', 'Not specified')}")

                        # Handle potential parameter naming differences
                        epochs_key = 'train_epochs' if 'train_epochs' in parameters else 'epochs'
                        st.markdown(
                            f"**Epochs:** {parameters.get(epochs_key, 'Not specified')}")

                        st.markdown(
                            f"**Learning Rate:** {parameters.get('learning_rate', 'Not specified')}")

                        # Safely handle control tasks parameter
                        control_tasks = parameters.get(
                            'use_control_tasks', None)
                        if control_tasks is not None:
                            control_tasks_display = 'Yes' if control_tasks else 'No'
                        else:
                            control_tasks_display = 'Not specified'
                        st.markdown(
                            f"**Control Tasks:** {control_tasks_display}")

                # Visualizations tab
                with run_tabs[2]:
                    # Create sub-tabs for different types of visualizations
                    viz_tabs = st.tabs(
                        ["Linear Probe Results", "Sparse Autoencoder Results"])

                    # Linear Probe Visualizations
                    with viz_tabs[0]:
                        st.subheader("Linear Probe Analysis")
                        plots_found = False

                        # Accuracy plot
                        accuracy_plot_path = os.path.join(
                            run_folder, "accuracy_plot.png")
                        if os.path.exists(accuracy_plot_path):
                            plots_found = True
                            st.caption("📈 ACCURACY PLOT")
                            st.image(accuracy_plot_path,
                                     use_container_width=True)

                        # Alignment Strength plot
                        alignment_strength_plot_path = os.path.join(
                            run_folder, "alignment_strength_plot.png")
                        if os.path.exists(alignment_strength_plot_path):
                            plots_found = True
                            st.caption("🔗 ALIGNMENT STRENGTH BY LAYER")
                            st.image(alignment_strength_plot_path,
                                     use_container_width=True)

                        # PCA plot
                        pca_plot_path = os.path.join(
                            run_folder, "pca_plot.png")
                        if os.path.exists(pca_plot_path):
                            plots_found = True
                            st.caption("🔍 PCA VISUALIZATION")
                            st.image(pca_plot_path, use_container_width=True)

                        # Truth direction plot
                        truth_direction_plot_path = os.path.join(
                            run_folder, "proj_plot.png")
                        if os.path.exists(truth_direction_plot_path):
                            plots_found = True
                            st.caption("🧭 TRUTH DIRECTION PLOT")
                            st.image(truth_direction_plot_path,
                                     use_container_width=True)

                        if not plots_found:
                            st.info("No linear probe plots found for this run.")

                    # Sparse Autoencoder Visualizations
                    with viz_tabs[1]:
                        st.subheader("Sparse Autoencoder Analysis")
                        sae_content_found = False

                        # Check if this run has autoencoder results
                        autoencoder_stats_path = os.path.join(
                            run_folder, "autoencoder_stats.json")
                        sparsity_plot_path = os.path.join(
                            run_folder, "sparsity_plot.png")
                        l1_sparsity_plot_path = os.path.join(
                            run_folder, "l1_sparsity_plot.png")
                        reconstruction_error_plot_path = os.path.join(
                            run_folder, "reconstruction_error_plot.png")

                        if os.path.exists(autoencoder_stats_path):
                            sae_content_found = True
                            # Load and display autoencoder stats
                            try:
                                with open(autoencoder_stats_path) as f:
                                    autoencoder_stats = json.load(f)

                                # Display basic info
                                st.caption("🔧 AUTOENCODER CONFIGURATION")
                                config_cols = st.columns(3)
                                with config_cols[0]:
                                    st.metric("L1 Coefficient", autoencoder_stats.get(
                                        "l1_coefficient", "N/A"))
                                    st.metric("Type", autoencoder_stats.get(
                                        "autoencoder_type", "N/A"))
                                with config_cols[1]:
                                    bottleneck = autoencoder_stats.get(
                                        "bottleneck_dim", "N/A")
                                    bottleneck_display = "Same as input" if bottleneck == 0 else bottleneck
                                    st.metric("Hidden Dimension",
                                              bottleneck_display)
                                    st.metric("Tied Weights", "Yes" if autoencoder_stats.get(
                                        "tied_weights", False) else "No")
                                with config_cols[2]:
                                    st.metric("Epochs", autoencoder_stats.get(
                                        "training_epochs", "N/A"))
                                    st.metric("Learning Rate", autoencoder_stats.get(
                                        "learning_rate", "N/A"))
                            except Exception as e:
                                st.warning(
                                    f"Error loading autoencoder stats: {str(e)}")

                            # Display sparsity plot
                            if os.path.exists(sparsity_plot_path):
                                st.caption("📊 SPARSITY PERCENTAGE BY LAYER")
                                st.image(sparsity_plot_path,
                                         use_container_width=True)

                            # Display L1 sparsity plot
                            if os.path.exists(l1_sparsity_plot_path):
                                st.caption("📉 L1 SPARSITY MEASURE BY LAYER")
                                st.image(l1_sparsity_plot_path,
                                         use_container_width=True)

                            # Display reconstruction error plot
                            if os.path.exists(reconstruction_error_plot_path):
                                st.caption("🔄 RECONSTRUCTION ERROR BY LAYER")
                                st.image(reconstruction_error_plot_path,
                                         use_container_width=True)

                            # Create expandable section to show raw data if available
                            try:
                                # Check if we have dimension data
                                has_dimension_data = "layer_dimensions" in autoencoder_stats and "input_dimensions" in autoencoder_stats

                                if has_dimension_data or "sparsity_values" in autoencoder_stats or "reconstruction_errors" in autoencoder_stats:
                                    with st.expander("View Detailed Autoencoder Statistics"):
                                        # If we have layer dimensions, show them in a separate section
                                        if has_dimension_data:
                                            st.subheader("Autoencoder Dimensions")
                                            dimension_data = []
                                            for i in range(len(autoencoder_stats["layer_dimensions"])):
                                                input_dim = autoencoder_stats["input_dimensions"][i]
                                                latent_dim = autoencoder_stats["layer_dimensions"][i]
                                                ratio = latent_dim / input_dim if input_dim > 0 else 0
                                                dimension_data.append({
                                                    'Layer': i,
                                                    'Input Dimension': input_dim,
                                                    'Latent Dimension': latent_dim,
                                                    'Ratio (Latent/Input)': f"{ratio:.2f}x",
                                                    'Type': 'Overcomplete' if ratio > 1 else ('Undercomplete' if ratio < 1 else 'Same Dimension')
                                                })

                                            dimension_df = pd.DataFrame(dimension_data)
                                            st.dataframe(dimension_df)

                                            # Calculate total parameters
                                            tied_weights = autoencoder_stats.get("tied_weights", True)
                                            total_params = sum(input_dim * latent_dim * (1 if tied_weights else 2)
                                                             for input_dim, latent_dim in zip(autoencoder_stats["input_dimensions"], autoencoder_stats["layer_dimensions"]))
                                            st.info(f"Total autoencoder parameters: {total_params:,} " +
                                                   f"(with {'tied' if tied_weights else 'untied'} weights)")

                                            st.markdown("---")
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
                                                st.info(
                                                    "No sparsity values available")

                                        with col2:
                                            st.subheader(
                                                "Reconstruction Errors")
                                            if "reconstruction_errors" in autoencoder_stats:
                                                recon_df = pd.DataFrame({
                                                    "Layer": range(len(autoencoder_stats["reconstruction_errors"])),
                                                    "MSE": autoencoder_stats["reconstruction_errors"]
                                                })
                                                st.dataframe(recon_df)
                                            else:
                                                st.info(
                                                    "No reconstruction error values available")
                            except Exception as e:
                                st.warning(
                                    f"Error displaying sparsity data: {str(e)}")

                        # Check for feature grid data
                        feature_grid_dir = os.path.join(run_folder, "feature_grids")
                        if os.path.exists(feature_grid_dir) and os.path.isdir(feature_grid_dir):
                            sae_content_found = True

                            # Display feature grid data
                            st.markdown("### Feature Grid Analysis")
                            st.markdown("This shows examples that most strongly activate each feature in the sparse autoencoder.")

                            # Get all feature grid files
                            feature_grid_files = sorted(
                                [f for f in os.listdir(feature_grid_dir) if f.startswith("layer_") and f.endswith("_feature_grid.json")],
                                key=lambda x: int(x.split("_")[1])  # Sort by layer number
                            )

                            if feature_grid_files:
                                # Create tabs for each layer
                                grid_tabs = st.tabs([f"Layer {f.split('_')[1]}" for f in feature_grid_files])

                                for i, grid_file in enumerate(feature_grid_files):
                                    with grid_tabs[i]:
                                        layer_num = grid_file.split("_")[1]
                                        file_path = os.path.join(feature_grid_dir, grid_file)

                                        try:
                                            with open(file_path, 'r') as f:
                                                feature_data = json.load(f)

                                                # Group features into sets of 10 (or fewer for the last set)
                                                feature_sets = []
                                                current_set = []

                                                for feature_info in feature_data:
                                                    current_set.append(feature_info)
                                                    if len(current_set) == 10:
                                                        feature_sets.append(current_set)
                                                        current_set = []

                                                # Add any remaining features
                                                if current_set:
                                                    feature_sets.append(current_set)

                                                # Create page tabs if we have many features
                                                if len(feature_sets) > 1:
                                                    page_tabs = st.tabs([f"Features {i*10+1}-{min((i+1)*10, len(feature_data))}" for i in range(len(feature_sets))])

                                                    for page_idx, feature_set in enumerate(feature_sets):
                                                        with page_tabs[page_idx]:
                                                            # Create tabs for each feature in this set
                                                            feature_tabs = st.tabs([f"Feature {f['feature_idx']} | Act: {f['mean_activation']:.4f}" for f in feature_set])

                                                            for feat_idx, feature_info in enumerate(feature_set):
                                                                with feature_tabs[feat_idx]:
                                                                    feature_idx = feature_info['feature_idx']
                                                                    mean_activation = feature_info['mean_activation']
                                                                    top_examples = feature_info['top_examples']

                                                                    if top_examples:
                                                                        # With up to 10 examples, use a multi-column layout for efficiency
                                                                        if len(top_examples) > 6:
                                                                            # For 7-10 examples, use a three-column layout
                                                                            cols = st.columns(3)
                                                                            for j, example in enumerate(top_examples):
                                                                                col_idx = j % 3  # Distribute among 3 columns
                                                                                with cols[col_idx]:
                                                                                    st.markdown(f"**Ex {j+1}** (Act: {example['activation']:.4f})")
                                                                                    st.markdown(f"> {example['text']}")
                                                                                    st.divider()
                                                                        elif len(top_examples) > 2:
                                                                            # For 3-6 examples, use a two-column layout
                                                                            col1, col2 = st.columns(2)
                                                                            for j, example in enumerate(top_examples):
                                                                                if j % 2 == 0:  # Even examples in left column
                                                                                    with col1:
                                                                                        st.markdown(f"**Example {j+1}** (Activation: {example['activation']:.4f})")
                                                                                        st.markdown(f"> {example['text']}")
                                                                                        st.divider()
                                                                                else:  # Odd examples in right column
                                                                                    with col2:
                                                                                        st.markdown(f"**Example {j+1}** (Activation: {example['activation']:.4f})")
                                                                                        st.markdown(f"> {example['text']}")
                                                                                        st.divider()
                                                                        else:
                                                                            # For 1-2 examples, use single column layout
                                                                            for j, example in enumerate(top_examples):
                                                                                st.markdown(f"**Example {j+1}** (Activation: {example['activation']:.4f})")
                                                                                st.markdown(f"> {example['text']}")
                                                                                st.divider()
                                                                    else:
                                                                        st.info("No examples with positive activation found for this feature.")
                                                else:
                                                    # If few features, just create tabs directly
                                                    feature_tabs = st.tabs([f"Feature {f['feature_idx']} | Act: {f['mean_activation']:.4f}" for f in feature_data])

                                                    for feat_idx, feature_info in enumerate(feature_data):
                                                        with feature_tabs[feat_idx]:
                                                            feature_idx = feature_info['feature_idx']
                                                            mean_activation = feature_info['mean_activation']
                                                            top_examples = feature_info['top_examples']

                                                            if top_examples:
                                                                # With up to 10 examples, use a multi-column layout for efficiency
                                                                if len(top_examples) > 6:
                                                                    # For 7-10 examples, use a three-column layout
                                                                    cols = st.columns(3)
                                                                    for j, example in enumerate(top_examples):
                                                                        col_idx = j % 3  # Distribute among 3 columns
                                                                        with cols[col_idx]:
                                                                            st.markdown(f"**Ex {j+1}** (Act: {example['activation']:.4f})")
                                                                            st.markdown(f"> {example['text']}")
                                                                            st.divider()
                                                                elif len(top_examples) > 2:
                                                                    # For 3-6 examples, use a two-column layout
                                                                    col1, col2 = st.columns(2)
                                                                    for j, example in enumerate(top_examples):
                                                                        if j % 2 == 0:  # Even examples in left column
                                                                            with col1:
                                                                                st.markdown(f"**Example {j+1}** (Activation: {example['activation']:.4f})")
                                                                                st.markdown(f"> {example['text']}")
                                                                                st.divider()
                                                                        else:  # Odd examples in right column
                                                                            with col2:
                                                                                st.markdown(f"**Example {j+1}** (Activation: {example['activation']:.4f})")
                                                                                st.markdown(f"> {example['text']}")
                                                                                st.divider()
                                                                else:
                                                                    # For 1-2 examples, use single column layout
                                                                    for j, example in enumerate(top_examples):
                                                                        st.markdown(f"**Example {j+1}** (Activation: {example['activation']:.4f})")
                                                                        st.markdown(f"> {example['text']}")
                                                                        st.divider()
                                                            else:
                                                                st.info("No examples with positive activation found for this feature.")
                                        except Exception as e:
                                            st.warning(f"Error loading feature grid data: {str(e)}")
                            else:
                                st.info("No feature grid data files found")

                        if not sae_content_found:
                            st.info(
                                "No sparse autoencoder analysis was performed for this run.")

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
                                    layer_viz_found = False

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
                                        layer_viz_found = True
                                        st.image(
                                            probe_weights_path, caption="Probe Neuron Weights", use_container_width=True)
                                    if os.path.exists(activation_diff_path):
                                        layer_viz_found = True
                                        st.image(
                                            activation_diff_path, caption="Mean Activation Difference (True-False)", use_container_width=True)
                                    if os.path.exists(truth_proj_path):
                                        layer_viz_found = True
                                        st.image(
                                            truth_proj_path, caption="Truth Direction Projection", use_container_width=True)
                                    if os.path.exists(conf_matrix_path):
                                        layer_viz_found = True
                                        st.image(
                                            conf_matrix_path, caption="Confusion Matrix", use_container_width=True)
                                    if os.path.exists(neuron_alignment_path):
                                        layer_viz_found = True
                                        st.image(
                                            neuron_alignment_path, caption="Neuron Alignment (Weight vs. Activation Diff)", use_container_width=True)

                                    if not layer_viz_found:
                                        st.info(
                                            f"No visualizations found for Layer {layer_num_str}")

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
                        try:
                            with open(log_file_path, "rb") as f:
                                st.download_button(
                                    label="📥 Download Log File",
                                    data=f,
                                    file_name="log.txt",
                                    mime="text/plain"
                                )
                        except Exception as e:
                            st.warning(f"Error loading log file: {str(e)}")
                    else:
                        st.info("No log file found for this run.")

                # Data Files tab
                with run_tabs[4]:
                    st.subheader("Download Data Files")

                    # Check for representations
                    train_representations_path = os.path.join(
                        run_folder, "train_representations.npy")
                    test_representations_path = os.path.join(
                        run_folder, "test_representations.npy")
                    probe_weights_path = os.path.join(
                        run_folder, "probe_weights.json")
                    probe_weights_pt_path = os.path.join(
                        run_folder, "probe_weights.pt")

                    # Create columns for better layout
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### Representations")
                        files_found = False

                        # Train representations
                        if os.path.exists(train_representations_path):
                            files_found = True
                            try:
                                with open(train_representations_path, "rb") as f:
                                    st.download_button(
                                        label="📥 Download Train Representations (NPY)",
                                        data=f,
                                        file_name="train_representations.npy",
                                        mime="application/octet-stream"
                                    )
                            except Exception as e:
                                st.warning(
                                    f"Error loading train representations: {str(e)}")

                            # Check for metadata JSON
                            train_meta_path = train_representations_path.replace(
                                '.npy', '_metadata.json')
                            if os.path.exists(train_meta_path):
                                try:
                                    with open(train_meta_path, "rb") as f:
                                        st.download_button(
                                            label="📥 Download Train Representations Metadata (JSON)",
                                            data=f,
                                            file_name="train_representations_metadata.json",
                                            mime="application/json"
                                        )
                                except Exception as e:
                                    st.warning(
                                        f"Error loading train metadata: {str(e)}")

                        # Test representations
                        if os.path.exists(test_representations_path):
                            files_found = True
                            try:
                                with open(test_representations_path, "rb") as f:
                                    st.download_button(
                                        label="📥 Download Test Representations (NPY)",
                                        data=f,
                                        file_name="test_representations.npy",
                                        mime="application/octet-stream"
                                    )
                            except Exception as e:
                                st.warning(
                                    f"Error loading test representations: {str(e)}")

                            # Check for metadata JSON
                            test_meta_path = test_representations_path.replace(
                                '.npy', '_metadata.json')
                            if os.path.exists(test_meta_path):
                                try:
                                    with open(test_meta_path, "rb") as f:
                                        st.download_button(
                                            label="📥 Download Test Representations Metadata (JSON)",
                                            data=f,
                                            file_name="test_representations_metadata.json",
                                            mime="application/json"
                                        )
                                except Exception as e:
                                    st.warning(
                                        f"Error loading test metadata: {str(e)}")

                        if not files_found:
                            st.info(
                                "No representation files found for this run.")

                    with col2:
                        st.markdown("### Linear Probe Weights")
                        files_found = False

                        # Probe weights JSON metadata
                        if os.path.exists(probe_weights_path):
                            files_found = True
                            try:
                                with open(probe_weights_path, "rb") as f:
                                    st.download_button(
                                        label="📥 Download Probe Weights Metadata (JSON)",
                                        data=f,
                                        file_name="probe_weights.json",
                                        mime="application/json"
                                    )
                            except Exception as e:
                                st.warning(
                                    f"Error loading probe weights metadata: {str(e)}")

                        # Probe weights PyTorch model
                        if os.path.exists(probe_weights_pt_path):
                            files_found = True
                            try:
                                with open(probe_weights_pt_path, "rb") as f:
                                    st.download_button(
                                        label="📥 Download Probe Models (PyTorch)",
                                        data=f,
                                        file_name="probe_weights.pt",
                                        mime="application/octet-stream"
                                    )
                            except Exception as e:
                                st.warning(
                                    f"Error loading probe models: {str(e)}")

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
                                try:
                                    import zipfile
                                    import io

                                    # Create in-memory zip file
                                    zip_buffer = io.BytesIO()
                                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                                        for file in layer_weight_files:
                                            file_path = os.path.join(
                                                run_folder, file)
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
                                except Exception as e:
                                    st.warning(
                                        f"Error creating layer weights zip: {str(e)}")
                                    # Fallback to individual downloads if zip creation fails
                                    # Show first 5 only as fallback
                                    for file in sorted(layer_weight_files)[:5]:
                                        try:
                                            file_path = os.path.join(
                                                run_folder, file)
                                            with open(file_path, "rb") as f:
                                                st.download_button(
                                                    label=f"📥 {file}",
                                                    data=f,
                                                    file_name=file,
                                                    mime="application/octet-stream",
                                                    # Unique key for each button
                                                    key=f"download_{file}"
                                                )
                                        except Exception as e:
                                            st.warning(
                                                f"Error loading {file}: {str(e)}")
                            else:
                                # If only a few files, provide individual download buttons
                                for file in sorted(layer_weight_files):
                                    try:
                                        file_path = os.path.join(
                                            run_folder, file)
                                        with open(file_path, "rb") as f:
                                            st.download_button(
                                                label=f"📥 {file}",
                                                data=f,
                                                file_name=file,
                                                mime="application/octet-stream",
                                                # Unique key for each button
                                                key=f"download_{file}"
                                            )
                                    except Exception as e:
                                        st.warning(
                                            f"Error loading {file}: {str(e)}")

                        if not files_found:
                            st.info("No probe weight files found for this run.")

                    # Display file information and help text
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
                        try:
                            import zipfile
                            import io

                            # Create in-memory zip file
                            zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                                for file in data_files:
                                    file_path = os.path.join(run_folder, file)
                                    try:
                                        zipf.write(file_path, arcname=file)
                                    except Exception as e:
                                        st.warning(
                                            f"Error adding {file} to zip: {str(e)}")

                                # Also include parameters and results
                                if os.path.exists(os.path.join(run_folder, "parameters.json")):
                                    zipf.write(os.path.join(
                                        run_folder, "parameters.json"), arcname="parameters.json")
                                if os.path.exists(os.path.join(run_folder, "results.json")):
                                    zipf.write(os.path.join(
                                        run_folder, "results.json"), arcname="results.json")

                            # Reset buffer position
                            zip_buffer.seek(0)

                            # Create download button for zip
                            st.download_button(
                                label=f"📥 Download All Data Files ({len(data_files)+2} files)",
                                data=zip_buffer,
                                file_name=f"{run_id}_all_data.zip",
                                mime="application/zip"
                            )
                        except Exception as e:
                            st.warning(f"Error creating zip file: {str(e)}")
                            # Offer individual downloads for important files as fallback
                            st.markdown(
                                "Could not create zip file. Try downloading individual files from the sections above.")
                    else:
                        st.info("No data files found for this run.")
    else:
        st.info("No saved runs found.")
else:
    st.info("No saved runs found.")
