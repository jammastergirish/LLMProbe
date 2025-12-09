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

quick_view = st.checkbox("⚡ Enable Quick View Mode (Skip loading large visualizations)", value=True)

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
                        # Only create the zip file when requested with a button
                        if st.button("📥 Prepare Full Run Download", key=f"prepare_zip_{run_id}"):
                            with st.spinner("Creating zip file of the entire run folder..."):
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
                        else:
                            st.info("Click the button above to prepare the download of the entire run folder.")

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

                        # Display TrueFalse categories if available and dataset is truefalse
                        if parameters.get('dataset') == 'truefalse' and 'truefalse_categories' in parameters:
                            truefalse_categories = parameters.get('truefalse_categories', [])
                            if truefalse_categories:
                                st.markdown("**TrueFalse Categories:**")
                                categories_str = ", ".join(truefalse_categories)
                                st.markdown(f"- {categories_str}")

                # Visualizations tab
                with run_tabs[2]:
                    # Create sub-tabs for different types of visualizations
                    viz_tabs = st.tabs(
                        ["Linear Probe Results"])

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
                            # If quick_view is disabled or button is pressed, load visualizations
                            if not quick_view or st.button("🔍 Load Per-Layer Visualizations", key=f"load_layer_viz_{run_id}"):
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
                                # Show summary of available per-layer visualizations without loading them
                                st.info(f"Found per-layer visualizations for {len(layer_subdirs)} layers. Click the button above to load them.")
                                # Show a list of available layers
                                st.markdown("**Available layers:** " + ", ".join([f"Layer {d}" for d in layer_subdirs]))
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
                        st.write(f"Found {len(data_files)} data files that can be downloaded.")

                        if st.button("📦 Prepare Data Files Download", key=f"prepare_data_files_{run_id}"):
                            with st.spinner(f"Creating zip with {len(data_files)} data files..."):
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
