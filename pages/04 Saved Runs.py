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
    else:
        st.info("No saved runs found.")
else:
    st.info("No saved runs found.")
