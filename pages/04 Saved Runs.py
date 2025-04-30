import os
import json
import streamlit as st
from utils.file_manager import SAVED_DATA_DIR
import pandas as pd

st.set_page_config(page_title="Saved Runs", layout="wide")

st.title("📊 Saved Runs")

# List all saved runs
if os.path.exists(SAVED_DATA_DIR):
    run_folders = [f for f in os.listdir(
        SAVED_DATA_DIR) if os.path.isdir(os.path.join(SAVED_DATA_DIR, f))]
    if run_folders:
        for run_id in run_folders:
            run_folder = os.path.join(SAVED_DATA_DIR, run_id)

            # Load parameters and results
            with open(os.path.join(run_folder, "parameters.json")) as f:
                parameters = json.load(f)
            with open(os.path.join(run_folder, "results.json")) as f:
                results = json.load(f)

            # Display run details
        with st.expander(f"📅 {parameters['datetime']} | 🤖 {parameters['model_name']} | 📊 {parameters['dataset']} | 🔍 {parameters['output_activation']}"):

            st.json(parameters)
            # Create tabs for different sections
            run_tabs = st.tabs(["📋 Overview", "⚙️ Parameters", "📈 Visualizations"])

            # Overview tab
            with run_tabs[0]:
                st.markdown(f"""
                <div style="padding: 15px; border-radius: 5px; background-color: #f0f5ff; border-left: 5px solid #4361ee;">
                    <h3 style="margin-top: 0; color: #4361ee;">Run Information</h3>
                    <p><strong>ID:</strong> {run_id}</p>
                    <p><strong>Date:</strong> {parameters['datetime']}</p>
                    <p><strong>Model:</strong> {parameters['model_name']}</p>
          
                </div>
                """, unsafe_allow_html=True)

                #           <p><strong>Best Layer:</strong> {results['best_layer']} (Accuracy: {results['best_accuracy']:.4f})</p>

            # Parameters tab
            with run_tabs[1]:
                # Create a more visually appealing parameter display
                st.markdown(
                    "<h3 style='text-align: center; color: #333;'>Configuration Parameters</h3>", unsafe_allow_html=True)

                # Create columns for parameters
                param_cols = st.columns(2)

                # Model parameters
                with param_cols[0]:
                    st.markdown("""
                    <div style="padding: 10px; border-radius: 5px; background-color: #f0f8ff; margin-bottom: 10px;">
                        <h4 style="margin-top: 0;">🤖 Model Configuration</h4>
                    </div>
                    """, unsafe_allow_html=True)

                    # Format the model parameters in a more readable way
                    st.markdown(f"**Model Name:** {parameters['model_name']}")
                    st.markdown(
                        f"**Output Activation:** {parameters['output_activation']}")
                    st.markdown(
                        f"**Device:** {parameters.get('device', 'Not specified')}")

                # Training parameters
                with param_cols[1]:
                    st.markdown("""
                    <div style="padding: 10px; border-radius: 5px; background-color: #f0fff4; margin-bottom: 10px;">
                        <h4 style="margin-top: 0;">🧠 Training Configuration</h4>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"**Dataset:** {parameters['dataset']}")
                    st.markdown(f"**Batch Size:** {parameters['batch_size']}")
                    st.markdown(f"**Epochs:** {parameters['train_epochs']}")
                    st.markdown(f"**Learning Rate:** {parameters['learning_rate']}")
                    st.markdown(
                        f"**Control Tasks:** {'Yes' if parameters['use_control_tasks'] else 'No'}")

            # Visualizations tab
            with run_tabs[2]:
                st.markdown(
                    "<h3 style='text-align: center; color: #333;'>Analysis Visualizations</h3>", unsafe_allow_html=True)

                # Create columns for the images
                vis_cols = st.columns(1)

                # Accuracy plot
                accuracy_plot_path = os.path.join(run_folder, "accuracy_plot.png")
                if os.path.exists(accuracy_plot_path):
                    with vis_cols[0]:
                        st.markdown("""
                        <div style="padding: 10px; border-radius: 5px; background-color: #fff0f0; margin-bottom: 10px;">
                            <h4 style="margin-top: 0;">📈 Accuracy Plot</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        st.image(accuracy_plot_path, use_column_width=True)

                # PCA plot
                pca_plot_path = os.path.join(run_folder, "pca_plot.png")
                if os.path.exists(pca_plot_path):
                    with vis_cols[0]:
                        st.markdown("""
                        <div style="padding: 10px; border-radius: 5px; background-color: #f0f0ff; margin-bottom: 10px; margin-top: 20px;">
                            <h4 style="margin-top: 0;">🔍 PCA Visualization</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        st.image(pca_plot_path, use_column_width=True)

                # Truth direction plot
                truth_direction_plot_path = os.path.join(run_folder, "proj_plot.png")
                if os.path.exists(truth_direction_plot_path):
                    with vis_cols[0]:
                        st.markdown("""
                        <div style="padding: 10px; border-radius: 5px; background-color: #fffcf0; margin-bottom: 10px; margin-top: 20px;">
                            <h4 style="margin-top: 0;">🧭 Truth Direction Plot</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        st.image(truth_direction_plot_path, use_column_width=True)
    else:
        st.info("No saved runs found.")
else:
    st.info("No saved runs found.")
