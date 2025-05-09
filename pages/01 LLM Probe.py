import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import gc
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
import nest_asyncio

# Import from refactored modules
from utils.models import model_options
from utils.file_manager import create_run_folder, save_json, save_graph
from utils.memory import estimate_memory_requirements
from utils.load import (
    load_model_and_tokenizer, 
    load_dataset, 
    get_hidden_states_batched, 
    is_decoder_only_model, 
    get_num_layers
)
from utils.probe import (
    LinearProbe,
    train_and_evaluate_model,
    calculate_mean_activation_difference,
    calculate_alignment_strengths,
    get_top_k_neurons,
    calculate_confusion_matrix,
    create_metrics_dataframe,
    plot_truth_direction_projection,
    plot_confusion_matrix,
    plot_probe_weights
)
from utils.ui import (
    create_model_tracker,
    create_dataset_tracker,
    create_embedding_tracker,
    create_training_tracker
)
from utils.visualizations import (
    plot_accuracy_by_layer,
    plot_selectivity_by_layer,
    plot_pca_grid,
    plot_truth_projections,
    plot_neuron_alignment,
    plot_alignment_strength_by_layer
)

warnings.filterwarnings('ignore')
nest_asyncio.apply()

st.set_page_config(page_title="LLMProbe", layout="wide")

st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FFFFFF;
        padding-bottom: 1rem;
        border-bottom: 2px solid #FFFFFF;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.2rem;
        font-weight: 400;
        color: #FFFFFF;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #FFFFFF;
        padding-top: 1rem;
        border-top: 1px solid #FFFFFF;
        margin-top: 1.5rem;
    }
    .info-text {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #FFFFFF;
    }
    .status-success {
        color: #2e7d32;
        font-weight: 600;
    }
    .status-running {
        color: #f57c00;
        font-weight: 600;
    }
    .status-idle {
        color: #757575;
        font-weight: 400;
    }
</style>

<div class="main-title">Probing Large Language Models</div>""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="padding: 5px; border-radius: 5px; margin-bottom: 20px;">
    <h2 style="color: white; margin: 0;">Configuration</h2>
</div>
""", unsafe_allow_html=True)

model_name = st.sidebar.selectbox("📚 Model", model_options)

if model_name == "custom":
    model_name = st.sidebar.text_input("Custom Model Name")
    if not model_name:
        st.sidebar.error("Please enter a model.")

dataset_source = st.sidebar.selectbox(" 📊 Dataset",
                                      ["truefalse", "truthfulqa", "boolq", "arithmetic", "fever", "custom"])

all_tf_splits = [
    "animals", "cities", "companies",
    "inventions", "facts", "elements", "generated"
]

if dataset_source == "truefalse":
    selected_tf_splits = st.sidebar.multiselect(
        "Select TrueFalse dataset categories",
        options=all_tf_splits,
        default=all_tf_splits
    )
    tf_splits = selected_tf_splits
else:
    tf_splits = all_tf_splits

if dataset_source == "custom":
    custom_file = st.sidebar.file_uploader(
        "Upload CSV file with 'statement' and 'label' (containing 1 or 0) columns",
        type=["csv"],
        help="CSV should have 'statement' column for text and 'label' column with 1 (true) or 0 (false)"
    )

    # Preview of uploaded data
    if custom_file is not None:
        try:
            df_preview = pd.read_csv(custom_file)
            if 'statement' not in df_preview.columns or 'label' not in df_preview.columns:
                st.sidebar.error(
                    "CSV must contain 'statement' and 'label' columns")
            else:
                st.sidebar.success(f"Loaded {len(df_preview)} examples")
                st.sidebar.dataframe(df_preview.head(
                    3), use_container_width=True)
                # Reset file pointer for later use
                custom_file.seek(0)
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {str(e)}")

use_control_tasks = st.sidebar.checkbox("Use control tasks", value=True)

if is_decoder_only_model(model_name):
    output_layer = st.sidebar.selectbox(
        "🧠 Output Activation", ["resid_post", "attn_out", "mlp_out"])
else:
    output_layer = st.sidebar.selectbox(
        "🧠 Embedding Strategy", ["CLS", "mean", "max", "token_index_0"])

# Device selection
device_options = []
if torch.cuda.is_available():
    device_options.append("cuda")
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device_options.append("mps")
device_options.append("cpu")

device_name = st.sidebar.selectbox("💻 Compute", device_options)
device = torch.device(device_name)

with st.sidebar.expander("⚙️ Probe Options"):
    train_epochs = st.number_input(
        "Training epochs", min_value=10, max_value=500, value=100)
    learning_rate = st.number_input(
        "Learning rate", min_value=0.0001, max_value=0.1, value=0.01, format="%.4f")
    max_samples = st.number_input(
        "Max samples per dataset", min_value=100, max_value=10000, value=5000)
    test_size = st.slider("Test split ratio", min_value=0.1,
                          max_value=0.5, value=0.2, step=0.05)
    batch_size = st.number_input("Batch size", min_value=1, max_value=64, value=16,
                                 help="Larger batches are faster but use more memory. Use smaller values for large models.")

run_button = st.sidebar.button(
    "🚀 Run Analysis", type="primary", use_container_width=True)

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("""
    <div class="section-header">Model Configuration</div>
    """, unsafe_allow_html=True)

    # Convert device to string to avoid PyArrow error
    config_df = pd.DataFrame({
        'Parameter': ['Model', 'Dataset', 'Control Tasks', 'Output Layer', 'Device'],
        'Value': [model_name, dataset_source, str(use_control_tasks), output_layer, str(device)]
    })
    st.table(config_df)

with col2:
    st.markdown("""
    <div class="section-header">Statistics</div>
    """, unsafe_allow_html=True)

    # Create placeholder for stats that will be filled later
    stats_placeholder = st.empty()
    stats_placeholder.info("Statistics will appear when analysis runs")
    with st.expander("📚 Understanding Memory Requirements"):
        st.markdown("""
        ### How Memory is Calculated

        - **Parameter Memory**: Calculated as `number of parameters × bytes per parameter`
        - **Activation Memory**: Calculated as `batch_size × sequence_length × hidden_dimension × number_of_layers × bytes_per_value`
        - **Total Memory**: Sum of parameter and activation memory, with a 20% overhead factor

        Larger batch sizes and sequence lengths will significantly increase memory usage. Consider reducing these values if you encounter out-of-memory errors.
        """)

# Create columns for progress indicators
progress_col1, progress_col2 = st.columns(2)

with progress_col1:
    # st.markdown('<div class="progress-container">', unsafe_allow_html=True)
    st.markdown('#### 📚 Load Model')
    model_status = st.empty()
    model_status.markdown(
        '<span class="status-idle">Waiting to start...</span>', unsafe_allow_html=True)
    model_progress_bar = st.progress(0)
    model_progress_text = st.empty()
    model_detail = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

    # st.markdown('<div class="progress-container">', unsafe_allow_html=True)
    st.markdown('#### 🔍 Create Representations')
    embedding_status = st.empty()
    embedding_status.markdown(
        '<span class="status-idle">Waiting to start...</span>', unsafe_allow_html=True)
    embedding_progress_bar = st.progress(0)
    embedding_progress_text = st.empty()
    embedding_detail = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

with progress_col2:
    # st.markdown('<div class="progress-container">', unsafe_allow_html=True)
    st.markdown('#### 📊 Load Dataset')
    dataset_status = st.empty()
    dataset_status.markdown(
        '<span class="status-idle">Waiting to start...</span>', unsafe_allow_html=True)
    dataset_progress_bar = st.progress(0)
    dataset_progress_text = st.empty()
    dataset_detail = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

    # st.markdown('<div class="progress-container">', unsafe_allow_html=True)
    st.markdown('#### 🧠 Train Probe')
    training_status = st.empty()
    training_status.markdown(
        '<span class="status-idle">Waiting to start...</span>', unsafe_allow_html=True)
    training_progress_bar = st.progress(0)
    training_progress_text = st.empty()
    training_detail = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

# Log area
with st.expander("📋 Detailed Log", expanded=False):
    log_container = st.container()
    log_placeholder = log_container.empty()
    log_text = []

    def add_log(message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_text.append(f"[{timestamp}] {message}")
        log_placeholder.code("\n".join(log_text), language="")

# Results area
st.markdown("""
<div class="section-header">Results</div>
""", unsafe_allow_html=True)

# Create Main Tabs
main_tabs = st.tabs(
    ["Probe Analysis", "Sparse Autoencoder Analysis (Coming Soon)"])

# Setup Probe Analysis Sub-Tabs and placeholders
with main_tabs[0]:
    probe_tabs = st.tabs(["Accuracy Analysis", "PCA Visualization",
                          "Truth Direction Analysis", "Data View"])
    accuracy_tab_container = probe_tabs[0]
    pca_tab_container = probe_tabs[1]
    projection_tab_container = probe_tabs[2]
    data_tab_container = probe_tabs[3]

    # Define empty containers within the sub-tabs for later population
    with accuracy_tab_container:
        accuracy_plot = st.empty()
        selectivity_plot = st.empty()
    with pca_tab_container:
        pca_plot = st.empty()
    with projection_tab_container:
        projection_plot = st.empty()
    with data_tab_container:
        # data_display = st.empty() # Content will be added directly later
        pass  # Data view content is complex, added dynamically

# Placeholder for the second main tab
with main_tabs[1]:
    st.info("Analysis for Sparse Autoencoders will be added here.")

# Create progress trackers using the UI module
model_tracker = create_model_tracker(model_status, model_progress_bar, model_progress_text, model_detail, add_log)
dataset_tracker = create_dataset_tracker(dataset_status, dataset_progress_bar, dataset_progress_text, dataset_detail, add_log)
embedding_tracker = create_embedding_tracker(embedding_status, embedding_progress_bar, embedding_progress_text, embedding_detail, add_log)
training_tracker = create_training_tracker(training_status, training_progress_bar, training_progress_text, training_detail, add_log)

def mark_complete(status_element, message="Complete"):
    """Mark this stage as complete"""
    status_element.markdown(
        f'<span class="status-success">{message}</span>', unsafe_allow_html=True)


def save_fig(fig, filename):
    """Save figure to disk"""
    fig.savefig(filename)
    add_log(f"Saved figure to {filename}")


# Main app logic
if run_button:
    run_folder, run_id = create_run_folder(
        model_name=model_name, dataset=dataset_source)

    # Reset progress displays
    add_log(
        f"Starting analysis with model: {model_name}, dataset: {dataset_source}")

    initial_stats_df = pd.DataFrame({
        'Statistic': [
            'Model',
            'Dataset',
            'Compute Device',
            'Batch Size',
            'Control Tasks',
            'Status'
        ],
        'Value': [
            model_name,
            dataset_source,
            str(device),
            str(batch_size),
            str(use_control_tasks),
            "Loading model and calculating detailed statistics..."
        ]
    })
    stats_placeholder.table(initial_stats_df)

    try:
        # 1. Load model with progress
        model_tracker.update(0, "Loading model...", "Initializing")
        tokenizer, model = load_model_and_tokenizer(
            model_name, model_tracker.update, device)
        mark_complete(model_status)

        memory_estimates = estimate_memory_requirements(model, batch_size)

        # 2. Load dataset with progress
        dataset_tracker.update(0, "Loading dataset...", "Initializing")

        # Pass custom_file if using custom dataset
        examples = []
        if dataset_source == "custom":
            if custom_file is not None:
                examples = load_dataset(
                    dataset_source,
                    dataset_tracker.update,
                    max_samples=max_samples,
                    custom_file=custom_file,
                    tf_splits=tf_splits
                )
            else:
                dataset_tracker.update(
                    1.0, "No file uploaded", "Please upload a CSV file")
                st.error("Please upload a CSV file for custom dataset")
                st.stop()
        else:
            examples = load_dataset(
                dataset_source,
                dataset_tracker.update,
                max_samples=max_samples,
                custom_file=None,
                tf_splits=tf_splits
            )

        # Check if we got any examples
        if len(examples) == 0:
            st.error(
                "No examples were loaded. Please check your dataset configuration.")
            st.stop()

        mark_complete(dataset_status)

        # Split data
        train_examples, test_examples = train_test_split(
            examples, test_size=test_size, random_state=42, shuffle=True
        )

        # Update stats display
        stats_df = pd.DataFrame({
            'Statistic': [
                'Total Examples',
                'Training Examples',
                'Test Examples',
                'Model Type',
                'Model Size',
                'Parameter Memory',
                'Activation Memory',
                'Current Memory Usage',
                'Precision',
                'Example'
            ],
            'Value': [
                len(examples),
                len(train_examples),
                len(test_examples),
                "Decoder-only" if is_decoder_only_model(
                    model_name) else "Encoder-only/Encoder-decoder",
                memory_estimates["param_count"],
                memory_estimates["param_memory"],
                memory_estimates["activation_memory"],
                memory_estimates["current_usage"],
                memory_estimates["precision"],
                str(train_examples[0]["text"][:50] +
                    "...") if train_examples else "N/A"
            ]
        })
        stats_placeholder.table(stats_df)

        # Get the number of layers
        num_layers = get_num_layers(model)

        # Save parameters to JSON file
        parameters = {
            "model_name": model_name,
            "dataset": dataset_source,
            "output_activation": output_layer,
            "batch_size": batch_size,
            "train_epochs": train_epochs,
            "learning_rate": learning_rate,
            "use_control_tasks": use_control_tasks,
            "device": str(device),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "test_size": test_size,
            "num_layers": num_layers,
            "num_examples": len(examples)
        }
        save_json(parameters, os.path.join(run_folder, "parameters.json"))
        add_log(f"Saved parameters to {os.path.join(run_folder, 'parameters.json')}")

        # 3. Extract embeddings with progress
        embedding_tracker.update(
            0, "Extracting embeddings for TRAIN set...", "Initializing")

        # Extract train embeddings
        train_hidden_states, train_labels = get_hidden_states_batched(
            train_examples, model, tokenizer, model_name, output_layer,
            dataset_type="TRAIN", progress_callback=embedding_tracker.update, 
            batch_size=batch_size, device=device
        )

        # Extract test embeddings
        embedding_tracker.update(
            0, "Extracting embeddings for TEST set...", "Initializing")
        test_hidden_states, test_labels = get_hidden_states_batched(
            test_examples, model, tokenizer, model_name, output_layer,
            dataset_type="TEST", progress_callback=embedding_tracker.update, 
            batch_size=batch_size, device=device
        )
        mark_complete(embedding_status)

        # 4. Train probes with progress
        training_tracker.update(0, "Training probes...", "Initializing")

        num_layers = get_num_layers(model)
        results = train_and_evaluate_model(
            train_hidden_states, train_labels,
            test_hidden_states, test_labels,
            num_layers, use_control_tasks,
            progress_callback=training_tracker.update,
            epochs=train_epochs, lr=learning_rate, device=device
        )
        mark_complete(training_status)

        # 5. Plot and display results
        with main_tabs[0]:
            # Selectivity plot (if using control tasks)
            acc_df = pd.DataFrame({
                'Layer': range(num_layers),
                'Accuracy': results['accuracies'],
                'Loss': results['test_losses']
            })
            if use_control_tasks:
                acc_df['Control Accuracy'] = results['control_accuracies']
                acc_df['Selectivity'] = results['selectivities']

            st.dataframe(acc_df)

            # Save results to JSON file
            results_to_save = {
                "accuracies": results['accuracies'],
                "test_losses": results['test_losses']
            }
            if use_control_tasks:
                results_to_save["control_accuracies"] = results['control_accuracies']
                results_to_save["selectivities"] = results['selectivities']

            save_json(results_to_save, os.path.join(run_folder, "results.json"))
            add_log(f"Saved results to {os.path.join(run_folder, 'results.json')}")

            # --- Calculate Alignment Strengths for all layers ---
            alignment_strengths, all_layers_mean_diff_activations, probe_weights = calculate_alignment_strengths(
                test_hidden_states, test_labels, results, num_layers
            )

            with accuracy_tab_container:  # Display in the first sub-tab of Probe Analysis
                if use_control_tasks and results['selectivities']:
                    fig_sel = plot_selectivity_by_layer(
                        results['selectivities'], results['accuracies'],
                        results['control_accuracies'], model_name, dataset_source
                    )
                    selectivity_plot.pyplot(fig_sel)
                    with st.expander("What does this chart show?", expanded=False):
                        st.markdown("""
                        This chart visualizes the performance of the linear truth probes across different layers of the model.

                        - **Accuracy (Blue Line):** Shows the percentage of test statements the probe for each layer correctly classified as true or false. Higher accuracy means the probe found a better truth-related signal in that layer. An accuracy of 0.5 is chance-level.
                        - **Control Accuracy (Yellow Dashed Line):** Shows the accuracy of a control probe trained on the same layer but with *shuffled labels*. This helps check if the main probe's accuracy is due to real learning or fitting to noise. Ideally, control accuracy is around 0.5.
                        - **Selectivity (Green Line):** Calculated as `Accuracy - Control Accuracy`. A high selectivity score suggests the probe genuinely learned a truth-distinguishing feature, not just random patterns.

                        The x-axis is the layer number (earlier to later). This shows how the linear decodability of truth changes with model depth.
                        If only the blue "Accuracy" line is shown, it means control tasks were not run, so selectivity isn't calculated.
                        """)
                else:
                    fig_acc = plot_accuracy_by_layer(
                        results['accuracies'], model_name, dataset_source)
                    accuracy_plot.pyplot(fig_acc)
                    save_graph(fig_acc, os.path.join(run_folder, "accuracy_plot.png"))
                    with st.expander("What does this chart show?", expanded=False):
                        st.markdown("""
                        This chart visualizes the performance of the linear truth probes across different layers of the model.

                        - **Accuracy (Blue Line):** Shows the percentage of test statements the probe for each layer correctly classified as true or false. Higher accuracy means the probe found a better truth-related signal in that layer. An accuracy of 0.5 is chance-level.

                        The x-axis is the layer number (earlier to later). This shows how the linear decodability of truth changes with model depth.
                        (Control tasks were not run, so selectivity and control accuracy are not displayed).
                        """)

                # Display Alignment Strength Plot
                if alignment_strengths:
                    fig_align_strength = plot_alignment_strength_by_layer(
                        alignment_strengths, model_name, dataset_source, run_folder
                    )
                    st.pyplot(fig_align_strength)
                    with st.expander("What does the Alignment Strength chart show?", expanded=False):
                        st.markdown("""
                        This chart displays the **Alignment Strength** for each layer, measured as the Pearson correlation coefficient between two sets of values for all neurons in that layer:
                        1.  **Mean Activation Difference (True - False):** How much each neuron's average activation changes when the model processes TRUE statements versus FALSE statements.
                        2.  **Probe Weight:** The weight assigned to each neuron by the trained linear probe for that layer.

                        **Interpretation:**
                        -   **Correlation near +1:** Strong positive alignment. Neurons that are naturally more active for TRUE statements are also given positive (excitatory for TRUE) weights by the probe, and neurons more active for FALSE get negative weights. The probe is leveraging a clear, direct signal.
                        -   **Correlation near -1:** Strong negative alignment. Neurons more active for TRUE are given negative weights by the probe (and vice versa). This suggests the probe is learning an inverse relationship or relying on suppression of truth-aligned neurons to detect falsehood (or vice versa).
                        -   **Correlation near 0:** Weak or no linear alignment. The probe's weights don't show a strong linear relationship with the neurons' natural True/False activation differences. The probe might be learning more complex, non-linear patterns, or the truth signal might be weak/diffuse in that layer with respect to these two measures.

                        This plot helps understand how directly the probe's learned strategy aligns with the raw activation patterns related to truth at each layer.
                        """)
                else:
                    st.info("Alignment strength data could not be computed.")

            # Restore PCA Tab Content
            with pca_tab_container:
                pca_plot.info("Generating PCA visualization...")
                fig_pca = plot_pca_grid(
                    test_hidden_states, test_labels, results['probes'], model_name, dataset_source)
                pca_plot.pyplot(fig_pca)
                with st.expander("What does this chart show?", expanded=False):
                    st.markdown("""
                    This grid of plots visualizes the hidden state activations from each layer of the model after being reduced to two dimensions using **Principal Component Analysis (PCA)**.
                    PCA finds the two directions (principal components) that capture the most variance in the high-dimensional activation data.

                    - **Each small plot** corresponds to a different layer in the model.
                    - **Points:** Each point represents a single statement from your test set.
                        - **Green points** are statements labeled as "True."
                        - **Red points** are statements labeled as "False."
                    - **Separation:** If true and false statements form distinct clusters in this 2D view, it suggests that the activations at that layer, even when simplified to 2D, contain information that can distinguish them.
                    - **Misclassified Points (Blue Circles):** Points circled in blue are those that the linear probe for that layer misclassified. This shows where the probe's decision boundary in the original high-dimensional space doesn't perfectly align with the true labels.
                    - **Decision Boundary (Dashed Line):** The dashed black line (if present) is an *approximation* of the linear probe's decision boundary, projected into this 2D PCA space.
                    - **Variance Explained (Var=X% in title):** This percentage indicates how much of the original variance in the high-dimensional activations is captured by the two principal components shown. A higher percentage means the 2D plot is a more faithful representation of the data's spread.

                    Looking across layers, you can see if and where the representations of true and false statements become more separable in this simplified 2D view.
                    """)

            # Restore Projection Tab Content
            with projection_tab_container:
                projection_plot.info(
                    "Generating truth projection histograms...")
                fig_proj = plot_truth_projections(
                    test_hidden_states, test_labels, results['probes'])
                projection_plot.pyplot(fig_proj)
                with st.expander("What does this chart show?", expanded=False):
                    st.markdown("""
                    This grid of plots visualizes how well the hidden state activations for true and false statements separate when projected onto the **"truth direction"** learned by the linear probe for each layer.

                    - **Each small plot** corresponds to a different layer in the model.
                    - **"Truth Direction":** For each layer, the linear probe learns a weight vector. This vector defines a direction in the high-dimensional activation space that the probe associates with "truth."
                    - **Projection:** Activations from the test set are projected onto this learned truth direction, resulting in a single scalar value for each statement.
                    - **Histograms:**
                        - **Green Histogram:** Distribution of projected values for statements that are actually **True**.
                        - **Red Histogram:** Distribution of projected values for statements that are actually **False**.
                    - **Separation & Overlap:** Ideally, the green and red histograms should be well-separated with minimal overlap. The `Overlap` value in the title quantifies this mixing (lower is better).
                    - **Decision Boundary (Vertical Dashed Line):** Represents the probe's decision threshold (usually at x=0).
                    - **Accuracy (Acc=X.XX in title):** The probe's accuracy for that layer.

                    This helps visualize, layer by layer, how distinctly the probe's learned truth direction separates true and false statements.
                    """)

            # Restore Data Tab Content
            with data_tab_container:
                layer_tabs = st.tabs(
                    [f"Layer {i}" for i in range(num_layers)])

                # Display analysis for the selected layer tab
                for i, layer_tab in enumerate(layer_tabs):
                    with layer_tab:
                        selected_layer = i

                        # --- Chart 1: Probe Neuron Weights ---
                        st.subheader(
                            f"Probe Neuron Weights for Layer {selected_layer}")
                        if results and 'probes' in results and selected_layer < len(results['probes']):
                            probe = results['probes'][selected_layer]
                            fig_probe_weights = plot_probe_weights(probe, selected_layer, run_folder)
                            st.pyplot(fig_probe_weights)

                            with st.expander("What does this chart show?", expanded=False):
                                st.markdown("""
                                This chart displays the **learned weight** assigned by the simple linear probe to each neuron (or element in the hidden dimension) for this specific layer.

                                - **Positive Weight (bar goes up):** Indicates that if this neuron has a high activation, the probe is more likely to classify the input statement as **TRUE**.
                                - **Negative Weight (bar goes down):** Indicates that if this neuron has a high activation, the probe is more likely to classify the input statement as **FALSE** (conversely, low activation might suggest TRUE to the probe).
                                - **Weight close to Zero:** The probe does not consider this neuron particularly important for its true/false classification at this layer.

                                Essentially, these weights show which neurons the probe has identified as most influential for distinguishing true from false statements based on the activations at this layer.
                                """)
                        else:
                            st.info(
                                "Probe weights are not available for this layer.")

                        # --- Chart 2: Difference in Mean Activations (True - False) ---
                        st.subheader(
                            f"Mean Activation Difference (True - False) for Layer {selected_layer}")
                        if test_hidden_states.nelement() > 0 and test_labels.nelement() > 0:
                            diff_activations, mean_true, mean_false = calculate_mean_activation_difference(
                                test_hidden_states, test_labels, selected_layer
                            )

                            if diff_activations is not None:
                                fig_diff_activations, ax_diff_activations = plt.subplots(
                                    figsize=(12, 4))
                                ax_diff_activations.bar(
                                    range(len(diff_activations)), diff_activations)
                                ax_diff_activations.set_title(
                                    f"Mean Activation Difference (True - False) - Layer {selected_layer}")
                                ax_diff_activations.set_xlabel(
                                    "Neuron Index in Hidden Dimension")
                                ax_diff_activations.set_ylabel(
                                    "Mean Activation Difference")
                                ax_diff_activations.grid(
                                    True, axis='y', linestyle='--', alpha=0.7)
                                plt.tight_layout()

                                # --- SAVE FIGURE ---
                                layer_save_dir = os.path.join(
                                    run_folder, "layers", str(selected_layer))
                                os.makedirs(layer_save_dir, exist_ok=True)
                                save_graph(fig_diff_activations, os.path.join(
                                    layer_save_dir, "activation_diff.png"))
                                # --- END SAVE ---

                                st.pyplot(fig_diff_activations)

                                with st.expander("What does this chart show?", expanded=False):
                                    st.markdown("""
                                    This chart displays the difference between the **mean (average) activation** of each neuron when the model processes **TRUE** statements versus when it processes **FALSE** statements from the test set.

                                    - **Positive Bar (bar goes up):** This neuron is, on average, **more active** when the input statement is TRUE compared to when it's false.
                                    - **Negative Bar (bar goes down):** This neuron is, on average, **less active** (or more negatively active) when the input statement is TRUE compared to when it's false. This means it tends to be more active for FALSE statements.
                                    - **Bar close to Zero:** This neuron's average activation level is similar for both true and false statements in the test set; its raw activity doesn't strongly distinguish between them on average.

                                    This visualization helps identify neurons whose raw activation levels (independent of any probe) show a systematic difference based on the ground truth label of the statements.
                                    """)
                            elif mean_true is None:
                                st.info(
                                    f"No true statements in the test set for layer {selected_layer} to calculate activation differences.")
                            elif mean_false is None:
                                st.info(
                                    f"No false statements in the test set for layer {selected_layer} to calculate activation differences.")
                            else:
                                st.info(
                                    f"Not enough data to calculate activation differences for layer {selected_layer}.")
                        else:
                            st.info(
                                "Test hidden states or labels are empty, cannot plot activation differences.")

                        # --- Chart 3: Neuron Alignment (Probe Weight vs. Activation Difference) ---
                        st.subheader(
                            f"Neuron Alignment: Probe Weight vs. Activation Difference - Layer {selected_layer}")
                        if results and 'probes' in results and selected_layer < len(results['probes']) and 'diff_activations' in locals() and diff_activations is not None:
                            probe = results['probes'][selected_layer]
                            probe_weights_for_alignment = probe.linear.weight[0].cpu().detach().numpy()

                            # Plot using the visualization module
                            fig_neuron_alignment = plot_neuron_alignment(
                                diff_activations,
                                probe_weights_for_alignment,
                                selected_layer,
                                run_folder
                            )
                            st.pyplot(fig_neuron_alignment)

                            with st.expander("What does this chart show?", expanded=False):
                                st.markdown("""
                                This scatter plot visualizes the relationship between two key properties of each neuron (or element in the hidden dimension) at this layer:

                                1.  **Mean Activation Difference (x-axis):** How much a neuron's average activation changes when the model processes TRUE statements versus FALSE statements.
                                    *   *Positive x-values:* Neuron is more active for TRUE statements.
                                    *   *Negative x-values:* Neuron is more active for FALSE statements.
                                    *   *Values near zero:* Neuron shows little difference in average activation.

                                2.  **Probe Weight (y-axis):** The weight assigned to this neuron by the trained linear probe.
                                    *   *Positive y-values:* High activation contributes to a TRUE prediction by the probe.
                                    *   *Negative y-values:* High activation contributes to a FALSE prediction by the probe.
                                    *   *Values near zero:* Neuron is not considered important by the probe.

                                **Interpretation of Quadrants:**

                                *   **Top-Right (High Diff, High Weight - e.g., Green Zone):** These neurons are naturally more active for TRUE statements AND the probe gives them a positive weight (considers them indicative of TRUE). This indicates strong alignment; the probe leverages a natural signal.
                                *   **Bottom-Left (Low Diff, Low Weight - e.g., Red Zone):** These neurons are naturally more active for FALSE statements (negative difference) AND the probe gives them a negative weight (considers them indicative of FALSE). This also indicates strong alignment.
                                *   **Top-Left (Low Diff, High Weight - e.g., Blue Zone):** These neurons don't show a strong natural preference for TRUE (x near 0 or negative), but the probe gives them a positive weight. The probe might be finding a subtle pattern or relying on this neuron in combination with others.
                                *   **Bottom-Right (High Diff, Low Weight - e.g., Purple Zone):** These neurons are naturally more active for TRUE statements, but the probe gives them a negative weight. This suggests a misalignment or a more complex role for this neuron.
                                *   **Near Origin (0,0):** Neurons that are neither strongly discriminative on their own nor heavily weighted by the probe.

                                **Point Size:** The size of each point is proportional to the product of the absolute mean activation difference and the absolute probe weight. Larger points highlight neurons that are strongly indicative in *both* aspects (high natural difference and high probe importance).
                                """)
                        else:
                            st.info(
                                "Neuron alignment data is not available. This usually requires both probe weights and activation differences to be successfully computed.")

                        # --- Top-K Influential Neurons ---
                        st.subheader(
                            f"Top 10 Influential Neurons for Layer {selected_layer}")
                        if 'diff_activations' in locals() and diff_activations is not None and \
                           results and 'probes' in results and selected_layer < len(results['probes']):
                            current_probe_weights = results['probes'][selected_layer].linear.weight[0].cpu().detach().numpy()
                            
                            top_k_data = get_top_k_neurons(diff_activations, current_probe_weights, k=10)

                            if top_k_data:
                                df_top_k = pd.DataFrame(top_k_data)
                                st.dataframe(df_top_k.style.format({
                                    "Contribution Score (abs(Diff*Weight))": "{:.4f}",
                                    "Mean Activation Difference": "{:.4f}",
                                    "Probe Weight": "{:.4f}"
                                }))
                            else:
                                st.info(
                                    "No influential neurons to display for this layer.")

                            with st.expander("What does this table show?", expanded=False):
                                st.markdown("""
                                This table lists the top neurons for this layer, ranked by their combined influence. The influence is measured by the **Contribution Score**, which is the absolute product of:
                                1.  **Mean Activation Difference:** How much the neuron's average activation differs when processing TRUE versus FALSE statements.
                                2.  **Probe Weight:** The weight assigned to this neuron by the linear probe.

                                Neurons with a high contribution score are those that both show a strong natural distinction between true/false statements AND are heavily relied upon by the probe for its classification.
                                -   **Neuron Index:** The index of the neuron within the layer's hidden dimension.
                                -   **Mean Activation Difference:** Positive means more active for TRUE; negative means more active for FALSE.
                                -   **Probe Weight:** Positive means the probe uses its activation to predict TRUE; negative for FALSE.
                                """)
                        else:
                            st.info(
                                "Top-K influential neuron data cannot be computed for this layer. This typically requires activation differences and probe weights.")

                        # Show details for selected layer in columns
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader(f"Layer {selected_layer} Details")
                            probe = results['probes'][selected_layer]

                            # Calculate metrics with the analysis module
                            metrics = calculate_confusion_matrix(
                                test_hidden_states, test_labels, probe, selected_layer
                            )
                            
                            # Display metrics
                            metrics_df = create_metrics_dataframe(metrics)
                            st.table(metrics_df)

                            # Add truth direction projection visualization
                            st.subheader("Truth Direction Projection")
                            fig_proj_individual = plot_truth_direction_projection(
                                test_hidden_states, test_labels, probe, selected_layer, run_folder
                            )
                            st.pyplot(fig_proj_individual)
                            
                            with st.expander("What does this chart show?", expanded=False):
                                st.markdown("""
                                This chart visualizes how well the hidden state activations for true and false statements from the test set separate when projected onto the **"truth direction"** learned by the linear probe specifically for **this layer**.

                                - **"Truth Direction":** The linear probe for this layer learned a weight vector, defining a direction in this layer's activation space that the probe associates with "truth."
                                - **Projection:** Activations from the test set are projected onto this learned truth direction, giving a single scalar value per statement.
                                - **Histograms:**
                                    - **Green Histogram:** Distribution of projected values for **True** statements.
                                    - **Red Histogram:** Distribution of projected values for **False** statements.
                                - **Separation:** Ideally, the green and red histograms should be well-separated.
                                - **Decision Boundary (Vertical Dashed Line):** Represents this probe's decision threshold (usually at x=0).

                                This chart helps assess how clearly this specific layer's probe distinguishes true from false statements along its learned truth axis.
                                """)

                        with col2:
                            st.subheader("Confusion Matrix")
                            # Plot confusion matrix using the analysis module
                            fig_confusion = plot_confusion_matrix(metrics, selected_layer, run_folder)
                            st.pyplot(fig_confusion)
                            
                            with st.expander("What does this chart show?", expanded=False):
                                st.markdown("""
                                This table summarizes the performance of the truth probe for this specific layer on the test set.

                                - **Rows** represent the **Actual** labels (False or True).
                                - **Columns** represent the **Predicted** labels (False or True) made by the probe.

                                The cells show the counts of test examples falling into each category:

                                - **Top-Left (Actual False, Predicted False):** True Negatives (TN) - Correctly identified as false.
                                - **Top-Right (Actual False, Predicted True):** False Positives (FP) - Incorrectly identified as true (Type I Error).
                                - **Bottom-Left (Actual True, Predicted False):** False Negatives (FN) - Incorrectly identified as false (Type II Error).
                                - **Bottom-Right (Actual True, Predicted True):** True Positives (TP) - Correctly identified as true.

                                Ideally, for a good probe, the numbers on the main diagonal (TN and TP) should be high, while the off-diagonal numbers (FP and FN) should be low.
                                """)

                            # Add examples
                            st.subheader("Example Predictions")

                            # Get some examples from the test set
                            test_feats = test_hidden_states[:, selected_layer, :]
                            with torch.no_grad():
                                test_outputs = probe(test_feats)
                                test_preds = (test_outputs > 0.5).long()
                                test_probs = test_outputs.cpu().numpy()
                                
                                # Get correct examples
                                correct_indices = (test_preds == test_labels).nonzero(as_tuple=True)[0]
                                
                                # Get incorrect examples
                                incorrect_indices = (test_preds != test_labels).nonzero(as_tuple=True)[0]
                                
                                # Display a few examples
                                st.subheader("Correct Examples")
                                if len(correct_indices) > 0:
                                    for i in range(min(3, len(correct_indices))):
                                        idx = correct_indices[i].item()
                                        text = test_examples[idx]["text"]
                                        actual = "True" if test_examples[idx]["label"] == 1 else "False"
                                        pred = "True" if test_preds[idx].item() == 1 else "False"
                                        conf = test_probs[idx].item() if pred == "True" else 1 - test_probs[idx].item()
                                        
                                        st.info(f"**Statement:** {text}\n\n**Actual:** {actual} | **Predicted:** {pred} | **Confidence:** {conf:.2f}")
                                else:
                                    st.info("No correct examples found.")
                                    
                                st.subheader("Incorrect Examples")
                                if len(incorrect_indices) > 0:
                                    for i in range(min(3, len(incorrect_indices))):
                                        idx = incorrect_indices[i].item()
                                        text = test_examples[idx]["text"]
                                        actual = "True" if test_examples[idx]["label"] == 1 else "False"
                                        pred = "True" if test_preds[idx].item() == 1 else "False"
                                        conf = test_probs[idx].item() if pred == "True" else 1 - test_probs[idx].item()
                                        
                                        st.error(f"**Statement:** {text}\n\n**Actual:** {actual} | **Predicted:** {pred} | **Confidence:** {conf:.2f}")
                                else:
                                    st.info("No incorrect examples found.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        add_log(f"ERROR: {str(e)}")