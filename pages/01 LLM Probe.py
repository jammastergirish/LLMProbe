from utils.models import model_options
import nest_asyncio
import streamlit as st
import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.file_manager import create_run_folder, save_json, save_graph
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import random
import os
import warnings
import gc
import time
from datetime import datetime
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


def estimate_memory_requirements(model, batch_size, seq_length=128):
    """Estimate memory requirements dynamically from the model"""

    # Get model parameters
    if hasattr(model, "config"):
        # For HuggingFace models
        hidden_dim = getattr(model.config, "hidden_size", 0)
        num_layers = getattr(model.config, "num_hidden_layers", 0) + 1
    elif hasattr(model, "cfg"):
        # For TransformerLens models
        hidden_dim = getattr(model.cfg, "d_model", 0)
        num_layers = getattr(model.cfg, "n_layers", 0)
    else:
        return {"param_memory": "Unknown", "activation_memory": "Unknown"}

    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())

    # Get precision (default to FP32 if can't determine)
    if next(model.parameters()).dtype == torch.float16:
        precision = 2  # bytes for FP16
    elif next(model.parameters()).dtype == torch.int8:
        precision = 1  # bytes for INT8/quantized
    else:
        precision = 4  # bytes for FP32

    # Calculate memory in GB
    param_memory = (param_count * precision) / (1024**3)

    # Activation memory estimate: batch_size × seq_length × hidden_dim × num_layers × precision
    activation_memory = (batch_size * seq_length *
                         hidden_dim * num_layers * precision) / (1024**3)

    # Get current GPU memory usage if available
    current_memory_usage = "N/A"
    if torch.cuda.is_available():
        try:
            current_memory = torch.cuda.memory_allocated() / (1024**3)
            current_memory_usage = f"{current_memory:.2f} GB"
        except:
            pass

    # Free any temporary tensors
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "param_count": f"{param_count/1e9:.2f}B parameters",
        "param_memory": f"{param_memory:.2f} GB",
        "activation_memory": f"{activation_memory:.2f} GB",
        "precision": f"{precision*8} bit" if precision < 4 else "32 bit",
        "current_usage": current_memory_usage
    }


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
            import pandas as pd
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


def is_decoder_only_model(model_name):
    decoder_keywords = ["gpt", "llama", "mistral",
                        "pythia", "deepseek", "qwen", "gemma"]
    return any(keyword in model_name.lower() for keyword in decoder_keywords)


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


def load_model_and_tokenizer(model_name, progress_callback):
    """Load model and tokenizer with progress updates"""
    progress_callback(0.1, "Initializing model loading process...",
                      "Preparing tokenizer and model configuration")

    if is_decoder_only_model(model_name):
        progress_callback(0.2, "Detected decoder-only model architecture",
                          f"Loading {model_name} with TransformerLens for better compatibility")

        try:
            # Import necessary libraries
            progress_callback(
                0.3, "Importing TransformerLens library...", "Setting up model dependencies")
            import transformer_lens
            from transformer_lens import HookedTransformer
            from transformers import AutoTokenizer

            # Load tokenizer first
            progress_callback(0.4, "Loading tokenizer...",
                              f"Fetching tokenizer configuration for {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            progress_callback(0.5, "Configuring tokenizer settings...",
                              "Setting padding token and padding side")

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

            # Now load the model
            progress_callback(0.6, "Loading HookedTransformer model...",
                              f"This may take a while for {model_name}")
            model = HookedTransformer.from_pretrained(
                model_name, device=device)

            # Report model statistics
            n_layers = model.cfg.n_layers
            d_model = model.cfg.d_model
            progress_callback(0.9, f"Model loaded: {n_layers} layers, {d_model} dimensions",
                              f"Using device: {str(device)}")

            progress_callback(1.0, "Model and tokenizer successfully loaded",
                              f"Ready to process with {model_name}")

        except Exception as e:
            progress_callback(
                1.0, f"Error loading model: {str(e)}", "Check model name or connection")
            raise e
    else:
        try:
            from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

            progress_callback(0.3, "Detected encoder or encoder-decoder architecture",
                              f"Loading {model_name} using Hugging Face Transformers")

            progress_callback(0.4, "Loading tokenizer...",
                              f"Fetching tokenizer for {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            progress_callback(0.5, "Configuring tokenizer settings...",
                              "Setting padding token and padding side")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right" if not is_decoder_only_model(
                model_name) else "left"

            progress_callback(0.7, "Loading model...",
                              f"This may take a while for {model_name}")
            model_class = AutoModelForCausalLM if is_decoder_only_model(
                model_name) else AutoModel
            model = model_class.from_pretrained(
                model_name, output_hidden_states=True).to(device)
            model.eval()

            # Get model statistics
            n_layers = model.config.num_hidden_layers
            d_model = model.config.hidden_size
            progress_callback(0.9, f"Model loaded: {n_layers} layers, {d_model} dimensions",
                              f"Using device: {str(device)}")

            progress_callback(1.0, "Model and tokenizer successfully loaded",
                              f"Ready to process with {model_name}")
        except Exception as e:
            progress_callback(
                1.0, f"Error loading model: {str(e)}", "Check model name or connection")
            raise e

    return tokenizer, model


def load_dataset(dataset_source, progress_callback, max_samples=5000, custom_file=None):
    """Load dataset with progress updates"""
    examples = []

    if dataset_source == "custom" and custom_file is not None:
        progress_callback(0.1, "Loading custom dataset...",
                          "Processing uploaded CSV file")
        try:
            import pandas as pd
            custom_df = pd.read_csv(custom_file)

            if 'statement' not in custom_df.columns or 'label' not in custom_df.columns:
                progress_callback(1.0, "Error: CSV must contain 'statement' and 'label' columns",
                                  "Please check your CSV format and try again")
                return []

            # Clean and validate data
            custom_df = custom_df.dropna(subset=['statement', 'label'])

            # Ensure labels are 0 or 1
            if not all(label in [0, 1] for label in custom_df['label'].unique()):
                progress_callback(1.0, "Error: Labels must be 0 or 1",
                                  "Please check your labels and try again")
                return []

            # Process each row
            for idx, row in enumerate(custom_df.itertuples()):
                if idx % 10 == 0:
                    progress = 0.1 + (idx / len(custom_df)) * 0.9
                    progress_callback(progress, f"Processing custom example {idx+1}/{len(custom_df)}",
                                      f"Statement: {row.statement[:50]}...")

                examples.append({
                    "text": row.statement,
                    "label": int(row.label)
                })

                # Limit dataset size if needed
                if len(examples) >= max_samples:
                    break

            progress_callback(1.0, f"Loaded custom dataset: {len(examples)} examples",
                              "Custom dataset processed successfully")

            return examples

        except Exception as e:
            progress_callback(1.0, f"Error loading custom dataset: {str(e)}",
                              "Please check your CSV format and try again")
            return []

    if dataset_source in ["fever", "all"]:
        progress_callback(0.9, "Preparing to load FEVER dataset...",
                          "Initializing FEVER dataset from Hugging Face")
        try:
            from datasets import load_dataset
            # You can also use "validation"
            fever = load_dataset(
                "fever", 'v1.0', split="train", trust_remote_code=True)
            start_examples = len(examples)
            # add_log(f"FEVER EXAMPLE 1: {fever[0]}")

            for idx, row in enumerate(fever):
                label = row.get("label", None)
                claim = row.get("claim", "")

                if label == "SUPPORTS":
                    examples.append({"text": claim, "label": 1})
                elif label == "REFUTES":
                    examples.append({"text": claim, "label": 0})
                else:
                    continue  # skip "NOT ENOUGH INFO" or None

                if len(examples) - start_examples >= max_samples and dataset_source != "all":
                    break

                if idx % 1000 == 0:
                    progress = 0.9 + (idx / min(len(fever), max_samples)) * 0.1
                    progress_callback(progress, f"Processing FEVER example {idx+1}",
                                      f"Claim: {claim[:60]}... Label: {label}")

            progress_callback(1.0, f"Loaded FEVER: {len(examples) - start_examples} examples added",
                              f"Total examples: {len(examples)}")
        except Exception as e:
            progress_callback(1.0, f"Error loading FEVER: {str(e)}",
                              "Continuing with other datasets if selected")

    if dataset_source in ["truthfulqa", "all"]:
        progress_callback(0.1, "Preparing to load TruthfulQA dataset...",
                          "Initializing dataset loading from Hugging Face")
        try:
            from datasets import load_dataset
            progress_callback(0.2, "Loading TruthfulQA (multiple_choice)...",
                              "Downloading and processing TruthfulQA dataset")

            tq = load_dataset("truthful_qa", "multiple_choice")["validation"]
            total_qa_pairs = 0

            progress_callback(0.25, "Processing TruthfulQA examples...",
                              "Extracting question-answer pairs with truth labels")

            for row_idx, row in enumerate(tq):
                if row_idx % 10 == 0:
                    progress = 0.25 + (row_idx / len(tq)) * 0.15
                    progress_callback(progress, f"Processing TruthfulQA example {row_idx+1}/{len(tq)}",
                                      f"Question: {row.get('question', '')[:50]}...")

                q = row.get("question", "")
                targets = row.get("mc1_targets", {})
                choices = targets.get("choices", [])
                labels = targets.get("labels", [])
                for answer, label in zip(choices, labels):
                    examples.append({"text": f"{q} {answer}", "label": label})
                    total_qa_pairs += 1

                    # Limit dataset size if needed
                    if total_qa_pairs >= max_samples and dataset_source != "all":
                        break

                if total_qa_pairs >= max_samples and dataset_source != "all":
                    break

            progress_callback(0.4, f"Loaded TruthfulQA: {total_qa_pairs} examples",
                              f"Question-answer pairs with truth labels processed")
        except Exception as e:
            progress_callback(0.4, f"Error loading TruthfulQA: {str(e)}",
                              "Continuing with other datasets if selected")

    if dataset_source in ["boolq", "all"]:
        progress_callback(0.4, "Preparing to load BoolQ dataset...",
                          "Initializing BoolQ dataset from Hugging Face")
        try:
            from datasets import load_dataset
            progress_callback(0.45, "Loading BoolQ dataset...",
                              "Downloading and processing BoolQ dataset")

            bq = load_dataset("boolq")["train"]
            start_examples = len(examples)

            for idx, row in enumerate(bq):
                if idx % 50 == 0:
                    progress = 0.45 + (idx / len(bq)) * 0.15
                    progress_callback(progress, f"Processing BoolQ example {idx+1}/{len(bq)}",
                                      f"Question: {row['question'][:50]}...")

                question = row["question"]
                passage = row["passage"]
                label = 1 if row["answer"] else 0
                examples.append(
                    {"text": f"{question} {passage}", "label": label})

                # Limit dataset size if needed
                if len(examples) - start_examples >= max_samples and dataset_source != "all":
                    break

            progress_callback(0.6, f"Loaded BoolQ: Total {len(examples)} examples",
                              f"Added {len(examples) - start_examples} examples from BoolQ")
        except Exception as e:
            progress_callback(0.6, f"Error loading BoolQ: {str(e)}",
                              "Continuing with other datasets if selected")

    if dataset_source in ["truefalse", "all"]:
        progress_callback(0.6, "Preparing to load TrueFalse dataset...",
                          "Initializing true-false datasets from multiple sources")
        try:
            from datasets import load_dataset, concatenate_datasets

            progress_callback(0.65, "Loading TrueFalse dataset splits...",
                              f"Processing {len(tf_splits)} dataset categories")

            datasets_list = []
            for i, split in enumerate(tf_splits):
                split_progress = 0.65 + (i / len(tf_splits)) * 0.1
                progress_callback(split_progress, f"Loading TrueFalse split: {split}",
                                  f"Processing split {i+1}/{len(tf_splits)}")

                split_ds = load_dataset("pminervini/true-false", split=split)
                datasets_list.append(split_ds)

            tf = concatenate_datasets(datasets_list)
            start_examples = len(examples)

            for idx, row in enumerate(tf):
                if idx % 100 == 0:
                    progress = 0.75 + (idx / min(len(tf), max_samples)) * 0.1
                    progress_callback(progress, f"Processing TrueFalse example {idx+1}/{len(tf)}",
                                      f"Statement: {row['statement'][:50]}...")

                examples.append(
                    {"text": row["statement"], "label": row["label"]})

                # Limit dataset size if needed
                if len(examples) - start_examples >= max_samples and dataset_source != "all":
                    break

                # Also limit the total if we're doing "all" datasets to avoid memory issues
                if dataset_source == "all" and len(examples) >= max_samples * 3:
                    break

            progress_callback(0.85, f"Loaded TrueFalse: Added {len(examples) - start_examples} examples",
                              f"Total examples so far: {len(examples)}")
        except Exception as e:
            progress_callback(0.85, f"Error loading TrueFalse: {str(e)}",
                              "Continuing with other datasets if selected")

    if dataset_source in ["arithmetic", "all"]:
        progress_callback(0.85, "Generating arithmetic dataset...",
                          "Creating synthetic true/false arithmetic examples")

        def generate_arithmetic_dataset(n=5000):
            data = []
            for i in range(n):
                if i % 100 == 0:
                    progress = 0.85 + (i / n) * 0.1
                    progress_callback(progress, f"Generating arithmetic example {i+1}/{n}",
                                      "Creating balanced true/false arithmetic statements")

                a = random.randint(0, 100)
                b = random.randint(0, 100)

                # 50% chance of being true
                if len(data) % 2 == 0:
                    correct_sum = a + b
                    text = f"{a} + {b} = {correct_sum}"
                    label = 1
                else:
                    incorrect_sum = a + b + \
                        random.choice([i for i in range(-10, 11) if i != 0])
                    text = f"{a} + {b} = {incorrect_sum}"
                    label = 0

                data.append({"text": text, "label": label})

            return data

        arithmetic = generate_arithmetic_dataset(min(5000, max_samples))
        start_examples = len(examples)
        examples.extend(arithmetic)

        progress_callback(0.95, f"Generated arithmetic dataset: {len(arithmetic)} examples",
                          f"Added {len(arithmetic)} arithmetic examples")

    progress_callback(1.0, f"Prepared {len(examples)} labeled examples for probing",
                      f"Dataset preparation complete with {len(examples)} total examples")
    return examples


def get_hidden_states_batched(examples, model, tokenizer, model_name, output_layer,
                              dataset_type="", return_layer=None, progress_callback=None,
                              batch_size=16):
    """Extract hidden states with batching for better performance"""
    all_hidden_states = []
    all_labels = []

    is_decoder = is_decoder_only_model(model_name)
    is_transformerlens = "HookedTransformer" in str(type(model))

    # Get model dimensions
    if is_transformerlens:
        n_layers = model.cfg.n_layers
        d_model = model.cfg.d_model
    else:
        n_layers = getattr(model.config, "num_hidden_layers", 12) + 1
        d_model = getattr(model.config, "hidden_size", 768)

    # Process in batches
    num_batches = math.ceil(len(examples) / batch_size)
    progress_callback(0, f"Processing {len(examples)} examples in {num_batches} batches",
                      f"Using batch size of {batch_size}")

    for batch_idx in range(0, len(examples), batch_size):
        batch_end = min(batch_idx + batch_size, len(examples))
        batch = examples[batch_idx:batch_end]

        # Update progress
        progress = batch_idx / len(examples)
        progress_callback(progress, f"Processing batch {batch_idx//batch_size + 1}/{num_batches}",
                          f"Examples {batch_idx+1}-{batch_end} of {len(examples)}")

        batch_texts = [ex["text"] for ex in batch]
        batch_labels = [ex["label"] for ex in batch]

        # Process the batch based on model type
        if is_transformerlens:
            # TransformerLens doesn't support true batching with run_with_cache,
            # so we process examples individually but still in batch chunks
            batch_hidden_states = []
            for text_idx, text in enumerate(batch_texts):
                tokens = tokenizer.encode(text, return_tensors="pt").to(device)
                _, cache = model.run_with_cache(tokens)

                pos = -1 if is_decoder else 0
                layer_outputs = [
                    cache[output_layer, layer_idx][0, pos, :]
                    for layer_idx in range(n_layers)
                ]
                hidden_stack = torch.stack(layer_outputs)
                batch_hidden_states.append(hidden_stack)
        else:
            # Standard transformers batching
            if "qwen" in model_name.lower():
                # Special handling for Qwen chat models
                encoded_inputs = []
                for text in batch_texts:
                    messages = [{"role": "user", "content": text}]
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False)
                    encoded_inputs.append(prompt)

                # Tokenize as a batch
                inputs = tokenizer(encoded_inputs, padding=True, truncation=True,
                                   return_tensors="pt", max_length=128)
            else:
                # Standard tokenization for other models
                inputs = tokenizer(batch_texts, padding=True, truncation=True,
                                   return_tensors="pt", max_length=128)

            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                hidden_states = outputs.hidden_states

                batch_hidden_states = []

                # Process each example in the batch
                for example_idx in range(len(batch)):
                    # Extract embeddings for each layer
                    example_layers = []

                    for layer_idx, layer in enumerate(hidden_states):
                        # Get representation based on selected strategy
                        if is_decoder:
                            # For decoder models, use the last token
                            if hasattr(inputs, "attention_mask"):
                                # Get position of last non-padding token
                                seq_len = inputs["attention_mask"][example_idx].sum(
                                ).item()
                                token_repr = layer[example_idx, seq_len-1, :]
                            else:
                                # Just use last token
                                token_repr = layer[example_idx, -1, :]
                        elif output_layer == "CLS":
                            # Use first token for BERT-like models
                            token_repr = layer[example_idx, 0, :]
                        elif output_layer == "mean":
                            # Mean pooling (average all tokens)
                            if hasattr(inputs, "attention_mask"):
                                # Only consider non-padding tokens
                                mask = inputs["attention_mask"][example_idx].unsqueeze(
                                    -1)
                                token_repr = (
                                    layer[example_idx] * mask).sum(dim=0) / mask.sum()
                            else:
                                token_repr = layer[example_idx].mean(dim=0)
                        elif output_layer == "max":
                            # Max pooling
                            if hasattr(inputs, "attention_mask"):
                                # Apply mask to avoid including padding tokens
                                mask = inputs["attention_mask"][example_idx].unsqueeze(
                                    -1)
                                masked_layer = layer[example_idx] * \
                                    mask - 1e9 * (1 - mask)
                                token_repr = masked_layer.max(dim=0).values
                            else:
                                token_repr = layer[example_idx].max(
                                    dim=0).values
                        elif output_layer.startswith("token_index_"):
                            # Use specific token index
                            index = int(output_layer.split("_")[-1])
                            seq_len = inputs["attention_mask"][example_idx].sum().item(
                            ) if hasattr(inputs, "attention_mask") else layer.size(1)
                            safe_index = min(index, seq_len - 1)
                            token_repr = layer[example_idx, safe_index, :]
                        else:
                            raise ValueError(
                                f"Unsupported output layer: {output_layer}")

                        example_layers.append(token_repr)

                    # Stack layers for this example
                    example_stack = torch.stack(example_layers)
                    batch_hidden_states.append(example_stack)

        # Collect results from this batch
        all_hidden_states.extend(batch_hidden_states)
        all_labels.extend(batch_labels)

        # Small sleep to allow UI to update
        time.sleep(0.01)

    # Convert to tensors
    all_hidden_states = torch.stack(all_hidden_states).to(
        device)  # [num_examples, num_layers, hidden_dim]
    all_labels = torch.tensor(all_labels).to(device)

    # Update to 100%
    progress_callback(1.0, f"Completed processing all {len(examples)} examples",
                      f"Created tensor of shape {all_hidden_states.shape}")

    # Return full tensor or specific layer
    if return_layer is not None:
        return all_hidden_states[:, return_layer, :], all_labels
    else:
        return all_hidden_states, all_labels


class LinearProbe(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = torch.nn.Linear(dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x)).squeeze(-1)


def train_probe(features, labels, epochs=100, lr=1e-2):
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


def get_num_layers(model):
    if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
        return model.config.num_hidden_layers + 1
    elif hasattr(model, "cfg") and hasattr(model.cfg, "n_layers"):
        return model.cfg.n_layers
    else:
        raise AttributeError(
            "Cannot determine number of layers for this model")


def train_and_evaluate_model(train_hidden_states, train_labels, test_hidden_states, test_labels,
                             num_layers, use_control_tasks, progress_callback=None, epochs=100, lr=0.01):
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

        for epoch in range(epochs):
            if epoch % 10 == 0 or epoch == epochs - 1:
                epoch_progress = main_progress + (epoch / epochs) / num_layers
                progress_callback(epoch_progress,
                                  f"Layer {layer+1}/{num_layers}: Epoch {epoch+1}/{epochs}",
                                  f"Training linear probe for truth detection")

            optimizer.zero_grad()
            outputs = probe(train_feats)
            loss = criterion(outputs, train_labels.float())
            loss.backward()
            optimizer.step()

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
                train_feats, shuffled_labels, epochs=epochs, lr=lr)

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
            preds = (probe(torch.tensor(feats).to(device))
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
                    device)).cpu().numpy()
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

# --- NEW FUNCTION: Plot Neuron Alignment ---


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

    # Add a colorbar if using color encoding that varies meaningfully
    # cbar = fig.colorbar(scatter, label='Importance (abs(Diff * Weight))')
    # cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()

    # Save the figure
    layer_save_dir = os.path.join(run_folder, "layers", str(layer_index))
    os.makedirs(layer_save_dir, exist_ok=True)
    save_graph(fig, os.path.join(layer_save_dir, "neuron_alignment.png"))

    return fig
# --- END NEW FUNCTION ---

# --- NEW FUNCTION: Plot Alignment Strength by Layer ---


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
# --- END NEW FUNCTION ---

# Update progress functions with enhanced UI


def update_model_progress(progress, message, details=""):
    model_status.markdown(
        '<span class="status-running">Running</span>', unsafe_allow_html=True)
    model_progress_bar.progress(progress)
    model_progress_text.markdown(f"**{message}**")
    model_detail.text(details)
    add_log(f"Load Model ({progress:.0%}): {message} - {details}")


def update_dataset_progress(progress, message, details=""):
    dataset_status.markdown(
        '<span class="status-running">Running</span>', unsafe_allow_html=True)
    dataset_progress_bar.progress(progress)
    dataset_progress_text.markdown(f"**{message}**")
    dataset_detail.text(details)
    add_log(f"Load Dataset ({progress:.0%}): {message} - {details}")


def update_embedding_progress(progress, message, details=""):
    embedding_status.markdown(
        '<span class="status-running">Running</span>', unsafe_allow_html=True)
    embedding_progress_bar.progress(progress)
    embedding_progress_text.markdown(f"**{message}**")
    embedding_detail.text(details)
    add_log(f"Create Representations ({progress:.0%}): {message} - {details}")


def update_training_progress(progress, message, details=""):
    training_status.markdown(
        '<span class="status-running">Running</span>', unsafe_allow_html=True)
    training_progress_bar.progress(progress)
    training_progress_text.markdown(f"**{message}**")
    training_detail.text(details)
    add_log(f"Train Probe ({progress:.0%}): {message} - {details}")


def mark_complete(status_element, message="Complete"):
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
        update_model_progress(0, "Loading model...", "Initializing")
        tokenizer, model = load_model_and_tokenizer(
            model_name, update_model_progress)
        mark_complete(model_status)

        memory_estimates = estimate_memory_requirements(model, batch_size)

        # 2. Load dataset with progress
        update_dataset_progress(0, "Loading dataset...", "Initializing")

        # Pass custom_file if using custom dataset
        examples = []
        if dataset_source == "custom":
            if custom_file is not None:
                examples = load_dataset(
                    dataset_source,
                    update_dataset_progress,
                    max_samples=max_samples,
                    custom_file=custom_file  # Pass the custom file directly
                )
            else:
                update_dataset_progress(
                    1.0, "No file uploaded", "Please upload a CSV file")
                st.error("Please upload a CSV file for custom dataset")
                st.stop()
        else:
            examples = load_dataset(
                dataset_source,
                update_dataset_progress,
                max_samples=max_samples,
                custom_file=None  # Or just omit the parameter
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

        # 3. Extract embeddings with progress
        update_embedding_progress(
            0, "Extracting embeddings for TRAIN set...", "Initializing")

        # Extract train embeddings
        train_hidden_states, train_labels = get_hidden_states_batched(
            train_examples, model, tokenizer, model_name, output_layer,
            dataset_type="TRAIN", progress_callback=update_embedding_progress, batch_size=batch_size
        )

        # Extract test embeddings
        update_embedding_progress(
            0, "Extracting embeddings for TEST set...", "Initializing")
        test_hidden_states, test_labels = get_hidden_states_batched(
            test_examples, model, tokenizer, model_name, output_layer,
            dataset_type="TEST", progress_callback=update_embedding_progress, batch_size=batch_size
        )
        mark_complete(embedding_status)

        # 4. Train probes with progress
        update_training_progress(0, "Training probes...", "Initializing")

        num_layers = get_num_layers(model)
        results = train_and_evaluate_model(
            train_hidden_states, train_labels,
            test_hidden_states, test_labels,
            num_layers, use_control_tasks,
            progress_callback=update_training_progress,
            epochs=train_epochs, lr=learning_rate
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

            # --- Calculate Alignment Strengths for all layers ---
            all_layers_mean_diff_activations_for_corr = []
            probe_weights_for_corr = []
            alignment_strengths = []

            if test_hidden_states.nelement() > 0 and test_labels.nelement() > 0:
                for layer_idx in range(num_layers):
                    layer_feats_corr = test_hidden_states[:, layer_idx, :]
                    true_indices_corr = (
                        test_labels == 1).nonzero(as_tuple=True)[0]
                    false_indices_corr = (
                        test_labels == 0).nonzero(as_tuple=True)[0]

                    if len(true_indices_corr) > 0 and len(false_indices_corr) > 0:
                        mean_true_corr = layer_feats_corr[true_indices_corr].mean(
                            dim=0).cpu().numpy()
                        mean_false_corr = layer_feats_corr[false_indices_corr].mean(
                            dim=0).cpu().numpy()
                        diff_act_corr = mean_true_corr - mean_false_corr
                        all_layers_mean_diff_activations_for_corr.append(
                            diff_act_corr)

                        current_probe_weights_corr = results['probes'][layer_idx].linear.weight[0].cpu(
                        ).detach().numpy()
                        probe_weights_for_corr.append(
                            current_probe_weights_corr)

                        # Ensure both arrays are 1D and have the same length
                        if diff_act_corr.ndim == 1 and current_probe_weights_corr.ndim == 1 and len(diff_act_corr) == len(current_probe_weights_corr) and len(diff_act_corr) > 1:
                            correlation = np.corrcoef(
                                diff_act_corr, current_probe_weights_corr)[0, 1]
                            alignment_strengths.append(correlation)
                        else:
                            # Append NaN if correlation cannot be computed
                            alignment_strengths.append(np.nan)
                    else:
                        all_layers_mean_diff_activations_for_corr.append(
                            np.array([]))  # Or None, or np.nan array
                        probe_weights_for_corr.append(np.array([]))
                        alignment_strengths.append(np.nan)
            # --- End Alignment Strength Calculation ---

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
                        -   **Correlation near -1:** Strong negative alignment. Neurons more active for TRUE are given negative weights by the probe (and vice-versa). This suggests the probe is learning an inverse relationship or relying on suppression of truth-aligned neurons to detect falsehood (or vice versa).
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
                            probe_weights = probe.linear.weight[0].cpu(
                            ).detach().numpy()

                            fig_probe_weights, ax_probe_weights = plt.subplots(
                                figsize=(12, 4))
                            ax_probe_weights.bar(
                                range(len(probe_weights)), probe_weights)
                            ax_probe_weights.set_title(
                                f"Probe Neuron Weights - Layer {selected_layer}")
                            ax_probe_weights.set_xlabel(
                                "Neuron Index in Hidden Dimension")
                            ax_probe_weights.set_ylabel("Weight Value")
                            ax_probe_weights.grid(
                                True, axis='y', linestyle='--', alpha=0.7)
                            plt.tight_layout()

                            # --- SAVE FIGURE ---
                            layer_save_dir = os.path.join(
                                run_folder, "layers", str(selected_layer))
                            os.makedirs(layer_save_dir, exist_ok=True)
                            save_graph(fig_probe_weights, os.path.join(
                                layer_save_dir, "probe_weights.png"))
                            # --- END SAVE ---

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
                            layer_feats = test_hidden_states[:,
                                                             selected_layer, :]

                            true_indices = (test_labels == 1).nonzero(
                                as_tuple=True)[0]
                            false_indices = (test_labels == 0).nonzero(
                                as_tuple=True)[0]

                            if len(true_indices) > 0 and len(false_indices) > 0:
                                mean_activations_true = layer_feats[true_indices].mean(
                                    dim=0).cpu().numpy()
                                mean_activations_false = layer_feats[false_indices].mean(
                                    dim=0).cpu().numpy()
                                diff_activations = mean_activations_true - mean_activations_false

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
                            elif len(true_indices) == 0:
                                st.info(
                                    f"No true statements in the test set for layer {selected_layer} to calculate activation differences.")
                            elif len(false_indices) == 0:
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
                            probe_weights_for_alignment = probe.linear.weight[0].cpu(
                                # Already defined as probe_weights earlier
                            ).detach().numpy()

                            # Ensure mean_diff (diff_activations) and probe_weights_for_alignment are available
                            # diff_activations should be available from the previous plot
                            fig_neuron_alignment = plot_neuron_alignment(
                                diff_activations,  # This is mean_activation_true - mean_activation_false
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
                            current_probe_weights = results['probes'][selected_layer].linear.weight[0].cpu(
                            ).detach().numpy()
                            contribution_scores = np.abs(
                                diff_activations * current_probe_weights)

                            k = 10
                            top_k_indices = np.argsort(
                                contribution_scores)[::-1][:k]

                            top_k_data = []
                            for rank, neuron_idx in enumerate(top_k_indices):
                                top_k_data.append({
                                    "Rank": rank + 1,
                                    "Neuron Index": neuron_idx,
                                    "Contribution Score (abs(Diff*Weight))": contribution_scores[neuron_idx],
                                    "Mean Activation Difference": diff_activations[neuron_idx],
                                    "Probe Weight": current_probe_weights[neuron_idx]
                                })

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

                            # Extract test features for this layer
                            test_feats = test_hidden_states[:,
                                                            selected_layer, :]

                            with torch.no_grad():
                                # Get predictions
                                test_outputs = probe(test_feats)
                                test_preds = (test_outputs > 0.5).long()

                                # Make sure tensors are on the same device
                                test_preds_device = test_preds.to(device)
                                test_labels_device = test_labels.to(device)

                                # Confusion matrix components
                                TP = ((test_preds_device == 1) & (
                                    test_labels_device == 1)).sum().item()
                                FP = ((test_preds_device == 1) & (
                                    test_labels_device == 0)).sum().item()
                                TN = ((test_preds_device == 0) & (
                                    test_labels_device == 0)).sum().item()
                                FN = ((test_preds_device == 0) & (
                                    test_labels_device == 1)).sum().item()

                                # Calculate metrics
                                accuracy = (TP + TN) / (TP + TN + FP +
                                                        FN) if (TP + TN + FP + FN) > 0 else 0
                                precision = TP / \
                                    (TP + FP) if (TP + FP) > 0 else 0
                                recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                                f1 = 2 * precision * recall / \
                                    (precision + recall) if (precision +
                                                             recall) > 0 else 0

                            # Display metrics
                            metrics_df = pd.DataFrame({
                                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'True Positives',
                                           'False Positives', 'True Negatives', 'False Negatives'],
                                'Value': [f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}",
                                          str(TP), str(FP), str(TN), str(FN)]
                            })
                            st.table(metrics_df)

                            # Add truth direction projection visualization
                            st.subheader("Truth Direction Projection")
                            with torch.no_grad():
                                projection = torch.matmul(
                                    test_feats, probe.linear.weight[0])

                                # Get projection values for true and false examples
                                true_proj = projection[test_labels == 1].cpu(
                                ).numpy()
                                false_proj = projection[test_labels == 0].cpu(
                                ).numpy()

                                # Create histogram
                                fig_proj_individual, ax_proj = plt.subplots(
                                    figsize=(8, 3))
                                bins = np.linspace(
                                    min(projection.min().item(), -3),
                                    max(projection.max().item(), 3),
                                    30
                                )

                                # Plot histograms
                                ax_proj.hist(true_proj, bins=bins, alpha=0.7,
                                             label="True", color="#4CAF50")
                                ax_proj.hist(false_proj, bins=bins, alpha=0.7,
                                             label="False", color="#F44336")

                                # Add a vertical line at the decision boundary (0.0)
                                ax_proj.axvline(x=0, color='black',
                                                linestyle='--', alpha=0.5)
                                ax_proj.set_xlabel(
                                    "Projection onto Truth Direction")
                                ax_proj.set_ylabel("Count")
                                ax_proj.legend()

                                # --- SAVE FIGURE ---
                                save_graph(fig_proj_individual, os.path.join(
                                    layer_save_dir, "truth_projection.png"))
                                # --- END SAVE ---

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
                            # Create a small confusion matrix plot
                            fig, ax = plt.subplots(figsize=(4, 3))
                            cm = np.array([[TN, FP], [FN, TP]])
                            im = ax.imshow(
                                cm, interpolation='nearest', cmap=plt.cm.Blues)
                            ax.set_title(
                                f"Layer {selected_layer} Confusion Matrix")

                            # Show all ticks and label them
                            ax.set_xticks(np.arange(2))
                            ax.set_yticks(np.arange(2))
                            ax.set_xticklabels(
                                ['Predicted False', 'Predicted True'])
                            ax.set_yticklabels(['Actual False', 'Actual True'])

                            # Rotate tick labels and set alignment
                            plt.setp(ax.get_xticklabels(), rotation=45,
                                     ha="right", rotation_mode="anchor")

                            # Loop over data dimensions and create text annotations
                            for i in range(2):
                                for j in range(2):
                                    ax.text(j, i, cm[i, j], ha="center", va="center",
                                            color="w" if cm[i, j] > cm.max()/2 else "black")

                            plt.tight_layout()

                            # --- SAVE FIGURE ---
                            # Note: fig object for confusion matrix is already defined in this scope
                            save_graph(fig, os.path.join(
                                layer_save_dir, "confusion_matrix.png"))
                            # --- END SAVE ---

                            st.pyplot(fig)
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
                            with torch.no_grad():
                                # Get confidence scores and move everything to CPU
                                confidences = test_outputs.cpu().numpy()

                                # Move tensors to CPU for consistent device
                                test_preds_cpu = test_preds.cpu()
                                test_labels_cpu = test_labels.cpu()

                                # Get the most confident correct and incorrect predictions
                                correct_mask = test_preds_cpu == test_labels_cpu
                                incorrect_mask = ~correct_mask

                                # Make sure all tensors are on the same device
                                test_labels_cpu = test_labels.cpu()
                                correct_mask_cpu = correct_mask.cpu()
                                incorrect_mask_cpu = incorrect_mask.cpu()

                                # Separate by true/false and correct/incorrect
                                true_correct = (test_labels_cpu ==
                                                1) & correct_mask_cpu
                                false_correct = (
                                    test_labels_cpu == 0) & correct_mask_cpu
                                true_incorrect = (test_labels_cpu ==
                                                  1) & incorrect_mask_cpu
                                false_incorrect = (test_labels_cpu ==
                                                   0) & incorrect_mask_cpu

                                # Get the confidence scores for each category
                                true_correct_conf = confidences[true_correct.cpu(
                                )]
                                false_correct_conf = confidences[false_correct.cpu(
                                )]
                                true_incorrect_conf = confidences[true_incorrect.cpu(
                                )]
                                false_incorrect_conf = confidences[false_incorrect.cpu(
                                )]

                                # Get indices for sorting
                                if len(true_correct_conf) > 0:
                                    true_correct_idx = torch.nonzero(
                                        true_correct.cpu()).cpu().numpy().flatten()
                                    true_correct_sorted = true_correct_idx[np.argsort(
                                        -true_correct_conf)]
                                else:
                                    true_correct_sorted = []

                                if len(false_correct_conf) > 0:
                                    false_correct_idx = torch.nonzero(
                                        false_correct.cpu()).cpu().numpy().flatten()
                                    false_correct_sorted = false_correct_idx[np.argsort(
                                        false_correct_conf)]
                                else:
                                    false_correct_sorted = []

                                if len(true_incorrect_conf) > 0:
                                    true_incorrect_idx = torch.nonzero(
                                        true_incorrect.cpu()).cpu().numpy().flatten()
                                    true_incorrect_sorted = true_incorrect_idx[np.argsort(
                                        true_incorrect_conf)]
                                else:
                                    true_incorrect_sorted = []

                                if len(false_incorrect_conf) > 0:
                                    false_incorrect_idx = torch.nonzero(
                                        false_incorrect.cpu()).cpu().numpy().flatten()
                                    false_incorrect_sorted = false_incorrect_idx[np.argsort(
                                        -false_incorrect_conf)]
                                else:
                                    false_incorrect_sorted = []

                            # Display examples
                            with st.expander("Most Confident Correct Predictions"):
                                # Show the most confident true positive and true negative
                                cols = st.columns(2)
                                with cols[0]:
                                    st.markdown(
                                        "**Most confident TRUE prediction (correctly predicted as TRUE)**")
                                    if len(true_correct_sorted) > 0:
                                        idx = true_correct_sorted[0]
                                        st.markdown(
                                            f"Example: `{test_examples[idx]['text']}`")
                                        st.markdown(
                                            f"Confidence: {confidences[idx]:.4f}")
                                    else:
                                        st.markdown("No examples found")

                                with cols[1]:
                                    st.markdown(
                                        "**Most confident FALSE prediction (correctly predicted as FALSE)**")
                                    if len(false_correct_sorted) > 0:
                                        idx = false_correct_sorted[0]
                                        st.markdown(
                                            f"Example: `{test_examples[idx]['text']}`")
                                        st.markdown(
                                            f"Confidence: {1-confidences[idx]:.4f}")
                                    else:
                                        st.markdown("No examples found")

                            with st.expander("Most Confident Incorrect Predictions"):
                                # Show the most confident false positive and false negative
                                cols = st.columns(2)
                                with cols[0]:
                                    st.markdown(
                                        "**Most confident FALSE prediction (incorrectly predicted as TRUE)**")
                                    if len(false_incorrect_sorted) > 0:
                                        idx = false_incorrect_sorted[0]
                                        st.markdown(
                                            f"Example: `{test_examples[idx]['text']}`")
                                        st.markdown(
                                            f"Confidence: {confidences[idx]:.4f}")
                                    else:
                                        st.markdown("No examples found")

                                with cols[1]:
                                    st.markdown(
                                        "**Most confident TRUE prediction (incorrectly predicted as FALSE)**")
                                    if len(true_incorrect_sorted) > 0:
                                        idx = true_incorrect_sorted[0]
                                        st.markdown(
                                            f"Example: `{test_examples[idx]['text']}`")
                                        st.markdown(
                                            f"Confidence: {1-confidences[idx]:.4f}")
                                    else:
                                        st.markdown("No examples found")

        with main_tabs[1]:
            st.info("Analysis for Sparse Autoencoders will be added here.")

        # Add completion message
        st.success(
            f"Analysis complete! Best layer: {np.argmax(results['accuracies'])} with accuracy {max(results['accuracies']):.4f}")

        # Save parameters
        parameters = {
            "model_name": model_name,
            "dataset": dataset_source,
            "output_activation": output_layer,
            "device": device_name,
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "use_control_tasks": use_control_tasks,
            "train_epochs": train_epochs,
            "learning_rate": learning_rate,
            "max_samples": max_samples,
            "test_size": test_size,
            "batch_size": batch_size
        }
        save_json(parameters, os.path.join(run_folder, "parameters.json"))

        # Save results
        results_summary = {
            "accuracy_by_layer": results['accuracies']
        }
        save_json(results_summary, os.path.join(run_folder, "results.json"))

        # Save graphs
        accuracy_plot_path = os.path.join(run_folder, "accuracy_plot.png")
        save_graph(fig_acc if not use_control_tasks else fig_sel,
                   accuracy_plot_path)

        if alignment_strengths:  # Save alignment strength plot if available
            alignment_strength_plot_path = os.path.join(
                run_folder, "alignment_strength_plot.png")
            # The plot is already saved by plot_alignment_strength_by_layer, so this explicit save_graph call might be redundant
            # but ensuring it by having the path defined for records is good.
            # save_graph(fig_align_strength, alignment_strength_plot_path) # Already saved in function

        pca_path = os.path.join(run_folder, "pca_plot.png")
        save_graph(fig_pca,
                   pca_path)

        proj_path = os.path.join(run_folder, "proj_plot.png")
        save_graph(fig_proj,
                   proj_path)

        st.success(f"Run saved successfully! Run ID: {run_id}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        add_log(f"ERROR: {str(e)}")
        import traceback
        st.code(traceback.format_exc(), language="python")
