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

model_options = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-1B", 
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.2-3B",
    "google/gemma-2-2b-it",
    "google/gemma-2-2b",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "mistralai/Mistral-7B-v0.1",
    "deepseek-ai/DeepSeek-V3-Base",
    "bert-base-uncased",
    "bert-large-uncased",
    "roberta-base",
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "custom"
]

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
    activation_memory = (batch_size * seq_length * hidden_dim * num_layers * precision) / (1024**3)
    
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
                st.sidebar.error("CSV must contain 'statement' and 'label' columns")
            else:
                st.sidebar.success(f"Loaded {len(df_preview)} examples")
                st.sidebar.dataframe(df_preview.head(3), use_container_width=True)
                # Reset file pointer for later use
                custom_file.seek(0)
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {str(e)}")

use_control_tasks = st.sidebar.checkbox("Use control tasks", value=True)

def is_decoder_only_model(model_name):
    decoder_keywords = ["gpt", "llama", "mistral", "pythia", "deepseek", "qwen", "gemma"]
    return any(keyword in model_name.lower() for keyword in decoder_keywords)

if is_decoder_only_model(model_name):
    output_layer = st.sidebar.selectbox("🧠 Output Activation", ["resid_post", "attn_out", "mlp_out"])
else:
    output_layer = st.sidebar.selectbox("🧠 Embedding Strategy", ["CLS", "mean", "max", "token_index_0"])

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
    train_epochs = st.number_input("Training epochs", min_value=10, max_value=500, value=100)
    learning_rate = st.number_input("Learning rate", min_value=0.0001, max_value=0.1, value=0.01, format="%.4f")
    max_samples = st.number_input("Max samples per dataset", min_value=100, max_value=10000, value=5000)
    test_size = st.slider("Test split ratio", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    batch_size = st.number_input("Batch size", min_value=1, max_value=64, value=16,
                                 help="Larger batches are faster but use more memory. Use smaller values for large models.")

run_button = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)

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
    model_status.markdown('<span class="status-idle">Waiting to start...</span>', unsafe_allow_html=True)
    model_progress_bar = st.progress(0)
    model_progress_text = st.empty()
    model_detail = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # st.markdown('<div class="progress-container">', unsafe_allow_html=True)
    st.markdown('#### 🔍 Create Representations')
    embedding_status = st.empty()
    embedding_status.markdown('<span class="status-idle">Waiting to start...</span>', unsafe_allow_html=True)
    embedding_progress_bar = st.progress(0)
    embedding_progress_text = st.empty()
    embedding_detail = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

with progress_col2:
    # st.markdown('<div class="progress-container">', unsafe_allow_html=True)
    st.markdown('#### 📊 Load Dataset')
    dataset_status = st.empty()
    dataset_status.markdown('<span class="status-idle">Waiting to start...</span>', unsafe_allow_html=True)
    dataset_progress_bar = st.progress(0)
    dataset_progress_text = st.empty()
    dataset_detail = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # st.markdown('<div class="progress-container">', unsafe_allow_html=True)
    st.markdown('#### 🧠 Train Probe')
    training_status = st.empty()
    training_status.markdown('<span class="status-idle">Waiting to start...</span>', unsafe_allow_html=True)
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
results_container = st.container()
st.markdown("""
<div class="section-header">Results</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["Accuracy Analysis", "PCA Visualization", "Truth Direction Analysis", "Data View"])

accuracy_tab = tabs[0]
pca_tab = tabs[1]
projection_tab = tabs[2] 
data_tab = tabs[3]

# Create empty containers for results
with accuracy_tab:
    accuracy_plot = st.empty()
    selectivity_plot = st.empty()

with pca_tab:
    pca_plot = st.empty()

with projection_tab:
    projection_plot = st.empty()

with data_tab:
    data_display = st.empty()
    layer_select_container = st.empty()
    layer_analysis = st.empty()

def load_model_and_tokenizer(model_name, progress_callback):
    """Load model and tokenizer with progress updates"""
    progress_callback(0.1, "Initializing model loading process...", "Preparing tokenizer and model configuration")
    
    if is_decoder_only_model(model_name):
        progress_callback(0.2, "Detected decoder-only model architecture", 
                         f"Loading {model_name} with TransformerLens for better compatibility")
        
        try:
            # Import necessary libraries
            progress_callback(0.3, "Importing TransformerLens library...", "Setting up model dependencies")
            import transformer_lens
            from transformer_lens import HookedTransformer
            from transformers import AutoTokenizer
            
            # Load tokenizer first
            progress_callback(0.4, "Loading tokenizer...", f"Fetching tokenizer configuration for {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            progress_callback(0.5, "Configuring tokenizer settings...", "Setting padding token and padding side")
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            
            # Now load the model
            progress_callback(0.6, "Loading HookedTransformer model...", f"This may take a while for {model_name}")
            model = HookedTransformer.from_pretrained(model_name, device=device)
            
            # Report model statistics
            n_layers = model.cfg.n_layers
            d_model = model.cfg.d_model
            progress_callback(0.9, f"Model loaded: {n_layers} layers, {d_model} dimensions", 
                             f"Using device: {str(device)}")
            
            progress_callback(1.0, "Model and tokenizer successfully loaded", 
                             f"Ready to process with {model_name}")
            
        except Exception as e:
            progress_callback(1.0, f"Error loading model: {str(e)}", "Check model name or connection")
            raise e
    else:
        try:
            from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
            
            progress_callback(0.3, "Detected encoder or encoder-decoder architecture", 
                             f"Loading {model_name} using Hugging Face Transformers")
            
            progress_callback(0.4, "Loading tokenizer...", f"Fetching tokenizer for {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            progress_callback(0.5, "Configuring tokenizer settings...", "Setting padding token and padding side")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right" if not is_decoder_only_model(model_name) else "left"
            
            progress_callback(0.7, "Loading model...", f"This may take a while for {model_name}")
            model_class = AutoModelForCausalLM if is_decoder_only_model(model_name) else AutoModel
            model = model_class.from_pretrained(model_name, output_hidden_states=True).to(device)
            model.eval()
            
            # Get model statistics
            n_layers = model.config.num_hidden_layers
            d_model = model.config.hidden_size
            progress_callback(0.9, f"Model loaded: {n_layers} layers, {d_model} dimensions", 
                             f"Using device: {str(device)}")
            
            progress_callback(1.0, "Model and tokenizer successfully loaded", 
                             f"Ready to process with {model_name}")
        except Exception as e:
            progress_callback(1.0, f"Error loading model: {str(e)}", "Check model name or connection")
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
                examples.append({"text": f"{question} {passage}", "label": label})
                
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
            
            tf_splits = [
                "animals", "cities", "companies", 
                "inventions", "facts", "elements", "generated"
            ]
            
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
                
                examples.append({"text": row["statement"], "label": row["label"]})
                
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
                    incorrect_sum = a + b + random.choice([i for i in range(-10, 11) if i != 0])
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
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
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
                                seq_len = inputs["attention_mask"][example_idx].sum().item()
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
                                mask = inputs["attention_mask"][example_idx].unsqueeze(-1)
                                token_repr = (layer[example_idx] * mask).sum(dim=0) / mask.sum()
                            else:
                                token_repr = layer[example_idx].mean(dim=0)
                        elif output_layer == "max":
                            # Max pooling
                            if hasattr(inputs, "attention_mask"):
                                # Apply mask to avoid including padding tokens
                                mask = inputs["attention_mask"][example_idx].unsqueeze(-1)
                                masked_layer = layer[example_idx] * mask - 1e9 * (1 - mask)
                                token_repr = masked_layer.max(dim=0).values
                            else:
                                token_repr = layer[example_idx].max(dim=0).values
                        elif output_layer.startswith("token_index_"):
                            # Use specific token index
                            index = int(output_layer.split("_")[-1])
                            seq_len = inputs["attention_mask"][example_idx].sum().item() if hasattr(inputs, "attention_mask") else layer.size(1)
                            safe_index = min(index, seq_len - 1)
                            token_repr = layer[example_idx, safe_index, :]
                        else:
                            raise ValueError(f"Unsupported output layer: {output_layer}")
                        
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
    all_hidden_states = torch.stack(all_hidden_states).to(device)  # [num_examples, num_layers, hidden_dim]
    all_labels = torch.tensor(all_labels).to(device)
    
    # Update to 100%
    progress_callback(1.0, f"Completed processing all {len(examples)} examples", 
                     f"Created tensor of shape {all_hidden_states.shape}")
    
    # Return full tensor or specific layer
    if return_layer is not None:
        return all_hidden_states[:, return_layer, :], all_labels
    else:
        return all_hidden_states, all_labels

# Function to extract hidden states
def get_hidden_states(examples, model, tokenizer, model_name, output_layer, dataset_type="", return_layer=None, progress_callback=None):
    """Extract hidden states without batching"""
    all_hidden_states = []
    labels = []
    
    is_decoder = is_decoder_only_model(model_name)
    is_transformerlens = "HookedTransformer" in str(type(model))
    
    # Get dimensions for pre-allocation if possible
    if is_transformerlens:
        n_layers = model.cfg.n_layers
        d_model = model.cfg.d_model
    else:
        if hasattr(model, "config"):
            if hasattr(model.config, "num_hidden_layers"):
                n_layers = model.config.num_hidden_layers + 1
            else:
                n_layers = 12  # Default fallback
            
            if hasattr(model.config, "hidden_size"):
                d_model = model.config.hidden_size
            else:
                d_model = 768  # Default fallback
    
    progress_callback(0, f"Preparing to process {len(examples)} {dataset_type} examples", 
                     f"Extracting features for each example one at a time")
    
    # Process each example
    for i, ex in enumerate(examples):
        # Show current example processing details
        progress = (i) / len(examples)
        example_text = ex["text"]
        if len(example_text) > 50:
            example_text = example_text[:47] + "..."
        
        progress_callback(progress, 
                         f"Processing {dataset_type} example {i+1}/{len(examples)}", 
                         f"Text: '{example_text}' | Label: {ex['label']}")
        
        # Process the example
        if is_transformerlens:
            tokens = tokenizer.encode(ex["text"], return_tensors="pt").to(model.cfg.device)
            
            # Run with cache to extract all activations
            _, cache = model.run_with_cache(tokens)
            
            # Choose position index (last token for decoder-only, first otherwise)
            pos = -1 if is_decoder else 0
            
            # Get activations from each layer
            layer_outputs = [
                cache[output_layer, layer_idx][0, pos, :]  # shape: (d_model,)
                for layer_idx in range(model.cfg.n_layers)
            ]
            
            # Stack into shape: (num_layers, d_model)
            hidden_stack = torch.stack(layer_outputs)
            
        else:
            if "qwen" in model_name.lower():
                # Wrap input with Qwen's chat format
                messages = [{"role": "user", "content": ex["text"]}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            else:
                prompt = ex["text"]

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                hidden_states = outputs.hidden_states
            
            # Use last token for decoder-only models, first for encoder-only
            if is_decoder:
                cls_embeddings = torch.stack([layer[:, -1, :] for layer in hidden_states])
            elif output_layer == "CLS":
                cls_embeddings = torch.stack([layer[:, 0, :] for layer in hidden_states])
            elif output_layer == "mean":
                cls_embeddings = torch.stack([
                    layer.mean(dim=1) for layer in hidden_states
                ])
            elif output_layer == "max":
                cls_embeddings = torch.stack([
                    layer.max(dim=1).values for layer in hidden_states
                ])
            elif output_layer.startswith("token_index_"):
                index = int(output_layer.split("_")[-1])
                layer_reprs = []

                for layer in hidden_states:
                    seq_len = layer.size(1)  # sequence length
                    safe_index = min(index, seq_len - 1)
                    layer_reprs.append(layer[:, safe_index, :])  # (1, dim)

                cls_embeddings = torch.stack(layer_reprs)  # shape: [num_layers, 1, dim]

                if index >= seq_len:
                    add_log(f"⚠️ Token index {index} out of bounds, using last token at position {seq_len - 1}")
            else:
                raise ValueError(f"Unsupported encoder output layer: {output_layer}")
            
            # [num_layers, hidden_dim]
            hidden_stack = cls_embeddings.squeeze(1)
        
        all_hidden_states.append(hidden_stack)
        labels.append(ex["label"])
        
        # Sleep a tiny bit to allow UI to update
        time.sleep(0.01)
    
    # Convert lists to tensors
    all_hidden_states = torch.stack(all_hidden_states).to(device)
    labels = torch.tensor(labels).to(device)
    
    # Update to 100%
    progress_callback(1.0, f"Completed processing all {len(examples)} {dataset_type} examples", 
                     f"Created tensor of shape {all_hidden_states.shape}")
    
    # Allow slicing if return_layer is specified
    if return_layer is not None:
        return all_hidden_states[:, return_layer, :], labels  # (N, D)
    else:
        return all_hidden_states, labels  # (N, L, D)

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
        raise AttributeError("Cannot determine number of layers for this model")

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
            
            shuffled_labels = train_labels[torch.randperm(train_labels.size(0))]
            ctrl_probe, _ = train_probe(train_feats, shuffled_labels, epochs=epochs, lr=lr)
            
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
    ax.set_title(f"Truth Detection Accuracy per Layer ({model_name})", fontsize=14)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.grid(True, alpha=0.3)
    # Add exact values as text labels
    for i, acc in enumerate(accuracies):
        ax.annotate(f"{acc:.3f}", (i, acc), textcoords="offset points", 
                   xytext=(0,5), ha='center')
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
                   xytext=(0,5), ha='center', color="#4CAF50")
    
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
            preds = (probe(torch.tensor(feats).to(device)) > 0.5).long().cpu().numpy()
        
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
            x_min, x_max = feats_2d[:, 0].min() - 0.5, feats_2d[:, 0].max() + 0.5
            y_min, y_max = feats_2d[:, 1].min() - 0.5, feats_2d[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                 np.linspace(y_min, y_max, 100))
            
            # Transform back to high-dimensional space (approximate)
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            high_dim_grid = pca.inverse_transform(grid_points)
            
            # Apply the probe
            with torch.no_grad():
                Z = probe(torch.tensor(high_dim_grid).float().to(device)).cpu().numpy()
            Z = Z.reshape(xx.shape)
            
            # Plot the decision boundary
            ax.contour(xx, yy, Z, levels=[0.5], colors='k', alpha=0.5, linestyles='--')
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
            projection = torch.matmul(feats, probe.linear.weight[0])  # shape: [N]
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
        ax.hist(false_proj, bins=bins, alpha=0.7, label="False", color="#F44336")
        
        # Add a vertical line at the decision boundary (0.0)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Calculate overlap
        hist_true, _ = np.histogram(true_proj, bins=bins)
        hist_false, _ = np.histogram(false_proj, bins=bins)
        overlap = np.minimum(hist_true, hist_false).sum() / max(1, max(hist_true.sum(), hist_false.sum()))
        
        ax.set_title(f"Layer {layer} (Acc={acc:.3f}, Overlap={overlap:.2f})")
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Only add legend to the first subplot
    if num_layers > 0:
        axs[0].legend(fontsize=8)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle("Projection onto Truth Direction per Layer", fontsize=20, y=0.98)
    return fig

# Update progress functions with enhanced UI
def update_model_progress(progress, message, details=""):
    model_status.markdown('<span class="status-running">Running</span>', unsafe_allow_html=True)
    model_progress_bar.progress(progress)
    model_progress_text.markdown(f"**{message}**")
    model_detail.text(details)
    add_log(f"Load Model ({progress:.0%}): {message} - {details}")

def update_dataset_progress(progress, message, details=""):
    dataset_status.markdown('<span class="status-running">Running</span>', unsafe_allow_html=True)
    dataset_progress_bar.progress(progress)
    dataset_progress_text.markdown(f"**{message}**")
    dataset_detail.text(details)
    add_log(f"Load Dataset ({progress:.0%}): {message} - {details}")

def update_embedding_progress(progress, message, details=""):
    embedding_status.markdown('<span class="status-running">Running</span>', unsafe_allow_html=True)
    embedding_progress_bar.progress(progress)
    embedding_progress_text.markdown(f"**{message}**")
    embedding_detail.text(details)
    add_log(f"Create Representations ({progress:.0%}): {message} - {details}")

def update_training_progress(progress, message, details=""):
    training_status.markdown('<span class="status-running">Running</span>', unsafe_allow_html=True)
    training_progress_bar.progress(progress)
    training_progress_text.markdown(f"**{message}**")
    training_detail.text(details)
    add_log(f"Train Probe ({progress:.0%}): {message} - {details}")

def mark_complete(status_element, message="Complete"):
    status_element.markdown(f'<span class="status-success">{message}</span>', unsafe_allow_html=True)

def save_fig(fig, filename):
    """Save figure to disk"""
    fig.savefig(filename)
    add_log(f"Saved figure to {filename}")


# Main app logic
if run_button:
    run_folder, run_id = create_run_folder(model_name = model_name, dataset = dataset_source)

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
        update_embedding_progress(0, "Extracting embeddings for TRAIN set...", "Initializing")
        
        # Extract train embeddings
        train_hidden_states, train_labels = get_hidden_states_batched(
            train_examples, model, tokenizer, model_name, output_layer, 
            dataset_type="TRAIN", progress_callback=update_embedding_progress, batch_size=batch_size
        )
        
        # Extract test embeddings
        update_embedding_progress(0, "Extracting embeddings for TEST set...", "Initializing")
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
        with accuracy_tab:
            # Selectivity plot (if using control tasks)
            if use_control_tasks and results['selectivities']:
                fig_sel = plot_selectivity_by_layer(
                    results['selectivities'], results['accuracies'], 
                    results['control_accuracies'], model_name, dataset_source
                )
                selectivity_plot.pyplot(fig_sel)
            else:
                fig_acc = plot_accuracy_by_layer(results['accuracies'], model_name, dataset_source)
                accuracy_plot.pyplot(fig_acc)
        
        with pca_tab:
            # PCA grid
            pca_plot.info("Generating PCA visualization...")
            fig_pca = plot_pca_grid(test_hidden_states, test_labels, results['probes'], model_name, dataset_source)
            pca_plot.pyplot(fig_pca)
        
        with projection_tab:
            # Truth projection grid
            projection_plot.info("Generating truth projection histograms...")
            fig_proj = plot_truth_projections(test_hidden_states, test_labels, results['probes'])
            projection_plot.pyplot(fig_proj)
        
        with data_tab:
            # Display numeric results
            data_display.subheader("Layer-wise Metrics")
            acc_df = pd.DataFrame({
                'Layer': range(num_layers),
                'Accuracy': results['accuracies'],
                'Loss': results['test_losses']
            })
            if use_control_tasks:
                acc_df['Control Accuracy'] = results['control_accuracies']
                acc_df['Selectivity'] = results['selectivities']

            data_display.dataframe(acc_df)

            # Find best layer
            best_layer = np.argmax(results['accuracies'])
            best_acc = results['accuracies'][best_layer]
            data_display.success(
                f"Best layer: {best_layer} with accuracy {best_acc:.4f}"
            )

            # Create one tab per layer
            layer_tabs = data_display.tabs([f"Layer {i}" for i in range(num_layers)])

            # Display analysis for the selected layer tab
            for i, layer_tab in enumerate(layer_tabs):
                with layer_tab:
                    selected_layer = i

                    # Show details for selected layer
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader(f"Layer {selected_layer} Details")
                        probe = results['probes'][selected_layer]

                        # Extract test features for this layer
                        test_feats = test_hidden_states[:, selected_layer, :]

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
                            accuracy = (TP + TN) / (TP + TN + FP + FN)
                            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                            f1 = 2 * precision * recall / \
                                (precision + recall) if (precision + recall) > 0 else 0

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
                            true_proj = projection[test_labels == 1].cpu().numpy()
                            false_proj = projection[test_labels == 0].cpu().numpy()

                            # Create histogram
                            fig_proj_individual, ax_proj = plt.subplots(figsize=(8, 3))
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
                            ax_proj.set_xlabel("Projection onto Truth Direction")
                            ax_proj.set_ylabel("Count")
                            ax_proj.legend()

                            st.pyplot(fig_proj_individual)

                    with col2:
                        st.subheader("Confusion Matrix")
                        # Create a small confusion matrix plot
                        fig, ax = plt.subplots(figsize=(4, 3))
                        cm = np.array([[TN, FP], [FN, TP]])
                        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                        ax.set_title(f"Layer {selected_layer} Confusion Matrix")

                        # Show all ticks and label them
                        ax.set_xticks(np.arange(2))
                        ax.set_yticks(np.arange(2))
                        ax.set_xticklabels(['Predicted False', 'Predicted True'])
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
                        st.pyplot(fig)

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
                            true_correct = (test_labels_cpu == 1) & correct_mask_cpu
                            false_correct = (test_labels_cpu == 0) & correct_mask_cpu
                            true_incorrect = (test_labels_cpu ==
                                            1) & incorrect_mask_cpu
                            false_incorrect = (test_labels_cpu ==
                                            0) & incorrect_mask_cpu

                            # Get the confidence scores for each category
                            true_correct_conf = confidences[true_correct.cpu()]
                            false_correct_conf = confidences[false_correct.cpu()]
                            true_incorrect_conf = confidences[true_incorrect.cpu()]
                            false_incorrect_conf = confidences[false_incorrect.cpu()]

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
                                    st.markdown(f"Confidence: {confidences[idx]:.4f}")
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
                                    st.markdown(f"Confidence: {confidences[idx]:.4f}")
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
        # Add completion message
        st.success(f"Analysis complete! Best layer: {best_layer} with accuracy {best_acc:.4f}")

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