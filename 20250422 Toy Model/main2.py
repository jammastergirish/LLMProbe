# /// script
# dependencies = [
# "streamlit",
#   "torch",
#   "transformers",
#   "datasets",
#   "matplotlib",
#   "scikit-learn",
#   "protobuf",
#   "tiktoken",
#   "blobfile",
#  "accelerate",
# "transformer-lens"
# ]
# ///


import streamlit as st
import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import random
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="LLM Truth Detection Probing", layout="wide")

st.title("Truth Detection in Language Models")
st.write("This app analyzes how language models encode truth vs. false statements across different layers.")

# Sidebar for model selection
st.sidebar.header("Configuration")

model_options = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-1B", 
    "meta-llama/Llama-3.2-3B-Instruct-SpinQuant_INT4_EO8",
    "mistralai/Mistral-7B-v0.1",
    "deepseek-ai/DeepSeek-V3-Base",
    "bert-base-uncased",
    "bert-large-uncased",
    "roberta-base",
    "gpt2"
]

# App inputs
model_name = st.sidebar.selectbox("Select model", model_options)
dataset_source = st.sidebar.selectbox("Select dataset", 
                                    ["truefalse", "truthfulqa", "boolq", "arithmetic", "all"])
use_control_tasks = st.sidebar.checkbox("Use control tasks", value=True)
output_layer = st.sidebar.selectbox("Output layer", 
                                   ["resid_post", "attn_out", "mlp_out"])

# Device selection
device_options = []
if torch.cuda.is_available():
    device_options.append("cuda")
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device_options.append("mps")
device_options.append("cpu")

device_name = st.sidebar.selectbox("Device", device_options)
device = torch.device(device_name)

# When the user clicks this button, the analysis will run
run_button = st.sidebar.button("Run Analysis", type="primary")

# Function to check if a model is decoder-only
def is_decoder_only_model(model_name):
    decoder_keywords = ["gpt", "llama", "mistral", "pythia", "deepseek"]
    return any(keyword in model_name.lower() for keyword in decoder_keywords)

# Create placeholder for model loading
model_load_placeholder = st.empty()

# Create placeholder for dataset loading
dataset_load_placeholder = st.empty()

# Progress bar for loading model
model_progress = st.empty()

# Progress bar for extracting embeddings
embedding_progress = st.empty()

# Progress bar for training probes
training_progress = st.empty()

# Placeholders for results
results_container = st.container()
accuracy_plot = results_container.empty()
selectivity_plot = results_container.empty()
pca_plot = results_container.empty()
projection_plot = results_container.empty()

def load_model_and_tokenizer(model_name, progress_callback):
    """Load model and tokenizer with progress updates"""
    progress_callback(0.2, "Loading tokenizer...")
    
    if is_decoder_only_model(model_name):
        import transformer_lens
        from transformer_lens import HookedTransformer
        
        progress_callback(0.5, "Loading model with HookedTransformer...")
        # Show model loading
        tokenizer = None  # This will be updated
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            
            progress_callback(0.7, "Loading HookedTransformer model...")
            model = HookedTransformer.from_pretrained(model_name, device=device)
            progress_callback(1.0, f"Model loaded: {model_name}")
        except Exception as e:
            progress_callback(1.0, f"Error loading model: {str(e)}")
            raise e
    else:
        try:
            from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
            
            progress_callback(0.4, "Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right" if not is_decoder_only_model(model_name) else "left"
            
            progress_callback(0.7, "Loading model...")
            model_class = AutoModelForCausalLM if is_decoder_only_model(model_name) else AutoModel
            model = model_class.from_pretrained(model_name, output_hidden_states=True).to(device)
            model.eval()
            progress_callback(1.0, "Model loaded successfully")
        except Exception as e:
            progress_callback(1.0, f"Error loading model: {str(e)}")
            raise e
    
    return tokenizer, model

def load_dataset(dataset_source, progress_callback):
    """Load dataset with progress updates"""
    examples = []
    
    if dataset_source in ["truthfulqa", "all"]:
        progress_callback(0.2, "Loading TruthfulQA dataset...")
        try:
            from datasets import load_dataset
            tq = load_dataset("truthful_qa", "multiple_choice")["validation"]
            
            for row in tq:
                q = row.get("question", "")
                targets = row.get("mc1_targets", {})
                choices = targets.get("choices", [])
                labels = targets.get("labels", [])
                for answer, label in zip(choices, labels):
                    examples.append({"text": f"{q} {answer}", "label": label})
            
            progress_callback(0.4, f"Loaded TruthfulQA: {len(examples)} examples")
        except Exception as e:
            progress_callback(0.4, f"Error loading TruthfulQA: {str(e)}")
    
    if dataset_source in ["boolq", "all"]:
        progress_callback(0.5, "Loading BoolQ dataset...")
        try:
            from datasets import load_dataset
            bq = load_dataset("boolq")["train"]
            
            for row in bq:
                question = row["question"]
                passage = row["passage"]
                label = 1 if row["answer"] else 0
                examples.append({"text": f"{question} {passage}", "label": label})
            
            progress_callback(0.7, f"Loaded BoolQ: Total {len(examples)} examples")
        except Exception as e:
            progress_callback(0.7, f"Error loading BoolQ: {str(e)}")
    
    if dataset_source in ["truefalse", "all"]:
        progress_callback(0.8, "Loading TrueFalse dataset...")
        try:
            from datasets import load_dataset, concatenate_datasets
            
            tf_splits = [
                "animals", "cities", "companies", 
                "inventions", "facts", "elements", "generated"
            ]
            
            datasets_list = []
            for split in tf_splits:
                split_ds = load_dataset("pminervini/true-false", split=split)
                datasets_list.append(split_ds)
            
            tf = concatenate_datasets(datasets_list)
            
            for row in tf:
                examples.append({"text": row["statement"], "label": row["label"]})
            
            progress_callback(0.9, f"Loaded TrueFalse: Total {len(examples)} examples")
        except Exception as e:
            progress_callback(0.9, f"Error loading TrueFalse: {str(e)}")
    
    if dataset_source in ["arithmetic", "all"]:
        progress_callback(0.95, "Generating arithmetic dataset...")
        
        def generate_arithmetic_dataset(n=5000):
            data = []
            while len(data) < n:
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
        
        arithmetic = generate_arithmetic_dataset(5000)
        examples.extend(arithmetic)
    
    progress_callback(1.0, f"Prepared {len(examples)} labeled examples for probing")
    return examples

# Function to extract hidden states
def get_hidden_states(examples, model, tokenizer, model_name, output_layer, return_layer=None, progress_callback=None):
    """Extract hidden states with progress updates"""
    all_hidden_states = []
    labels = []
    
    is_decoder = is_decoder_only_model(model_name)
    is_transformerlens = "HookedTransformer" in str(type(model))
    
    for i, ex in enumerate(examples):
        # Update progress every 50 examples
        if progress_callback and i % 50 == 0:
            progress = (i + 1) / len(examples)
            progress_callback(progress, f"Processing example {i+1}/{len(examples)}")
        
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
            inputs = tokenizer(ex["text"], return_tensors="pt", truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                hidden_states = outputs.hidden_states
            
            # Use last token for decoder-only models, first for encoder-only
            if is_decoder:
                cls_embeddings = torch.stack([layer[:, -1, :] for layer in hidden_states])
            else:
                cls_embeddings = torch.stack([layer[:, 0, :] for layer in hidden_states])
            
            # [num_layers, hidden_dim]
            hidden_stack = cls_embeddings.squeeze(1)
        
        all_hidden_states.append(hidden_stack)
        labels.append(ex["label"])
    
    # Update to 100%
    if progress_callback:
        progress_callback(1.0, "Completed processing all examples")
    
    # Shape: (N, L, D)
    all_hidden_states = torch.stack(all_hidden_states).to(device)
    labels = torch.tensor(labels).to(device)
    
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
                            num_layers, use_control_tasks, progress_callback=None):
    """Train probes across all layers and evaluate performance"""
    probes = []
    accuracies = []
    control_accuracies = []
    selectivities = []
    
    for layer in range(num_layers):
        # Update progress
        if progress_callback:
            progress = (layer + 1) / num_layers
            progress_callback(progress, f"Training probe for layer {layer+1}/{num_layers}")
        
        train_feats = train_hidden_states[:, layer, :]
        test_feats = test_hidden_states[:, layer, :]
        
        # Train probe
        probe, loss = train_probe(train_feats, train_labels)
        probes.append(probe)
        
        # Evaluate on test set
        with torch.no_grad():
            preds = (probe(test_feats) > 0.5).long()
            acc = (preds == test_labels).float().mean().item()
            accuracies.append(acc)
        
        # Control task (if enabled)
        if use_control_tasks:
            shuffled_labels = train_labels[torch.randperm(train_labels.size(0))]
            ctrl_probe, _ = train_probe(train_feats, shuffled_labels)
            
            with torch.no_grad():
                ctrl_preds = (ctrl_probe(test_feats) > 0.5).long()
                ctrl_acc = (ctrl_preds == test_labels).float().mean().item()
                control_accuracies.append(ctrl_acc)
                
                selectivity = acc - ctrl_acc
                selectivities.append(selectivity)
    
    # Update to 100%
    if progress_callback:
        progress_callback(1.0, "Completed training all probes")
    
    results = {
        'probes': probes,
        'accuracies': accuracies,
        'control_accuracies': control_accuracies if use_control_tasks else None,
        'selectivities': selectivities if use_control_tasks else None
    }
    
    return results

def plot_accuracy_by_layer(accuracies, model_name, dataset_source):
    """Plot accuracy by layer"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(accuracies)), accuracies, marker="o")
    ax.set_title(f"Truth Detection Accuracy per Layer ({model_name})")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.grid(True)
    plt.tight_layout()
    return fig

def plot_selectivity_by_layer(selectivities, model_name, dataset_source):
    """Plot selectivity by layer"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(selectivities)), selectivities, marker="o", label="Selectivity")
    ax.set_title(f"Selectivity per Layer ({model_name})")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Selectivity = Real Acc - Control Acc")
    ax.grid(True)
    plt.tight_layout()
    return fig

def plot_pca_grid(test_hidden_states, test_labels, probes, model_name, dataset_source):
    """Generate PCA grid visualization"""
    num_layers = test_hidden_states.shape[1]
    cols = math.ceil(math.sqrt(num_layers))
    rows = math.ceil(num_layers / cols)
    
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    if rows * cols > 1:
        axs = axs.flatten()
    
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
        
        # Get correct subplot
        if rows * cols > 1:
            ax = axs[layer]
        else:
            ax = axs
        
        # Plot PCA
        ax.scatter(
            feats_2d[lbls == 1][:, 0],
            feats_2d[lbls == 1][:, 1],
            color="green",
            alpha=0.6,
            label="True",
            s=10,
        )
        ax.scatter(
            feats_2d[lbls == 0][:, 0],
            feats_2d[lbls == 0][:, 1],
            color="red",
            alpha=0.6,
            label="False",
            s=10,
        )
        ax.set_title(f"Layer {layer} (Acc={acc:.2f})")
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.suptitle("PCA of CLS embeddings by Layer", fontsize=16, y=1.02)
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
        
        ax = axs[layer]
        ax.hist(
            projection[lbls == 1].cpu(), bins=30, alpha=0.6, label="True", color="green"
        )
        ax.hist(projection[lbls == 0].cpu(), bins=30, alpha=0.6, label="False", color="red")
        ax.set_title(f"Layer {layer}")
        ax.set_xticks([])
        ax.set_yticks([])
    
    if num_layers > 0:
        axs[0].legend()
    
    plt.tight_layout()
    plt.suptitle("Projection onto Truth Direction per Layer", fontsize=20, y=1.02)
    return fig

# Function to update progress in the Streamlit app
def update_model_progress(progress, message):
    model_load_placeholder.text(message)
    model_progress.progress(progress)

def update_dataset_progress(progress, message):
    dataset_load_placeholder.text(message)
    model_progress.progress(progress)

def update_embedding_progress(progress, message):
    embedding_progress.text(message)
    embedding_progress.progress(progress)

def update_training_progress(progress, message):
    training_progress.text(message)
    training_progress.progress(progress)

# Main app logic
if run_button:
    # Reset progress displays
    model_load_placeholder.empty()
    dataset_load_placeholder.empty()
    model_progress.empty()
    embedding_progress.empty()
    training_progress.empty()
    
    try:
        # 1. Load model with progress
        model_load_placeholder.text("Loading model...")
        model_progress.progress(0)
        tokenizer, model = load_model_and_tokenizer(model_name, update_model_progress)
        
        # 2. Load dataset with progress
        dataset_load_placeholder.text("Loading dataset...")
        model_progress.empty()
        model_progress = st.progress(0)
        examples = load_dataset(dataset_source, update_dataset_progress)
        
        # Split data
        train_examples, test_examples = train_test_split(
            examples, test_size=0.2, random_state=42, shuffle=True
        )
        
        st.write(f"Train: {len(train_examples)} examples, Test: {len(test_examples)} examples")
        
        # 3. Extract embeddings with progress
        embedding_progress.text("Extracting embeddings...")
        embedding_progress.progress(0)
        
        # Extract train embeddings
        train_hidden_states, train_labels = get_hidden_states(
            train_examples, model, tokenizer, model_name, output_layer, 
            progress_callback=update_embedding_progress
        )
        
        # Extract test embeddings
        test_hidden_states, test_labels = get_hidden_states(
            test_examples, model, tokenizer, model_name, output_layer,
            progress_callback=update_embedding_progress
        )
        
        # 4. Train probes with progress
        training_progress.text("Training probes...")
        training_progress.progress(0)
        
        num_layers = get_num_layers(model)
        results = train_and_evaluate_model(
            train_hidden_states, train_labels, 
            test_hidden_states, test_labels,
            num_layers, use_control_tasks,
            progress_callback=update_training_progress
        )
        
        # 5. Plot and display results
        with results_container:
            st.subheader("Results")
            
            # Accuracy plot
            fig_acc = plot_accuracy_by_layer(results['accuracies'], model_name, dataset_source)
            accuracy_plot.pyplot(fig_acc)
            
            # Selectivity plot (if using control tasks)
            if use_control_tasks and results['selectivities']:
                fig_sel = plot_selectivity_by_layer(results['selectivities'], model_name, dataset_source)
                selectivity_plot.pyplot(fig_sel)
            
            # PCA grid
            fig_pca = plot_pca_grid(test_hidden_states, test_labels, results['probes'], model_name, dataset_source)
            pca_plot.pyplot(fig_pca)
            
            # Truth projection grid
            fig_proj = plot_truth_projections(test_hidden_states, test_labels, results['probes'])
            projection_plot.pyplot(fig_proj)
            
            # Display numeric results
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Layer-wise Accuracy")
                acc_df = pd.DataFrame({
                    'Layer': range(num_layers),
                    'Accuracy': results['accuracies']
                })
                if use_control_tasks:
                    acc_df['Control Accuracy'] = results['control_accuracies']
                    acc_df['Selectivity'] = results['selectivities']
                st.dataframe(acc_df)
                
                # Find best layer
                best_layer = np.argmax(results['accuracies'])
                best_acc = results['accuracies'][best_layer]
                st.success(f"Best layer: {best_layer} with accuracy {best_acc:.4f}")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("Click 'Run Analysis' to start the process.")

# Add some explanation about the app
with st.expander("About this app"):
    st.write("""
    ### LLM Truth Detection Probing
    
    This app demonstrates how language models encode truth vs. falsehood in their internal representations.
    
    **How it works:**
    1. The app loads a language model and a dataset of true/false statements
    2. It extracts the hidden states from each layer of the model
    3. For each layer, it trains a linear probe to detect if a statement is true or false
    4. It analyzes the results with various visualizations
    
    **Datasets:**
    - TruthfulQA: Multiple-choice questions with true and false answers
    - BoolQ: Yes/no questions with true and false answers 
    - TrueFalse: Factual statements that are either true or false
    - Arithmetic: Simple addition statements that are either true or false
    
    **Visualizations:**
    - Accuracy by layer: Shows how well each layer encodes truth
    - Selectivity by layer: The accuracy difference between real and control tasks
    - PCA projections: 2D visualizations of the model's representations
    - Truth projections: Distributions of projections onto the truth direction
    """)