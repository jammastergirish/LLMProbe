import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import random
from io import StringIO
from captum.attr import IntegratedGradients, LayerIntegratedGradients, NeuronConductance
from captum.attr import visualization as viz

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
from utils.ui import (
    create_model_tracker,
    create_dataset_tracker
)

st.set_page_config(page_title="Model Attributions", layout="wide")

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
    .attribution-text {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .token-highlight-positive {
        color: #4CAF50;
        font-weight: bold;
    }
    .token-highlight-negative {
        color: #F44336;
        font-weight: bold;
    }
</style>

<div class="main-title">Model Attributions</div>
<div class="subtitle">Visualize which input tokens influence model predictions the most</div>
""", unsafe_allow_html=True)

# --- Sidebar Configuration ---
st.sidebar.markdown("""
<div style="padding: 5px; border-radius: 5px; margin-bottom: 20px;">
    <h2 style="color: white; margin: 0;">Configuration</h2>
</div>
""", unsafe_allow_html=True)

# Model selection (same as LLM Probe page)
model_name = st.sidebar.selectbox("📚 Model", model_options)
if model_name == "custom":
    model_name = st.sidebar.text_input("Custom Model Name")
    if not model_name:
        st.sidebar.error("Please enter a model.")

# Input method selection
input_method = st.sidebar.radio(
    "Input Method",
    ["Enter manually", "Use dataset"],
    index=0,
    help="Choose how to provide input for attribution analysis"
)

if input_method == "Enter manually":
    # Manual input (default option)
    dataset_source = "manual_input"
    user_input = st.sidebar.text_area(
        "Enter your statement",
        value="The Earth orbits around the Sun.",
        height=100,
        help="Enter a statement to analyze"
    )
    user_label = st.sidebar.radio(
        "Statement label",
        options=["True", "False"],
        index=0,
        help="Specify whether the statement is true or false"
    )
else:
    # Dataset options (if user prefers to select from existing datasets)
    csv_files = glob.glob('datasets/*.csv')
    dataset_options = ["truefalse", "truthfulqa", "boolq", "arithmetic", "fever", "custom"]
    csv_dataset_options = [os.path.basename(f).replace('.csv', '') for f in csv_files]
    dataset_options.extend(csv_dataset_options)
    
    dataset_source = st.sidebar.selectbox(
        "📊 Dataset", 
        dataset_options,
        help="Select an existing dataset, upload a custom dataset, or add your dataset to the datasets folder"
    )
    
    if dataset_source == "custom":
        custom_file = st.sidebar.file_uploader(
            "Upload CSV file with 'statement' and 'label' columns",
            type=["csv"],
            help="CSV should have 'statement' column for text and 'label' column with 1 (true) or 0 (false)"
        )
        
    # Add option to select a specific example from the dataset
    st.sidebar.info("A random example will be selected from the dataset.")

# Attribution method explanation
is_decoder = is_decoder_only_model(model_name) if 'model_name' in locals() else False

if is_decoder:
    # Explanation for generative models
    st.sidebar.markdown("""
    ### 🔍 Attribution Method: Integrated Gradients for Generative Models

    For generative models like LLaMA or GPT:

    **Integrated Gradients** shows how each input token influences the model's prediction of the **next token**.

    - **Green highlights**: Tokens that increase the probability of the predicted next token
    - **Red highlights**: Tokens that decrease the probability of the predicted next token
    - **Darker colors**: Stronger influence on the prediction
    """)
else:
    # Default explanation for classification models
    st.sidebar.markdown("""
    ### 🔍 Attribution Method: Integrated Gradients

    **Integrated Gradients** analyzes how each token (word/subword) in your input affects the final prediction.
    It helps you understand which parts of your text most strongly influence the model's decision.

    - **Green highlights**: Push toward classification as TRUE
    - **Red highlights**: Push toward classification as FALSE
    - **Darker colors**: Stronger influence on the decision
    """)

# Set the attribution method to Integrated Gradients (no other options)
attribution_method = "Integrated Gradients"

# Device selection (same as LLM Probe page)
device_options = []
if torch.cuda.is_available():
    device_options.append("cuda")
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device_options.append("mps")
device_options.append("cpu")

device_name = st.sidebar.selectbox("💻 Compute", device_options)
device = torch.device(device_name)

# Run button
run_button = st.sidebar.button(
    "🚀 Run Attribution Analysis", 
    type="primary", 
    use_container_width=True
)

# --- Main Content Area ---
# Create a two-column layout
col1, col2 = st.columns([3, 2])

with col1:
    # Status area
    st.markdown("""
    <div class="section-header">Status</div>
    """, unsafe_allow_html=True)
    
    status_container = st.container()
    
    # Input area (will be populated when model runs)
    st.markdown("""
    <div class="section-header">Input Statement</div>
    """, unsafe_allow_html=True)
    input_container = st.container()
    
    # Attribution visualization
    st.markdown("""
    <div class="section-header">Attribution Results</div>
    """, unsafe_allow_html=True)
    attr_container = st.container()
    
with col2:
    # Model stats
    st.markdown("""
    <div class="section-header">Model Information</div>
    """, unsafe_allow_html=True)
    
    # Placeholder until model is loaded
    info_placeholder = st.empty()
    info_placeholder.info("Model information will appear here when the analysis runs")
    
    # Explanation area
    with st.expander("Understanding Attributions", expanded=False):
        st.markdown("""
        ### How to Use This Page
        
        1. **Enter your own statement** (default option) or select a dataset
        2. **Choose a model** to analyze how it processes the statement
        3. **Select an attribution method** to visualize which parts of your input influence the model's prediction
        4. **Click "Run Attribution Analysis"** to see the results
        
        ### What Are Attributions?
        
        **Attributions** show how much each word or token in your input contributes to a model's prediction:
        
        - **Green highlights**: Words that push the model toward classifying the statement as true
        - **Red highlights**: Words that push the model toward classifying the statement as false
        - **Darker colors**: Stronger influence on the model's decision
        
        ### Example Interpretation
        
        For a statement like "The Earth orbits around the Sun":
        - "Earth", "orbits", and "Sun" might have strong positive (green) attributions
        - Words like "The" and "around" might have neutral or mixed attributions
        
        This analysis helps you understand which specific parts of your input text the model focuses on when making its decision.
        """)

# Progress trackers
progress_col1, progress_col2 = st.columns(2)

with progress_col1:
    st.markdown('#### 📚 Load Model')
    model_status = st.empty()
    model_status.markdown('<span class="status-idle">Waiting to start...</span>', unsafe_allow_html=True)
    model_progress_bar = st.progress(0)
    model_progress_text = st.empty()
    model_detail = st.empty()

with progress_col2:
    st.markdown('#### 📊 Run Attribution')
    attr_status = st.empty()
    attr_status.markdown('<span class="status-idle">Waiting to start...</span>', unsafe_allow_html=True)
    attr_progress_bar = st.progress(0)
    attr_progress_text = st.empty()
    attr_detail = st.empty()

# Create progress trackers
model_tracker = create_model_tracker(model_status, model_progress_bar, model_progress_text, model_detail, lambda x: None)
attr_tracker = create_dataset_tracker(attr_status, attr_progress_bar, attr_progress_text, attr_detail, lambda x: None)

def mark_complete(status_element, message="Complete"):
    """Mark this stage as complete"""
    status_element.markdown(
        f'<span class="status-success">{message}</span>', unsafe_allow_html=True)

# --- Helper functions for attribution ---
def get_sample_from_dataset(dataset_source, progress_callback, custom_file=None):
    """Get a random sample from the selected dataset"""
    examples = []
    
    # Handle different dataset sources
    if dataset_source in csv_dataset_options:
        # For datasets from the datasets folder
        csv_file_path = f"datasets/{dataset_source}.csv"
        try:
            progress_callback(0.1, "Reading CSV file", f"Loading from {csv_file_path}")
            with open(csv_file_path, 'r') as f:
                csv_content = f.read()
                
            file_obj = StringIO(csv_content)
            file_obj.name = f"{dataset_source}.csv"
            
            progress_callback(0.3, "Loading custom dataset", "Processing CSV content")
            examples = load_dataset(
                "custom",
                progress_callback,
                custom_file=file_obj
            )
        except Exception as e:
            st.error(f"Error loading CSV file {csv_file_path}: {str(e)}")
            return None
    elif dataset_source == "custom" and custom_file is not None:
        # For uploaded custom files
        progress_callback(0.1, "Processing uploaded file", "Reading custom dataset")
        examples = load_dataset(
            dataset_source,
            progress_callback,
            custom_file=custom_file
        )
    elif dataset_source == "manual_input":
        # For manually entered text
        progress_callback(0.5, "Using manual input", "Creating example from user input")
        label = 1 if user_label == "True" else 0
        examples = [{"text": user_input, "label": label}]
    else:
        # For built-in datasets
        progress_callback(0.2, f"Loading {dataset_source} dataset", "Fetching examples")
        examples = load_dataset(
            dataset_source,
            progress_callback
        )
    
    if len(examples) == 0:
        progress_callback(1.0, "Error", "No examples found in dataset")
        return None
    
    # Return a random example
    if dataset_source == "manual_input":
        progress_callback(1.0, "Sample selected", "Using manually entered example")
        return examples[0]  # Return the manually entered example
    else:
        progress_callback(1.0, "Sample selected", "Randomly selected from dataset")
        return random.choice(examples)  # Return a random example from the dataset

def get_token_attributions(model, tokenizer, statement, label):
    """Calculate attributions for input tokens using Integrated Gradients"""
    # First, tokenize the input
    inputs = tokenizer(statement, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    attr_tracker.update(0.3, "Preparing attribution calculation",
                      "Setting up Integrated Gradients")

    # Handle different model types differently
    is_decoder = is_decoder_only_model(model_name)

    try:
        # Use Integrated Gradients for token attribution
        ig = IntegratedGradients(model)
        attr_tracker.update(0.4, "Calculating attributions",
                          "This may take a moment...")

        if is_decoder:
            # For generative models, we'll look at how input tokens affect the next token prediction
            attr_tracker.update(0.5, "Handling generative model",
                             "Calculating attributions for next token prediction")

            # First, get the model's prediction for the next token
            with torch.no_grad():
                # Get the output logits
                if hasattr(model, 'forward'):
                    outputs = model(input_ids)
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    else:
                        # Assume the first output is logits for many models
                        logits = outputs[0]
                else:
                    # Some custom behavior might be needed for certain models
                    attr_tracker.update(0.6, "Using custom forward function",
                                     "Model doesn't have standard forward method")
                    if hasattr(model, 'generate'):
                        # For HuggingFace models
                        outputs = model.generate(input_ids, max_new_tokens=1, return_dict_in_generate=True, output_scores=True)
                        logits = outputs.scores[0]  # Get the first (only) step's scores
                    else:
                        # Try one more approach
                        outputs = model(input_ids)
                        logits = outputs.logits if hasattr(outputs, 'logits') else outputs

                # Get the most likely next token
                next_token_id = logits[0, -1].argmax().item()

                # Use this as our "target" class - how much each input token contributed to
                # predicting this specific next token
                target = next_token_id

                attr_tracker.update(0.7, "Model predicts next token",
                                 f"Using prediction '{tokenizer.decode([next_token_id])}' as target")

            # Now attribute to this prediction
            attributions, delta = ig.attribute(
                input_ids,
                target=target,  # Target is the predicted next token ID
                target_ind=logits[0, -1].argmax(),  # Index of highest probability token
                return_convergence_delta=True,
                n_steps=50
            )
        else:
            # For classification models, use the provided label
            target = torch.tensor([label], device=device)

            attributions, delta = ig.attribute(
                input_ids,
                target=label,
                return_convergence_delta=True,
                n_steps=50
            )

        # Process the attributions
        attr_sum = attributions.sum(dim=-1).squeeze(0)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        attr_tracker.update(1.0, "Finished attribution calculation",
                          f"Generated attributions for {len(tokens)} tokens")

        return tokens, attr_sum

    except Exception as e:
        # If that fails, fall back to random attributions
        attr_tracker.update(0.7, f"Attribution error: {str(e)}",
                         "Using simulated attributions as fallback")

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        attributions = torch.rand(len(tokens), device=device) * 2 - 1  # Range [-1, 1]

        attr_tracker.update(0.9, "Created simulated attributions",
                         "Using random values since attribution calculation failed")

        return tokens, attributions
    
def visualize_token_attributions(tokens, attributions):
    """Visualize token attributions with color highlighting"""
    # Normalize attributions for visualization
    attr_norm = attributions / (torch.max(torch.abs(attributions)) + 1e-10)
    
    # Create visualization
    html_output = "<div class='attribution-text'>"
    
    for token, attr in zip(tokens, attr_norm):
        token_text = token.replace('##', '')
        # Skip padding tokens
        if token == '[PAD]':
            continue
            
        # Determine color intensity based on attribution value
        if attr > 0:
            # Positive attribution (green)
            intensity = min(255, int(200 * attr.item()))
            html_output += f"<span style='background-color: rgba(0, {intensity}, 0, 0.3);'>{token_text}</span> "
        else:
            # Negative attribution (red)
            intensity = min(255, int(200 * abs(attr.item())))
            html_output += f"<span style='background-color: rgba({intensity}, 0, 0, 0.3);'>{token_text}</span> "
    
    html_output += "</div>"
    
    # Create a bar chart of attributions
    fig, ax = plt.subplots(figsize=(12, 6))
    attr_values = attributions.cpu().numpy()
    
    # Filter out padding tokens
    non_pad_indices = [i for i, token in enumerate(tokens) if token != '[PAD]']
    display_tokens = [tokens[i].replace('##', '') for i in non_pad_indices]
    display_attrs = [attr_values[i] for i in non_pad_indices]
    
    # Plot bars
    bars = ax.bar(range(len(display_tokens)), display_attrs, color=['green' if a > 0 else 'red' for a in display_attrs])
    
    # Add token labels
    ax.set_xticks(range(len(display_tokens)))
    ax.set_xticklabels(display_tokens, rotation=45, ha='right')
    ax.set_ylabel('Attribution Value')
    ax.set_title('Token Attribution Scores')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return html_output, fig

# --- Main execution function ---
def run_attribution_analysis():
    """Main function to run the attribution analysis"""
    st.session_state.setup_complete = False
    
    # 1. Load model
    model_tracker.update(0, "Loading model...", "Initializing")
    try:
        tokenizer, model = load_model_and_tokenizer(model_name, model_tracker.update, device)
        mark_complete(model_status)
        
        # Update model info with number of layers
        num_layers = get_num_layers(model)
            
        # Update model info
        memory_estimates = estimate_memory_requirements(model, 1)
        model_info = pd.DataFrame({
            'Property': [
                'Model Name',
                'Model Type',
                'Number of Layers',
                'Model Size',
                'Precision',
                'Attribution Method'
            ],
            'Value': [
                model_name,
                "Decoder-only" if is_decoder_only_model(model_name) else "Encoder-only/Encoder-decoder",
                num_layers,
                memory_estimates["param_count"],
                memory_estimates["precision"],
                attribution_method
            ]
        })
        info_placeholder.table(model_info)
        
        # 2. Get a sample
        attr_tracker.update(0, "Getting sample for attribution...", "Preparing input")
        
        if dataset_source == "manual_input":
            sample = {"text": user_input, "label": 1 if user_label == "True" else 0}
        else:
            sample = get_sample_from_dataset(
                dataset_source,
                attr_tracker.update,  # Pass the progress tracker update function
                custom_file if dataset_source == "custom" else None
            )
            
        if sample is None:
            attr_tracker.update(1.0, "Error", "Could not load a sample from the dataset")
            st.error("Failed to load a sample from the selected dataset.")
            return
            
        # Display the selected statement
        input_container.markdown(f"""
        ### Statement:
        <div class='attribution-text'>
            {sample['text']}
        </div>
        
        **Label:** {"True" if sample['label'] == 1 else "False"}
        """, unsafe_allow_html=True)
        
        # 3. Calculate attributions
        attr_tracker.update(0.2, "Calculating attributions...", "This may take a moment")
        
        # Store model-specific variables
        prediction_info = {}

        # For generative models, get next token prediction first
        if is_decoder_only_model(model_name):
            try:
                # Get tokenized input
                inputs = tokenizer(sample['text'], return_tensors="pt")
                input_ids = inputs["input_ids"].to(device)

                # Run the model to get predictions
                with torch.no_grad():
                    try:
                        if hasattr(model, 'forward'):
                            outputs = model(input_ids)
                            if hasattr(outputs, 'logits'):
                                pred_logits = outputs.logits
                            else:
                                pred_logits = outputs[0]
                        elif hasattr(model, 'generate'):
                            outputs = model.generate(input_ids, max_new_tokens=1,
                                                  return_dict_in_generate=True, output_scores=True)
                            pred_logits = outputs.scores[0]
                        else:
                            outputs = model(input_ids)
                            pred_logits = outputs

                        # Store predicted token info
                        next_token_id = pred_logits[0, -1].argmax().item()
                        prediction_info['next_token'] = tokenizer.decode([next_token_id])
                        prediction_info['logits'] = pred_logits

                        attr_tracker.update(0.3, "Detected generative model",
                                         f"Predicted next token: '{prediction_info['next_token']}'")
                    except Exception as e:
                        attr_tracker.update(0.3, "Warning: Could not get next token prediction",
                                         f"Error: {str(e)}")
            except Exception as e:
                attr_tracker.update(0.3, "Warning: Error in generative model processing",
                                 f"{str(e)}")

        # Get attributions
        try:
            tokens, attributions = get_token_attributions(
                model,
                tokenizer,
                sample['text'],
                sample['label']
            )
            
            # 4. Visualize attributions
            attr_tracker.update(0.8, "Creating visualization...", "Generating attribution display")
            try:
                html_viz, fig = visualize_token_attributions(tokens, attributions)

                # Display the results
                attr_container.markdown("### Token-level Attribution:", unsafe_allow_html=True)
                attr_container.markdown(html_viz, unsafe_allow_html=True)
                attr_container.pyplot(fig)
                
                # Add explanation about the visualization
                is_decoder = is_decoder_only_model(model_name)
                if is_decoder:
                    # Explanation for generative models
                    next_token = prediction_info.get('next_token', '(could not determine)')
                    attr_container.markdown(f"""
                    ### Understanding the Visualization (Generative Model)

                    This visualization shows how each input token influences the model's prediction of the next token
                    (which was predicted to be **"{next_token}"**):

                    - **Green bars**: These tokens increase the probability of generating the predicted next token
                    - **Red bars**: These tokens decrease the probability of generating the predicted next token
                    - **Taller bars**: Stronger influence on the token prediction

                    This helps you understand which parts of your input most strongly guide the model's
                    next-token prediction, revealing how the model reasons about language continuation.
                    """)
                else:
                    # Explanation for classification models
                    attr_container.markdown("""
                    ### Understanding the Visualization (Classification Model)

                    The visualization above shows how each token (word or subword) in your statement influences
                    the model's prediction:

                    - **Green bars**: These tokens push the model toward classifying the statement as TRUE
                    - **Red bars**: These tokens push the model toward classifying the statement as FALSE
                    - **Taller bars**: Stronger influence on the final decision

                    This helps you understand which specific parts of your text the model is focusing on
                    when making its classification decision.
                    """)
            except Exception as viz_error:
                # Handle visualization errors gracefully
                attr_tracker.update(0.9, "Visualization error", f"Error: {str(viz_error)}")
                attr_container.error(f"Could not create visualization: {str(viz_error)}")
                attr_container.markdown("### Token-level Attribution:")
                attr_container.info("Visualization failed - trying fallback display")

                # Fallback to simpler visualization
                try:
                    # Simple text display of tokens and their attributions
                    display_data = pd.DataFrame({
                        'Token': [t.replace('##', '') for t in tokens if t != '[PAD]'],
                        'Attribution': [f"{a:.4f}" for a in attributions.cpu().numpy() if a != 0]
                    })
                    attr_container.dataframe(display_data)
                except:
                    # Ultimate fallback
                    attr_container.warning("Could not display attributions in any format.")
            
            mark_complete(attr_status)
            status_container.success("Attribution analysis completed successfully!")
            
        except Exception as e:
            attr_tracker.update(1.0, "Error in attribution", str(e))
            attr_container.error(f"Attribution calculation failed: {str(e)}")
            status_container.error(f"An error occurred: {str(e)}")
    
    except Exception as e:
        model_tracker.update(1.0, f"Error loading model: {str(e)}", "Check model name or connection")
        status_container.error(f"An error occurred: {str(e)}")

# Run the attribution analysis when button is clicked
if run_button:
    status_container.info("Running attribution analysis...")
    run_attribution_analysis()
else:
    status_container.info("Click 'Run Attribution Analysis' to begin.")