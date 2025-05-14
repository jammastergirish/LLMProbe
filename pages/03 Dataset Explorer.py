import streamlit as st
import pandas as pd
from datasets import load_dataset
import os
import glob
import zipfile
from io import BytesIO

st.set_page_config(page_title="Dataset Explorer", layout="wide")

st.title("Dataset Explorer")

hf_datasets = ["truefalse", "truthfulqa", "boolq", "fever"]

csv_files = glob.glob('datasets/*.csv')
csv_dataset_options = [os.path.basename(f).replace('.csv', '') for f in csv_files]

all_datasets = hf_datasets + csv_dataset_options

st.sidebar.header("Dataset Selection")

dataset_source = st.sidebar.selectbox("Dataset", all_datasets)

def create_progress_tracker():
    progress_container = st.empty()
    progress_bar = st.progress(0)
    message = st.empty()
    
    def update(progress, status, detail=""):
        progress_container.markdown(f"**Status:** {status}")
        progress_bar.progress(progress)
        message.info(detail)
    
    return update

def load_hf_dataset(dataset_source, progress_callback):
    examples = []
    
    if dataset_source == "truefalse":
        progress_callback(0.1, "Loading TrueFalse dataset...")
        tf_splits = ["animals", "cities", "companies", "inventions", "facts", "elements", "generated"]
        
        for i, split in enumerate(tf_splits):
            split_progress = 0.1 + (i / len(tf_splits)) * 0.8
            progress_callback(split_progress, f"Loading TrueFalse split: {split}")
            
            try:
                split_ds = load_dataset("pminervini/true-false", split=split, trust_remote_code=True)
                split_examples = []
                
                # Process examples from this split
                for row in split_ds["train"]:
                    split_examples.append({
                        "text": row["statement"],
                        "label": row["label"]
                    })
                
                examples.extend(split_examples)
                progress_callback(split_progress + 0.1, f"Loaded {len(split_examples)} examples from {split}")
            except Exception as e:
                progress_callback(split_progress, f"Error loading split {split}: {str(e)}")
    
    elif dataset_source == "truthfulqa":
        progress_callback(0.1, "Loading TruthfulQA dataset...")
        try:
            tq = load_dataset("truthful_qa", "multiple_choice")["validation"]
            total_qa_pairs = 0
            
            for row_idx, row in enumerate(tq):
                if row_idx % 10 == 0:
                    progress = 0.1 + (row_idx / len(tq)) * 0.8
                    progress_callback(progress, f"Processing TruthfulQA example {row_idx+1}/{len(tq)}")
                
                q = row.get("question", "")
                targets = row.get("mc1_targets", {})
                choices = targets.get("choices", [])
                labels = targets.get("labels", [])
                
                for answer, label in zip(choices, labels):
                    examples.append({"text": f"{q} {answer}", "label": label})
                    total_qa_pairs += 1
            
            progress_callback(0.9, f"Loaded TruthfulQA: {total_qa_pairs} examples")
        except Exception as e:
            progress_callback(0.9, f"Error loading TruthfulQA: {str(e)}")
    
    elif dataset_source == "boolq":
        progress_callback(0.1, "Loading BoolQ dataset...")
        try:
            bq = load_dataset("boolq")["train"]
            
            for idx, row in enumerate(bq):
                if idx % 50 == 0:
                    progress = 0.1 + (idx / len(bq)) * 0.8
                    progress_callback(progress, f"Processing BoolQ example {idx+1}/{len(bq)}")
                
                question = row["question"]
                passage = row["passage"]
                label = 1 if row["answer"] else 0
                examples.append({"text": f"{question} {passage}", "label": label})
            
            progress_callback(0.9, f"Loaded BoolQ: {len(examples)} examples")
        except Exception as e:
            progress_callback(0.9, f"Error loading BoolQ: {str(e)}")
    
    elif dataset_source == "fever":
        progress_callback(0.1, "Loading FEVER dataset...")
        try:
            fever = load_dataset("fever", 'v1.0', split="train", trust_remote_code=True)
            
            for idx, row in enumerate(fever):
                if idx % 1000 == 0:
                    progress = 0.1 + (idx / len(fever)) * 0.8
                    progress_callback(progress, f"Processing FEVER example {idx+1}")
                
                label = row.get("label", None)
                claim = row.get("claim", "")
                
                if label == "SUPPORTS":
                    examples.append({"text": claim, "label": 1})
                elif label == "REFUTES":
                    examples.append({"text": claim, "label": 0})
            
            progress_callback(0.9, f"Loaded FEVER: {len(examples)} examples")
        except Exception as e:
            progress_callback(0.9, f"Error loading FEVER: {str(e)}")
    
    elif dataset_source == "azaria-mitchell":
        progress_callback(0.1, "Loading Azaria-Mitchell dataset...")
        try:
            base_dir = "datasets/azaria-mitchell"
            csv_files = glob.glob(f"{base_dir}/*_true_false.csv")
            
            # Check if CSV files exist
            if not csv_files:
                # Try to extract the dataset from the ZIP file
                progress_callback(0.2, "No CSV files found. Attempting to extract from ZIP...")
                extraction_success = extract_azaria_mitchell_dataset()
                
                if extraction_success:
                    progress_callback(0.3, "Dataset extracted successfully")
                    # Get the list of CSV files after extraction
                    csv_files = glob.glob(f"{base_dir}/*_true_false.csv")
                else:
                    progress_callback(1.0, "Failed to extract dataset", 
                                     "Please run the download_azaria_mitchell.py script first")
                    return []
            
            # Process each CSV file in the directory
            for i, csv_path in enumerate(csv_files):
                category = os.path.basename(csv_path).replace('_true_false.csv', '')
                split_progress = 0.3 + (i / len(csv_files)) * 0.6
                progress_callback(split_progress, f"Loading category: {category}")
                
                try:
                    df = pd.read_csv(csv_path)
                    
                    # Check for required columns
                    if 'statement' not in df.columns or 'label' not in df.columns:
                        progress_callback(split_progress, f"Skipping {category}: Missing required columns")
                        continue
                    
                    # Process each row
                    category_examples = []
                    for idx, row in df.iterrows():
                        category_examples.append({
                            "text": row['statement'],
                            "label": int(row['label'])
                        })
                    
                    examples.extend(category_examples)
                    progress_callback(split_progress + 0.1, f"Loaded {category}: {len(category_examples)} examples")
                except Exception as e:
                    progress_callback(split_progress, f"Error loading {category}: {str(e)}")
            
            if not examples:
                progress_callback(1.0, "No examples loaded", 
                                 "Check the dataset structure or re-download the dataset")
                return []
            
            progress_callback(0.95, f"Completed loading {len(examples)} examples")
        except Exception as e:
            progress_callback(0.95, f"Error with Azaria-Mitchell dataset: {str(e)}")
    
    progress_callback(1.0, f"Completed loading {len(examples)} examples")
    return examples

def load_csv_dataset(dataset_name, progress_callback):
    examples = []
    file_path = f"datasets/{dataset_name}.csv"
    
    try:
        progress_callback(0.1, f"Loading CSV file: {file_path}")
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check if it has the expected columns
        if 'statement' not in df.columns or 'label' not in df.columns:
            progress_callback(0.5, "CSV format error", "The CSV file must have 'statement' and 'label' columns")
            return []
        
        # Process rows
        progress_callback(0.6, f"Processing {len(df)} rows...")
        
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                progress = 0.6 + (idx / len(df)) * 0.3
                progress_callback(progress, f"Processing row {idx+1}/{len(df)}")
            
            examples.append({
                "text": row['statement'],
                "label": int(row['label'])
            })
        
        progress_callback(1.0, f"Loaded {len(examples)} examples from CSV")
    except Exception as e:
        progress_callback(1.0, f"Error loading CSV dataset: {str(e)}")
    
    return examples

def display_dataset(dataset_source):
    # Create progress tracker
    progress_callback = create_progress_tracker()
    
    # Check if it's a Hugging Face dataset or a local CSV
    if dataset_source in hf_datasets:
        examples = load_hf_dataset(dataset_source, progress_callback)
    else:
        examples = load_csv_dataset(dataset_source, progress_callback)
    
    if examples:
        # Create dataframe for display
        examples_df = pd.DataFrame([
            {
                "Statement": ex["text"],
                "Label": "True" if ex["label"] == 1 else "False"
            }
            for ex in examples
        ])
        
        # Show basic statistics
        st.header(f"{dataset_source} Dataset")
        st.write(f"Total examples: {len(examples)}")
        
        # Count by label
        true_count = sum(1 for ex in examples if ex["label"] == 1)
        false_count = len(examples) - true_count
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("True statements", true_count)
        with col2:
            st.metric("False statements", false_count)
        
        # Show all examples in a dataframe
        st.header("All Examples")
        st.dataframe(examples_df, use_container_width=True)
        
        # Add download button
        csv = examples_df.to_csv(index=False)
        st.download_button(
            label="Download Examples as CSV",
            data=csv,
            file_name=f"dataset_examples_{dataset_source}.csv",
            mime="text/csv",
        )
    else:
        st.error(
            "Failed to load dataset examples. The dataset might be empty or there was an error loading it."
        )

# If a dataset is selected, display it
if dataset_source:
    with st.spinner(f"Loading dataset: {dataset_source}..."):
        display_dataset(dataset_source)