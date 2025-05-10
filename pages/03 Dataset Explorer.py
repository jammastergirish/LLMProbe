import streamlit as st
import pandas as pd
from datasets import load_dataset
import random

st.set_page_config(page_title="Dataset Explorer", layout="wide")

# Main title
st.title("Dataset Explorer")

# Sidebar configuration
st.sidebar.header("Dataset Selection")

dataset_source = st.sidebar.selectbox("Dataset",
                                      ["truefalse", "truthfulqa", "boolq", "fever"])

max_samples = st.sidebar.slider(
    "Max samples", min_value=10, max_value=500, value=100, step=10)

# Function to load dataset samples


def load_dataset_samples(dataset_source, max_samples=100):
    examples = []

    # TrueFalse dataset
    if dataset_source == "truefalse":
        try:
            tf_splits = ["animals", "cities", "companies",
                         "inventions", "facts", "elements", "generated"]

            for split in tf_splits:
                split_ds = load_dataset(
                    "pminervini/true-false", split=split, trust_remote_code=True)
                samples_per_split = max(1, int(max_samples / len(tf_splits)))

                for j, row in enumerate(split_ds["train"]):
                    if j >= samples_per_split:
                        break

                    examples.append({
                        "dataset": f"truefalse_{split}",
                        "text": row["statement"],
                        "label": row["label"],
                        "label_text": "True" if row["label"] == 1 else "False"
                    })

                    if len(examples) >= max_samples:
                        break
        except Exception as e:
            st.error(f"Error loading TrueFalse: {str(e)}")

    # TruthfulQA dataset
    if dataset_source == "truthfulqa":
        try:
            tq = load_dataset("truthful_qa", "multiple_choice")["validation"]

            for i, row in enumerate(tq):
                if i >= max_samples:
                    break

                q = row.get("question", "")
                targets = row.get("mc1_targets", {})
                choices = targets.get("choices", [])
                labels = targets.get("labels", [])

                for j, (answer, label) in enumerate(zip(choices, labels)):
                    if j >= 2:  # Limit to 2 choices per question
                        continue

                    examples.append({
                        "dataset": "truthfulqa",
                        "text": f"{q} {answer}",
                        "label": label,
                        "label_text": "True" if label == 1 else "False"
                    })

                if len(examples) >= max_samples:
                    break
        except Exception as e:
            st.error(f"Error loading TruthfulQA: {str(e)}")

    # BoolQ dataset
    if dataset_source == "boolq":
        try:
            bq = load_dataset("boolq")["train"]

            for i, row in enumerate(bq):
                if i >= max_samples:
                    break

                question = row["question"]
                passage = row["passage"]
                # Truncate passage to avoid very long examples
                if len(passage) > 300:
                    passage = passage[:297] + "..."

                label = 1 if row["answer"] else 0

                examples.append({
                    "dataset": "boolq",
                    "text": f"Q: {question} A: Based on the passage: {passage}",
                    "label": label,
                    "label_text": "True" if label == 1 else "False"
                })

                if len(examples) >= max_samples:
                    break
        except Exception as e:
            st.error(f"Error loading BoolQ: {str(e)}")

    # FEVER dataset
    if dataset_source == "fever":
        try:
            fever = load_dataset(
                "fever", 'v1.0', split="train", trust_remote_code=True)

            for i, row in enumerate(fever):
                if i >= max_samples:
                    break

                label = row.get("label", None)
                claim = row.get("claim", "")

                if label == "SUPPORTS":
                    examples.append({
                        "dataset": "fever",
                        "text": claim,
                        "label": 1,
                        "label_text": "True"
                    })
                elif label == "REFUTES":
                    examples.append({
                        "dataset": "fever",
                        "text": claim,
                        "label": 0,
                        "label_text": "False"
                    })

                if len(examples) >= max_samples:
                    break
        except Exception as e:
            st.error(f"Error loading FEVER: {str(e)}")

    return examples


# Load dataset button
if st.sidebar.button("Load Dataset"):
    with st.spinner("Loading dataset examples..."):
        examples = load_dataset_samples(dataset_source, max_samples)

    if examples:
        # Create dataframe for display
        examples_df = pd.DataFrame([
            {
                "Dataset": ex["dataset"],
                "Text": ex["text"],
                "Label": ex["label_text"]
            }
            for ex in examples
        ])

        # Show basic statistics
        st.header("Dataset Statistics")
        st.write(f"Total examples: {len(examples)}")

        # Count by label
        true_count = sum(1 for ex in examples if ex["label"] == 1)
        false_count = len(examples) - true_count

        col1, col2 = st.columns(2)
        with col1:
            st.metric("True statements", true_count)
        with col2:
            st.metric("False statements", false_count)

        # Show examples
        st.header("Dataset Examples")
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
            "Failed to load dataset examples. Please check your selection or try again.")
else:
    st.info(
        "Please select a dataset from the sidebar and click 'Load Dataset' to begin.")
