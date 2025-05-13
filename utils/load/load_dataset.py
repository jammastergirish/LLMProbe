import random
from collections import Counter

def count_categories(examples):
    """Count the number of examples in each category based on label value

    Args:
        examples: List of examples, each with a 'label' field

    Returns:
        dict: Dictionary with label values as keys and counts as values
    """
    labels = [example['label'] for example in examples]
    return Counter(labels)

def load_dataset(dataset_source, progress_callback, max_samples=5000, custom_file=None, tf_splits=None, am_splits=None):
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

            if tf_splits is None:
                tf_splits = ["animals", "cities", "companies",
                             "inventions", "facts", "elements", "generated"]

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

    if dataset_source in ["azaria-mitchell", "all"]:
        progress_callback(0.6, "Preparing to load Azaria-Mitchell dataset...",
                          "Initializing azaria-mitchell true-false dataset from local files")
        try:
            import pandas as pd
            import os

            # Default categories if none are specified
            if am_splits is None:
                am_splits = ["animals", "cities", "companies",
                              "inventions", "facts", "elements", "generated"]

            progress_callback(0.65, "Loading Azaria-Mitchell dataset splits...",
                              f"Processing {len(am_splits)} dataset categories")

            start_examples = len(examples)
            base_dir = "datasets/azaria-mitchell"

            # Check if directory exists
            if not os.path.exists(base_dir):
                os.makedirs(base_dir, exist_ok=True)
                progress_callback(0.7, "Creating directory for Azaria-Mitchell dataset",
                                  f"Created directory {base_dir}")

            # Process each category
            for i, category in enumerate(am_splits):
                split_progress = 0.65 + (i / len(am_splits)) * 0.1
                progress_callback(split_progress, f"Loading Azaria-Mitchell split: {category}",
                                 f"Processing category {i+1}/{len(am_splits)}")

                csv_path = os.path.join(base_dir, f"{category}_true_false.csv")

                if os.path.exists(csv_path):
                    try:
                        # Load CSV file
                        category_df = pd.read_csv(csv_path)

                        # Check for required columns
                        if 'statement' not in category_df.columns or 'label' not in category_df.columns:
                            progress_callback(split_progress, f"Skipping {category} file: missing columns",
                                           f"CSV must contain 'statement' and 'label' columns")
                            continue

                        # Process each row
                        category_examples = []
                        for idx, row in enumerate(category_df.itertuples()):
                            category_examples.append({
                                "text": row.statement,
                                "label": int(row.label)
                            })

                        examples.extend(category_examples)
                        progress_callback(split_progress, f"Loaded {category} file",
                                       f"Added {len(category_examples)} examples from {category}")

                    except Exception as e:
                        progress_callback(split_progress, f"Error loading {category} file: {str(e)}",
                                       f"Skipping this category")
                else:
                    progress_callback(split_progress, f"File not found: {csv_path}",
                                   f"Skipping this category. Please download the dataset.")

            # Check if any examples were added
            if len(examples) - start_examples > 0:
                # Limit if needed
                if len(examples) - start_examples > max_samples and dataset_source != "all":
                    examples = examples[:start_examples + max_samples]

                progress_callback(0.85, f"Loaded Azaria-Mitchell dataset: Added {len(examples) - start_examples} examples",
                              f"Total examples so far: {len(examples)}")
            else:
                progress_callback(0.85, "No Azaria-Mitchell examples loaded",
                                "Please ensure the dataset files are properly downloaded to 'datasets/azaria-mitchell/[category]_true_false.csv'")

        except Exception as e:
            progress_callback(0.85, f"Error loading Azaria-Mitchell dataset: {str(e)}",
                          "Continuing with other datasets if selected")

    progress_callback(1.0, f"Prepared {len(examples)} labeled examples for probing",
                      f"Dataset preparation complete with {len(examples)} total examples")
    return examples