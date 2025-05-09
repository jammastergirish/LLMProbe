import random

def load_dataset(dataset_source, progress_callback, max_samples=5000, custom_file=None, tf_splits=None):
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