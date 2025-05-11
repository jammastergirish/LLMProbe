import torch
import time


def is_decoder_only_model(model_name):
    """Check if model is a decoder-only model based on its name."""
    # Convert model_name to string to ensure we can call lower() on it
    if not isinstance(model_name, str):
        model_name = str(model_name)

    decoder_keywords = ["gpt", "llama", "mistral",
                        "pythia", "deepseek", "qwen", "gemma"]
    return any(keyword in model_name.lower() for keyword in decoder_keywords)


def get_num_layers(model):
    """Get the number of layers in a model."""
    if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
        return model.config.num_hidden_layers + 1
    elif hasattr(model, "cfg") and hasattr(model.cfg, "n_layers"):
        return model.cfg.n_layers
    else:
        raise AttributeError(
            "Cannot determine number of layers for this model")


def load_model_and_tokenizer(model_name, progress_callback, device=torch.device("cpu")):

    if not isinstance(model_name, str):
        model_name = str(model_name)

    progress_callback(0.1, "Initializing model loading process...",
                      "Preparing tokenizer and model configuration")

    if "llama-4" in model_name.lower():
        use_transformerlens = False
    else:
        use_transformerlens = is_decoder_only_model(model_name)

    if use_transformerlens:
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


def get_hidden_states_batched(examples, model, tokenizer, model_name, output_layer,
                              dataset_type="", return_layer=None, progress_callback=None,
                              batch_size=16, device=torch.device("cpu")):
    """Extract hidden states with batching for better performance"""
    import math

    # Ensure model_name is a string
    if not isinstance(model_name, str):
        model_name = str(model_name)
    
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
        progress_callback(progress, f"Processing {dataset_type} batch {batch_idx//batch_size + 1}/{num_batches}",
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
    progress_callback(1.0, f"Completed processing all {dataset_type} {len(examples)} examples",
                      f"Created tensor of shape {all_hidden_states.shape}")

    # Return full tensor or specific layer
    if return_layer is not None:
        return all_hidden_states[:, return_layer, :], all_labels
    else:
        return all_hidden_states, all_labels