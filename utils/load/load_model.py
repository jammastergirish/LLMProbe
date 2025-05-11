import torch
import time
import os
import warnings


def is_decoder_only_model(model_name):
    """Check if model is a decoder-only model based on its name."""
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


def normalize_model_name(model_name):
    """Normalize model name to ensure it's properly formatted for loading"""
    if model_name is None:
        raise ValueError("Model name cannot be None")

    # Convert to string if it's not already
    model_name = str(model_name).strip()

    if not model_name:
        raise ValueError("Model name cannot be empty")

    # Handle common model prefixes
    if "/" not in model_name:
        # Add appropriate organization prefix based on model family
        if model_name.lower().startswith("qwen"):
            model_name = f"Qwen/{model_name}"
        elif model_name.lower().startswith("gemma-2") or model_name.lower().startswith("gemma2"):
            model_name = f"google/{model_name}"
        elif model_name.lower().startswith("deepseek") and "coder" not in model_name.lower():
            model_name = f"deepseek-ai/{model_name}"
        elif model_name.lower().startswith("deepseek-coder"):
            model_name = f"deepseek-ai/{model_name}"
        elif model_name.lower().startswith("llama-3") or model_name.lower().startswith("llama3"):
            model_name = f"meta-llama/{model_name}"

    return model_name


def load_model_and_tokenizer(model_name, progress_callback, device=torch.device("cpu")):
    """Load model and tokenizer with progress updates"""
    progress_callback(0.05, "Initializing model loading process...",
                      "Normalizing model name")

    # First, normalize the model name
    try:
        model_name = normalize_model_name(model_name)
        progress_callback(0.1, f"Model name normalized to: {model_name}",
                          "Preparing tokenizer and model configuration")
    except Exception as e:
        progress_callback(0.1, f"Error normalizing model name: {str(e)}",
                          "Will attempt to continue with original name")

    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
    import os

    # Check for problematic models - specifically the 3B variant
    skip_transformer_lens = False
    if ("llama-3.2-3b" in model_name.lower() or "llama-3-2-3b" in model_name.lower() or
        "qwen" in model_name.lower() or "gemma" in model_name.lower() or
            "deepseek" in model_name.lower()):
        skip_transformer_lens = True
        progress_callback(0.2, f"Detected model family requiring special handling: {model_name}",
                          "Using HuggingFace directly - skipping TransformerLens for compatibility")

    # Load tokenizer first (universal approach)
    progress_callback(0.3, "Loading tokenizer...",
                      f"Fetching tokenizer for {model_name}")

    tokenizer = None
    # Try multiple tokenizer loading strategies in sequence
    tokenizer_errors = []

    # Check if we're using a well-known model that needs special handling
    is_qwen = "qwen" in model_name.lower()
    is_gemma = "gemma" in model_name.lower()
    is_deepseek = "deepseek" in model_name.lower()

    # Special handling for models with known tokenizer issues
    if is_qwen or is_gemma or is_deepseek:
        progress_callback(0.32, f"Detected model requiring special tokenizer handling: {model_name}",
                          "Using specialized tokenizer loading")
        try:
            # For these models, we need to ensure trust_remote_code is True
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=False,
                revision="main"
            )
            progress_callback(0.35, "Special tokenizer loading successful",
                              f"Using specialized approach for {model_name}")
        except Exception as e:
            tokenizer_errors.append(f"Special loading: {str(e)}")
            progress_callback(0.36, f"Special tokenizer loading failed: {str(e)}",
                              "Trying direct FastTokenizer loading")
            try:
                # Try importing FastTokenizer directly for these models
                if is_qwen:
                    progress_callback(
                        0.37, "Attempting direct Qwen tokenizer import", "")
                    try:
                        from transformers import PreTrainedTokenizer
                        from transformers.models.auto.tokenization_auto import get_tokenizer_config
                        # Try to get the tokenizer config
                        tokenizer_config = get_tokenizer_config(model_name)
                        if tokenizer_config:
                            # Load the QWenTokenizer class directly
                            from transformers import AutoTokenizer
                            tokenizer = AutoTokenizer.from_pretrained(
                                model_name,
                                trust_remote_code=True
                            )
                        else:
                            # Fallback to a known working tokenizer
                            tokenizer = AutoTokenizer.from_pretrained(
                                "Qwen/Qwen2-7B-Instruct",
                                trust_remote_code=True
                            )
                    except Exception as qe:
                        tokenizer_errors.append(
                            f"Qwen direct loading: {str(qe)}")
                elif is_gemma:
                    progress_callback(
                        0.37, "Attempting direct Gemma tokenizer loading", "")
                    try:
                        # Try loading from google directly
                        tokenizer = AutoTokenizer.from_pretrained(
                            "google/gemma-2b",
                            trust_remote_code=True
                        )
                    except Exception as ge:
                        tokenizer_errors.append(
                            f"Gemma direct loading: {str(ge)}")
                elif is_deepseek:
                    progress_callback(
                        0.37, "Attempting direct DeepSeek tokenizer loading", "")
                    try:
                        # Try using a different model version
                        tokenizer = AutoTokenizer.from_pretrained(
                            "deepseek-ai/deepseek-moe-16b-base",
                            trust_remote_code=True
                        )
                    except Exception as de:
                        tokenizer_errors.append(
                            f"DeepSeek direct loading: {str(de)}")
            except Exception as fe:
                tokenizer_errors.append(f"Fast tokenizer import: {str(fe)}")
    else:
        # Standard loading for other models
        # Strategy 1: Standard loading
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True)
        except Exception as e:
            tokenizer_errors.append(f"Standard loading: {str(e)}")
            progress_callback(0.35, f"Tokenizer warning: {str(e)}",
                              "Trying alternative tokenizer approach...")

            # Strategy 2: Slow tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    use_fast=False,
                    trust_remote_code=True
                )
            except Exception as e2:
                tokenizer_errors.append(f"Slow tokenizer: {str(e2)}")
                progress_callback(0.4, f"Second tokenizer attempt failed: {str(e2)}",
                                  "Trying more specialized approach...")

                # Strategy 3: Special flags for problematic models
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        use_fast=False,
                        trust_remote_code=True,
                        legacy=True
                    )
                except Exception as e3:
                    tokenizer_errors.append(f"Legacy approach: {str(e3)}")
                    progress_callback(0.45, f"Third tokenizer attempt failed",
                                      "Trying one final approach...")

                    # Strategy 4: Last resort for extreme cases
                    try:
                        # For Llama 3.2-3B specifically, try the most compatible approach
                        if "llama-3.2-3b" in model_name.lower() or "llama-3-2-3b" in model_name.lower():
                            # Sometimes using the 1B tokenizer works
                            alt_model_name = model_name.replace(
                                "-3b", "-1b").replace("3B", "1B")
                            progress_callback(0.47, f"Attempting to use {alt_model_name} tokenizer",
                                              "This may help with compatibility issues")
                            tokenizer = AutoTokenizer.from_pretrained(
                                alt_model_name,
                                use_fast=False,
                                trust_remote_code=True
                            )
                        else:
                            # Try with specific revision for other models
                            tokenizer = AutoTokenizer.from_pretrained(
                                model_name,
                                use_fast=False,
                                trust_remote_code=True,
                                revision="main"
                            )
                    except Exception as e4:
                        tokenizer_errors.append(f"Final attempt: {str(e4)}")

    # If we still don't have a tokenizer after all attempts,
    # try using a default one as a last resort
    if tokenizer is None:
        progress_callback(0.48, "All specialized tokenizer attempts failed",
                          "Attempting to use a generic tokenizer as fallback")
        try:
            # Use a very basic tokenizer as fallback
            tokenizer = AutoTokenizer.from_pretrained(
                "gpt2",  # GPT-2 tokenizer is widely compatible
                trust_remote_code=True
            )
            progress_callback(0.49, "Using GPT-2 tokenizer as fallback",
                              "Note: This may affect results but allows processing to continue")
        except Exception as e5:
            all_errors = "\n".join(
                tokenizer_errors + [f"Generic fallback: {str(e5)}"])
            error_msg = f"All tokenizer loading attempts failed. Errors:\n{all_errors}"
            progress_callback(
                0.5, error_msg, "Critical error - cannot continue")
            raise RuntimeError(error_msg)

    if tokenizer is None:
        raise RuntimeError("Failed to initialize tokenizer after all attempts")

    progress_callback(0.5, "Configuring tokenizer settings...",
                      "Setting padding token and padding side")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" if not is_decoder_only_model(
        model_name) else "left"

    # Attempt TransformerLens loading if appropriate
    if is_decoder_only_model(model_name) and not skip_transformer_lens:
        try:
            progress_callback(0.6, "Detected decoder-only model architecture",
                              f"Attempting to load {model_name} with TransformerLens")

            # Import TransformerLens library
            import transformer_lens
            from transformer_lens import HookedTransformer

            # Standard loading for most models
            model = HookedTransformer.from_pretrained(
                model_name,
                device=device,
                fold_ln=False,  # Helps with some models
                center_writing_weights=False,
                center_unembed=False,
                tokenizer=tokenizer  # Pass the tokenizer explicitly
            )

            # Report model statistics
            n_layers = model.cfg.n_layers
            d_model = model.cfg.d_model

            progress_callback(0.9, f"TransformerLens model loaded: {n_layers} layers, {d_model} dimensions",
                              f"Using device: {str(device)}")

            return tokenizer, model

        except Exception as e:
            # If TransformerLens fails, fall back to Hugging Face
            progress_callback(
                0.7, f"TransformerLens loading failed: {str(e)}",
                "Falling back to standard Hugging Face implementation"
            )
            warnings.warn(
                f"TransformerLens loading failed: {str(e)}, using HuggingFace instead")

    # Standard Hugging Face loading (fallback for decoder models or primary for encoder models)
    progress_callback(0.75, "Loading model with Hugging Face Transformers...",
                      f"This may take a while for {model_name}")

    try:
        model_class = AutoModelForCausalLM if is_decoder_only_model(
            model_name) else AutoModel

        # Special handling for specific model families
        load_kwargs = {
            'output_hidden_states': True,
            'trust_remote_code': True,
            'torch_dtype': torch.float16 if torch.cuda.is_available() else torch.float32,
            'low_cpu_mem_usage': True
        }

        # Model-specific handling
        if "qwen" in model_name.lower():
            progress_callback(0.77, "Detected Qwen model",
                              "Adding special parameters for Qwen compatibility")

            # Qwen models often need these additional parameters
            load_kwargs.update({
                'device_map': 'auto',  # Let the model decide on device mapping
                'revision': 'main',    # Use the main branch
                'trust_remote_code': True,
                'quantization_config': None  # Disable quantization
            })
        elif "gemma" in model_name.lower():
            progress_callback(0.77, "Detected Gemma model",
                              "Using special parameters for Gemma")
            load_kwargs.update({
                'torch_dtype': torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                'attn_implementation': "flash_attention_2" if torch.cuda.is_available() else "eager"
            })
        elif "deepseek" in model_name.lower():
            progress_callback(0.77, "Detected DeepSeek model",
                              "Using special parameters for DeepSeek")

            # Some DeepSeek models need flash attention disabled
            load_kwargs.update({
                'trust_remote_code': True,
                'revision': 'main',
                'use_flash_attention_2': False  # Disable flash attention which can cause issues
            })

        # Try to load the model
        try:
            model = model_class.from_pretrained(
                model_name, **load_kwargs).to(device)
        except Exception as load_error:
            # If there's a "not found" error, try with organization prefix
            if "not found" in str(load_error).lower() or "404" in str(load_error):
                normalized_name = normalize_model_name(model_name)
                if normalized_name != model_name:
                    progress_callback(
                        0.78, f"Model not found, trying with normalized name: {normalized_name}", "")
                    model = model_class.from_pretrained(
                        normalized_name, **load_kwargs).to(device)
                else:
                    raise load_error
            # If there's a CUDA error, try with CPU
            elif "cuda" in str(load_error).lower() and device.type == "cuda":
                progress_callback(
                    0.78, "CUDA error detected, falling back to CPU", "")
                device = torch.device("cpu")
                load_kwargs['torch_dtype'] = torch.float32
                model = model_class.from_pretrained(
                    model_name, **load_kwargs).to(device)
            else:
                raise load_error

        model.eval()

        # Get model statistics
        if hasattr(model, "config"):
            if hasattr(model.config, "num_hidden_layers"):
                n_layers = model.config.num_hidden_layers
            elif hasattr(model.config, "n_layers"):
                n_layers = model.config.n_layers
            elif hasattr(model.config, "num_layers"):
                n_layers = model.config.num_layers
            else:
                # Default value if we can't determine
                n_layers = 12
                warnings.warn(
                    f"Could not determine number of layers, using default: {n_layers}")

            if hasattr(model.config, "hidden_size"):
                d_model = model.config.hidden_size
            elif hasattr(model.config, "d_model"):
                d_model = model.config.d_model
            elif hasattr(model.config, "n_embd"):
                d_model = model.config.n_embd
            else:
                # Default value if we can't determine
                d_model = 768
                warnings.warn(
                    f"Could not determine hidden size, using default: {d_model}")
        else:
            # Fallback defaults
            n_layers = 12
            d_model = 768
            warnings.warn(
                "Model has no config attribute, using default dimensions")

        progress_callback(0.9, f"HuggingFace model loaded: {n_layers} layers, {d_model} dimensions",
                          f"Using device: {str(device)}")

        progress_callback(1.0, "Model and tokenizer successfully loaded",
                          f"Ready to process with {model_name}")

        return tokenizer, model

    except Exception as e:
        # Special handling for common errors
        error_str = str(e)
        if "does not appear to have a file named pytorch_model.bin" in error_str and "qwen" in model_name.lower():
            # Try a different approach specific to Qwen models
            progress_callback(0.8, "Qwen model file not found with standard method",
                              "Trying alternative loading approach for Qwen")
            try:
                # Sometimes using specific revision helps
                load_kwargs.update({
                    'revision': 'refs/pr/1',  # Try a different branch
                    'use_safetensors': False  # Don't require safetensors format
                })
                # Convert to RoPE format if needed
                if "instruct" in model_name.lower():
                    from huggingface_hub import try_to_load_from_cache, hf_hub_download
                    # Check if it's available in a different format
                    repo_id = f"Qwen/{model_name.split('/')[-1]}" if "/" in model_name else f"Qwen/{model_name}"
                    config_file = hf_hub_download(
                        repo_id, "config.json", revision="main")
                    load_kwargs.update({'config': config_file})

                model = model_class.from_pretrained(
                    model_name, **load_kwargs).to(device)
                model.eval()

                progress_callback(0.9, "Qwen model loaded with special handling",
                                  f"Using device: {str(device)}")
                return tokenizer, model
            except Exception as e2:
                progress_callback(0.85, f"Alternative Qwen loading failed: {str(e2)}",
                                  "Trying one more approach - direct URL")
                try:
                    # As a last resort, try loading by direct model ID
                    if "/" not in model_name:
                        direct_model_name = f"Qwen/{model_name}"
                    else:
                        direct_model_name = model_name

                    model = model_class.from_pretrained(
                        direct_model_name,
                        trust_remote_code=True,
                        output_hidden_states=True,
                        device_map="auto"
                    ).to(device)
                    model.eval()

                    progress_callback(0.95, "Qwen model loaded with direct approach",
                                      f"Using device: {str(device)}")
                    return tokenizer, model
                except Exception as e3:
                    progress_callback(1.0, f"All Qwen loading approaches failed: {str(e3)}\nOriginal error: {str(e)}",
                                      "Check model name or connection")
                    raise RuntimeError(
                        f"Failed to load Qwen model after multiple attempts: {str(e3)}")

        progress_callback(
            1.0, f"Error loading model: {str(e)}", "Check model name or connection")
        raise e


def get_hidden_states_batched(examples, model, tokenizer, model_name, output_layer,
                              dataset_type="", return_layer=None, progress_callback=None,
                              batch_size=16, device=torch.device("cpu")):
    """Extract hidden states with batching for better performance"""
    import math

    all_hidden_states = []
    all_labels = []

    is_decoder = is_decoder_only_model(model_name)
    is_transformerlens = "HookedTransformer" in str(type(model))

    # Get model dimensions
    if is_transformerlens:
        n_layers = model.cfg.n_layers
        d_model = model.cfg.d_model
    else:
        # For HuggingFace models, handle different model configs
        if hasattr(model, "config"):
            if hasattr(model.config, "num_hidden_layers"):
                n_layers = model.config.num_hidden_layers + \
                    (1 if not is_transformerlens else 0)
            elif hasattr(model.config, "n_layers"):
                n_layers = model.config.n_layers
            elif hasattr(model.config, "num_layers"):
                n_layers = model.config.num_layers
            else:
                # Default value if we can't determine
                n_layers = 12
                warnings.warn(
                    f"Could not determine number of layers, using default: {n_layers}")

            if hasattr(model.config, "hidden_size"):
                d_model = model.config.hidden_size
            elif hasattr(model.config, "d_model"):
                d_model = model.config.d_model
            elif hasattr(model.config, "n_embd"):
                d_model = model.config.n_embd
            else:
                # Default value if we can't determine
                d_model = 768
                warnings.warn(
                    f"Could not determine hidden size, using default: {d_model}")
        else:
            # Fallback defaults
            n_layers = 12
            d_model = 768
            warnings.warn(
                "Model has no config attribute, using default dimensions")

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
                try:
                    tokens = tokenizer.encode(
                        text, return_tensors="pt").to(device)
                    _, cache = model.run_with_cache(tokens)

                    pos = -1 if is_decoder else 0

                    # Handle different cache structures
                    layer_outputs = []
                    for layer_idx in range(n_layers):
                        try:
                            # Try standard TransformerLens cache format
                            cache_key = (output_layer, layer_idx)
                            if cache_key in cache:
                                layer_outputs.append(
                                    cache[cache_key][0, pos, :])
                            else:
                                # Try alternative formats
                                alt_keys = [
                                    f"blocks.{layer_idx}.{output_layer}",
                                    f"layers.{layer_idx}.{output_layer}",
                                    f"{output_layer}_{layer_idx}"
                                ]
                                for key in alt_keys:
                                    if key in cache:
                                        layer_outputs.append(
                                            cache[key][0, pos, :])
                                        break
                                else:
                                    # If no key works, use zeros as placeholder
                                    layer_outputs.append(
                                        torch.zeros(d_model, device=device))
                                    warnings.warn(
                                        f"Could not find layer {layer_idx} in cache")
                        except Exception as e:
                            warnings.warn(
                                f"Error accessing layer {layer_idx}: {str(e)}")
                            layer_outputs.append(
                                torch.zeros(d_model, device=device))

                    hidden_stack = torch.stack(layer_outputs)
                    batch_hidden_states.append(hidden_stack)
                except Exception as e:
                    warnings.warn(
                        f"Error processing example {text_idx}: {str(e)}")
                    # Create a dummy tensor to keep processing going
                    dummy = torch.zeros((n_layers, d_model), device=device)
                    batch_hidden_states.append(dummy)
        else:
            # Standard transformers batching
            try:
                if "qwen" in model_name.lower():
                    # Special handling for Qwen chat models
                    encoded_inputs = []
                    for text in batch_texts:
                        try:
                            # Try chat template first
                            messages = [{"role": "user", "content": text}]
                            prompt = tokenizer.apply_chat_template(
                                messages, tokenize=False, add_generation_prompt=False)
                            encoded_inputs.append(prompt)
                        except Exception:
                            # Fall back to direct text if chat template fails
                            encoded_inputs.append(text)

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

                    # Handle different output formats
                    if hasattr(outputs, "hidden_states"):
                        hidden_states = outputs.hidden_states
                    elif isinstance(outputs, tuple) and len(outputs) > 1:
                        # Some models return hidden_states as second element
                        hidden_states = outputs[1]
                    else:
                        raise ValueError(
                            f"Model doesn't output hidden states in expected format")

                    batch_hidden_states = []

                    # Process each example in the batch
                    for example_idx in range(len(batch)):
                        # Extract embeddings for each layer
                        example_layers = []

                        for layer_idx, layer in enumerate(hidden_states):
                            # Get representation based on selected strategy
                            if is_decoder:
                                # For decoder models, use the last token
                                if "attention_mask" in inputs:
                                    # Get position of last non-padding token
                                    seq_len = inputs["attention_mask"][example_idx].sum(
                                    ).item()
                                    token_repr = layer[example_idx,
                                                       seq_len-1, :]
                                else:
                                    # Just use last token
                                    token_repr = layer[example_idx, -1, :]
                            elif output_layer == "CLS":
                                # Use first token for BERT-like models
                                token_repr = layer[example_idx, 0, :]
                            elif output_layer == "mean":
                                # Mean pooling (average all tokens)
                                if "attention_mask" in inputs:
                                    # Only consider non-padding tokens
                                    mask = inputs["attention_mask"][example_idx].unsqueeze(
                                        -1)
                                    token_repr = (
                                        layer[example_idx] * mask).sum(dim=0) / mask.sum()
                                else:
                                    token_repr = layer[example_idx].mean(dim=0)
                            elif output_layer == "max":
                                # Max pooling
                                if "attention_mask" in inputs:
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
                                seq_len = inputs["attention_mask"][example_idx].sum(
                                ).item() if "attention_mask" in inputs else layer.size(1)
                                safe_index = min(index, seq_len - 1)
                                token_repr = layer[example_idx, safe_index, :]
                            else:
                                raise ValueError(
                                    f"Unsupported output layer: {output_layer}")

                            example_layers.append(token_repr)

                        # Stack layers for this example
                        example_stack = torch.stack(example_layers)
                        batch_hidden_states.append(example_stack)
            except Exception as e:
                warnings.warn(f"Error processing batch: {str(e)}")
                # Create dummy tensors for the entire batch
                batch_hidden_states = [
                    torch.zeros((n_layers, d_model), device=device)
                    for _ in range(len(batch))
                ]

        # Collect results from this batch
        all_hidden_states.extend(batch_hidden_states)
        all_labels.extend(batch_labels)

        # Small sleep to allow UI to update
        time.sleep(0.01)

    # Convert to tensors
    all_hidden_states = torch.stack(all_hidden_states).to(
        device)  # [num_examples, num_layers, hidden_dim]

    # Convert labels to tensor, handling potential string labels
    try:
        all_labels = torch.tensor(all_labels).to(device)
    except:
        # For non-numeric labels, keep as list
        pass

    # Update to 100%
    progress_callback(1.0, f"Completed processing all {dataset_type} {len(examples)} examples",
                      f"Created tensor of shape {all_hidden_states.shape}")

    # Return full tensor or specific layer
    if return_layer is not None:
        return all_hidden_states[:, return_layer, :], all_labels
    else:
        return all_hidden_states, all_labels
