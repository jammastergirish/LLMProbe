# LLM Probe

LLM Probe is a tool for analyzing and visualizing representations in language models. It allows users to:

- Train linear probes to detect signals across different model layers
- Visualize how information is encoded in model representations
- Compare accuracy and selectivity across different model layers
- Analyze PCA visualizations of hidden state activations
- Explore how models separate statements
- View feature activations and neuron alignment with signals

LLM Probe supports various models and datasets, making it easy to explore how different language models encode and process factual information.

## Running the App

### Locally

You'll need the [HuggingFace CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli), and to be signed in (`huggingface-cli login`), in order to use the models listed.

Once that's done, run:

`./run.sh`

This will install the `uv` package manager, if you don't already have it, and start the Streamlit app.

(You may need to run `chmod +x run.sh` first.)

### Remotely

I'd recommend setting up an instance on [RunPod](https://runpod.io?ref=avnw83xb) as follows:

- Create a new pod and select your GPU.
- Press 'Edit Template' and add `8501` to the list of exposed HTTP ports.
- You'll likely also want to increase the persistent disk space (Volume Disk), assuming you want to use larger Large Language Models, so around 1-10 TB.
- Click, "Set Overrides," and, "Deploy On-Demand," and wait for the deployment to complete.
- You'll then want to SSH into the instance. Click, "Connect," follow the SSH instructions (setting up an SSH key, and pasting the public key into RunPod settings), and then follow the instructions to Connect.
- Once you're in via SSH, you'll need to run `git clone https://github.com/jammastergirish/LLMProbe && cd LLMProbe && ./runpod_firstrun.sh`, which will clone the repository, install the `uv` package manager, the HuggingFace CLI, and will prompt you to enter your HuggingFace token.
- It'll then run the Streamlit app, to which you can connect in your browser via instructions in the Connect panel at RunPod.
- On future runs, you can simply run `./run.sh` on the instance.