# LLM Probe

XXX

You'll need the [HuggingFace CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli), and to be signed (`huggingface-cli login`) in with a token, to use the models listed in LLM Probe.

## Running Locally

To run locally, you simply need to run the following command in Terminal. This will install the `uv` package manager, if you don't already have it, and start the Streamlit app.

`./run.sh`

(You may need to run `chmod +x run.sh` first.)

## Running Remotely

You'll likely want to run the app remotely in order to harness more compute power. I'd recommend setting up an instance on [RunPod](https://runpod.io?ref=avnw83xb) as follows:

- Create a new pod and select your GPU.
- Press 'Edit Template' and add `8501` to the list of exposed HTTP ports.
- You'll likely also want to increase the persistent disk space (Volume Disk), assuming you want to use larger Large Language Models, so around 1 TB.
- Click, "Set Overrides," and, "Deploy On-Demand," and wait for the deployment to complete.
- You'll then want to SSH into the instance. Click, "Connect," follow the SSH instructions (setting up an SSH key, and pasting the public key into RunPod settings), and then follow the instructions to Connect.
- Once you're in via SSH, you'll need to run `git clone https://github.com/jammastergirish/LLMProbe`. Then run `./runpod_firstrun.sh`, which will install the `uv` package manager, the HuggingFace CLI, and will prompt you to enter your HuggingFace token.
- It'll then run the Streamlit app, to which you can connect in your browser via instructions in the Connect panel at RunPod.
- On future runs, you can simply run `./run.sh` on the instance.

