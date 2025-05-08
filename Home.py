import streamlit as st
import torch
import platform
import psutil
import os
import subprocess

st.set_page_config(
    page_title="LLM Analysis Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 LLM Probe")

st.markdown("""
### Welcome

XX
""")

st.markdown("### 🖥️ Your Device")

# --- CPU Info ---


def get_nice_cpu_name():
    system = platform.system()
    if system == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True
            )
            return result.stdout.strip()
        except Exception:
            pass
    elif system == "Windows":
        return platform.processor()
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        except Exception:
            pass
    return platform.processor() or platform.machine() or "Unknown CPU"

# --- macOS Version Mapping ---


def get_os_name():
    def get_mac_os_codename(version: str) -> str:
        codename_map = {
            "15": "Sequoia",
            "14": "Sonoma",
            "13": "Ventura",
            "12": "Monterey",
            "11": "Big Sur",
            "10.15": "Catalina",
            "10.14": "Mojave",
            "10.13": "High Sierra",
            "10.12": "Sierra",
            "10.11": "El Capitan",
            "10.10": "Yosemite",
        }
        for key in codename_map:
            if version.startswith(key):
                return codename_map[key]
        return ""

    system = platform.system()
    if system == "Darwin":
        try:
            product_version = subprocess.check_output(
                ["sw_vers", "-productVersion"]
            ).decode().strip()
            codename = get_mac_os_codename(product_version)
            return f"macOS {codename} {product_version}" if codename else f"macOS {product_version}"
        except Exception:
            return "macOS (version unknown)"
    return f"{system} {platform.release()}"


# --- Gather All Info ---
cpu_info = get_nice_cpu_name()
num_threads = torch.get_num_threads()
total_cores = os.cpu_count()
mem_gb = psutil.virtual_memory().total / (1024 ** 3)
os_info = get_os_name()
python_version = platform.python_version()
torch_version = torch.__version__

# --- GPU Info ---
if torch.cuda.is_available():
    gpu = f"CUDA ({torch.cuda.get_device_name(0)})"
else:
    gpu = "Not available (Try [RunPod](https://runpod.io?ref=avnw83xb))"

mps = "Available via Apple Silicon" if getattr(torch.backends, "mps",
                                               None) and torch.backends.mps.is_available() else "Not available"

# --- Display as Markdown Table ---
st.markdown(f"""
| Component        | Details                                |
|------------------|-----------------------------------------|
| **CPU**          | {cpu_info} — {num_threads} threads / {total_cores} cores |
| **RAM**          | {mem_gb:.2f} GB                         |
| **GPU (CUDA)**   | {gpu}                                   |
| **MPS**          | {mps}                                   |
| **OS**           | {os_info}                               |
| **Python**       | {python_version}                        |
| **PyTorch**      | {torch_version}                         |
""")
