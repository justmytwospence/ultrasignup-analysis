# Base image for PyMC GPU analysis - CUDA 12.4 (compatible with RunPod driver 565)
# Build: docker buildx build --platform linux/amd64 -t justmytwospence/pymc-base:v4 --push .

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Environment setup for CUDA 12.4
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:/usr/local/cuda-12.4/extras/CUPTI/lib64:${LD_LIBRARY_PATH}

# Install Python 3.12 from deadsnakes PPA + core system dependencies
RUN rm -rf /var/lib/apt/lists/* && \
    apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    git \
    graphviz \
    libgraphviz-dev \
    curl \
    rsync \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Make Python 3.12 the default and install pip via ensurepip (avoids distutils issues)
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    python -m ensurepip --upgrade && \
    python -m pip install --upgrade pip

# Configure SSH server (key-only authentication)
RUN mkdir /var/run/sshd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config && \
    sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config

# Create SSH directory for authorized keys
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh

# Install JAX with CUDA 12 support
RUN pip install "jax[cuda12]>=0.4.30"

# Install PyMC stack with dependencies
RUN pip install \
    "pymc>=5.6.1" \
    "numpyro>=0.19.0" \
    "pymc-extras>=0.5.0" \
    "blackjax>=1.3" \
    "arviz>=0.22.0"

# Install base Python dependencies
RUN pip install \
    "duckdb>=1.0.0" \
    "pandas>=2.0.0" \
    "numpy>=1.24.0" \
    "matplotlib>=3.8.0" \
    "seaborn>=0.13.2" \
    "plotly>=5.18.0" \
    "igraph>=0.11.0" \
    "scipy>=1.11.0" \
    "graphviz>=0.20.3" \
    "ipywidgets>=8.1.7" \
    "jupyterlab>=4.0.0" \
    "ipykernel>=6.0.0" \
    "networkx>=3.0" \
    "qrcode[pil]>=8.2" \
    "requests>=2.31.0" \
    "tqdm>=4.66.0" \
    "rich" \
    "nvitop" \
    "psutil" \
    "gpustat"

# Register Python kernel for Jupyter 
RUN python -m ipykernel install --name=python3 --display-name="Python 3.12 (PyMC CUDA)"

# Create base working directories
RUN mkdir -p /workspace/analysis /workspace/data

# Install development tools (last layer - changes frequently)
RUN rm -rf /var/lib/apt/lists/* && apt-get update && apt-get install -y \
    tmux \
    htop \
    vim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/analysis

# Copy and set up entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose ports (inherited by derived images)
EXPOSE 22 8888

ENTRYPOINT ["/entrypoint.sh"]