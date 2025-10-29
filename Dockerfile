FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /app
COPY requirements.txt .

# install packages
RUN apt-get update && apt-get install -y \
    git \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip uv
RUN uv pip install --system trl[liger,peft,vlm] hf_transfer trackio
RUN uv pip install --system https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
RUN pip install -r requirements.txt

# Configure SSH for GitLab access
RUN mkdir -p /root/.ssh && \
    chmod 700 /root/.ssh

# Set Environment Variables
ENV WANDB_MODE="online"
ENV HF_CACHE="/app/hf_cache"

# Create directories for cache
RUN mkdir -p ${HF_CACHE}
