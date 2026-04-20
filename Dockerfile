FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py .

# Pre-download model (optional, speeds up cold start if network volume is used)
# RUN python3 -c "from diffusers import QwenImagePipeline; QwenImagePipeline.from_pretrained('Qwen/Qwen-image', torch_dtype=torch.bfloat16)"

# Start RunPod serverless
CMD ["python3", "-u", "handler.py"]
