"""RunPod Serverless Handler for Qwen Image (Text-to-Image).

Input format:
    {
        "input": {
            "prompt": "a cute cat",
            "negative_prompt": "",
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 50,
            "true_cfg_scale": 4.0,
            "seed": null
        }
    }

Output format:
    {
        "image": "base64_encoded_png",
        "seed": 12345
    }
"""

import base64
import io
import os
import torch
from PIL import Image
from diffusers import QwenImagePipeline

# ── Config ──────────────────────────────────────────────────────────
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen-image")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# Global pipeline (loaded once, reused across requests)
_pipe = None


def load_pipeline():
    """Load Qwen Image pipeline once and cache it."""
    global _pipe
    if _pipe is None:
        print(f"[Handler] Loading Qwen Image pipeline from {MODEL_ID}...")
        _pipe = QwenImagePipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
        )
        _pipe = _pipe.to(DEVICE)
        print("[Handler] Pipeline loaded.")
    return _pipe


def handler(job):
    """RunPod serverless handler."""
    job_input = job.get("input", {})

    prompt = job_input.get("prompt", "")
    negative_prompt = job_input.get("negative_prompt", "")
    width = int(job_input.get("width", 1024))
    height = int(job_input.get("height", 1024))
    num_inference_steps = int(job_input.get("num_inference_steps", 50))
    true_cfg_scale = float(job_input.get("true_cfg_scale", 4.0))
    seed = job_input.get("seed")

    if not prompt:
        return {"error": "No prompt provided."}

    # Load pipeline
    pipe = load_pipeline()

    # Set seed if provided
    generator = None
    if seed is not None:
        generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

    # Generate image
    print(f"[Handler] Generating: {prompt[:80]}...")
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=true_cfg_scale,
        generator=generator,
    )

    image = result.images[0]

    # Convert to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        "image": img_b64,
        "seed": seed,
    }


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
