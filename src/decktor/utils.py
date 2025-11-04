"""Utils for DeckTor."""

import base64
from pathlib import Path

import torch


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def get_prompt_template(prompt_path: str) -> str:
    """Get the prompt template for improving Anki cards."""
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()
    return prompt_template


def make_prompt(card: str, prompt_template: str) -> str:
    """Get the prompt for improving an Anki card."""
    return prompt_template.replace("{card}", card)


def get_gpu_info() -> dict:
    """Get GPU information and utilization statistics.

    Returns:
        dict: GPU information including memory usage, utilization, and recommendations.
    """
    if not torch.cuda.is_available():
        return {
            "available": False,
            "message": "No CUDA GPU available. Running on CPU will be very slow.",
        }

    info = {"available": True, "device_count": torch.cuda.device_count(), "devices": []}

    for i in range(torch.cuda.device_count()):
        device_props = torch.cuda.get_device_properties(i)
        memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
        total_memory = device_props.total_memory / 1024**3  # GB

        device_info = {
            "id": i,
            "name": device_props.name,
            "total_memory_gb": round(total_memory, 2),
            "allocated_memory_gb": round(memory_allocated, 2),
            "reserved_memory_gb": round(memory_reserved, 2),
            "free_memory_gb": round(total_memory - memory_reserved, 2),
            "compute_capability": f"{device_props.major}.{device_props.minor}",
        }
        info["devices"].append(device_info)

    return info
