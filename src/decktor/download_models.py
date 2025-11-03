"""Download and cache LLM models for Hugging Face."""

import argparse
import os

from rich import print
from rich.panel import Panel
from transformers import AutoModelForCausalLM, AutoTokenizer

from decktor.models import SUPPORTED_MODELS


def main():
    """Download and cache LLM models for Decktor."""
    parser = argparse.ArgumentParser(description="Download and cache LLM models for Decktor.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(SUPPORTED_MODELS.keys()),
        help="List of model names to download.",
    )
    args = parser.parse_args()

    print(Panel("Starting model download... This might take a while. Patientia fortium est."))
    for model_name in args.models:
        model_id = SUPPORTED_MODELS.get(model_name, {}).get("id", "")
        if not model_id:
            print(f"Model '{model_name}' is not supported. Skipping.")
            continue
        print(f"Downloading model: [bold magenta]{model_name} ({model_id})[/bold magenta]")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto"
        )
        print(f"[bold green]Model '{model_name}' downloaded and cached successfully.[/bold green]")


if __name__ == "__main__":
    main()
