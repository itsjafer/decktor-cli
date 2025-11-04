SUPPORTED_MODELS = {
    "Qwen3 32B": {
        "id": "Qwen/Qwen3-32B",
        "description": "Qwen3 32B base model. Slow but powerful.",
    },
    "Qwen3-30B-A3B-Thinking-2507": {
        "id": "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "description": "Qwen3 30B A3B Thinking 2507 variant. Faster but possibly less powerful.",
    },
    "Qwen3-4B-Thinking-2507": {
        "id": "Qwen/Qwen3-4B-Thinking-2507",
        "description": "Qwen3 4B Thinking 2507 variant. Very lightweight.",
    },
}


def get_supported_model_names():
    """Get a list of supported LLM model names."""
    return list(SUPPORTED_MODELS.keys())


def get_model_id(model_name: str) -> str:
    """Get the model ID for a supported LLM model."""
    return SUPPORTED_MODELS.get(model_name, {}).get("id", "")
