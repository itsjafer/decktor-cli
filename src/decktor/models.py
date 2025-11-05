SUPPORTED_MODELS = {
    "Qwen3-4B (Non-Thinking only)": {
        "id": "Qwen/Qwen3-4B-Instruct-2507",
        "description": "Low memory required, fast inference.",
    },
    "Qwen3-4B (Thinking only)": {
        "id": "Qwen/Qwen3-4B-Thinking-2507",
        "description": "Low memory required, slow inference.",
    },
    "Qwen3-32B (With/Without Thinking)": {
        "id": "Qwen/Qwen3-32B",
        "description": "High memory required, powerful, fast in non-thinking mode, slower in thinking mode.",
    },
}


def get_supported_model_names():
    """Get a list of supported LLM model names."""
    return list(SUPPORTED_MODELS.keys())


def get_model_id(model_name: str) -> str:
    """Get the model ID for a supported LLM model."""
    return SUPPORTED_MODELS.get(model_name, {}).get("id", "")
