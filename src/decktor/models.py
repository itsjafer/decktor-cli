SUPPORTED_MODELS = {
    "Gemini 2.5 Flash Lite": {
        "id": "gemini-2.5-flash-lite",
        "description": "Fast and lightweight API model.",
        "type": "api",
    },
    "Gemini 2.5 Flash": {
        "id": "gemini-2.5-flash",
        "description": "Balanced API model.",
        "type": "api",
    },
    "Gemini 3.0 Flash Preview": {
        "id": "gemini-3-flash-preview",
        "description": "Next-gen API model.",
        "type": "api",
        "thinking": False,
    },
}


def get_supported_model_names():
    """Get a list of supported LLM model names."""
    return list(SUPPORTED_MODELS.keys())


def get_model_id(model_name: str) -> str:
    """Get the model ID for a supported LLM model."""
    return SUPPORTED_MODELS.get(model_name, {}).get("id", "")
