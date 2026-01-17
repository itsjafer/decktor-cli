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
    "GPT4Free (Free)": {
        "id": "free",
        "description": "Uses g4f to access free LLM providers (no key required).",
        "type": "free",
    },
    # OpenAI Models
    "GPT-5 Mini": {
        "id": "gpt-5-mini",
        "description": "Fast, lightweight OpenAI model for real-time applications.",
        "type": "openai",
    },
    "GPT-5": {
        "id": "gpt-5",
        "description": "OpenAI's main developer model.",
        "type": "openai",
    },
    # Anthropic Models
    "Claude Haiku 4.5": {
        "id": "claude-haiku-4-5-latest",
        "description": "Anthropic's fastest and most cost-efficient model.",
        "type": "anthropic",
    },
    "Claude Sonnet 4.5": {
        "id": "claude-sonnet-4-5-latest",
        "description": "Balanced Anthropic model for performance and cost.",
        "type": "anthropic",
    },
}


def get_supported_model_names():
    """Get a list of supported LLM model names."""
    return list(SUPPORTED_MODELS.keys())


def get_model_id(model_name: str) -> str:
    """Get the model ID for a supported LLM model."""
    return SUPPORTED_MODELS.get(model_name, {}).get("id", "")
