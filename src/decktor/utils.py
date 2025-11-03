"""Utils for DeckTor."""


def get_prompt_template(prompt_path: str) -> str:
    """Get the prompt template for improving Anki cards."""
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()
    return prompt_template


def get_prompt(card: str, prompt_path: str) -> str:
    """Get the prompt for improving an Anki card."""
    prompt_template = get_prompt_template(prompt_path)
    return prompt_template.replace("{card}", card)
