"""Utils for DeckTor."""
from pathlib import Path
import base64


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
