import os
import time
import anthropic
from typing import Tuple, Dict, Any
from .base import LLMProvider
from decktor.models import SUPPORTED_MODELS


class AnthropicProvider(LLMProvider):
    def __init__(self, model_name: str):
        self.model_name = model_name
        model_info = SUPPORTED_MODELS.get(model_name, {})
        self.model_id = model_info.get("id", model_name)

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found in environment variables. Please set it in a .env file."
            )

        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        start_time = time.time()
        try:
            response = self.client.messages.create(
                model=self.model_id,
                max_tokens=8192,
                system="You are a helpful assistant. Always respond with valid JSON only, no other text.",
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.content[0].text

            input_tokens = response.usage.input_tokens if response.usage else 0
            output_tokens = response.usage.output_tokens if response.usage else 0

            metrics = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "generation_time": time.time() - start_time,
            }
            return content, metrics

        except Exception as e:
            return str(e), {}
