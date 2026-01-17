import os
import time
from openai import OpenAI
from typing import Tuple, Dict, Any
from .base import LLMProvider
from decktor.models import SUPPORTED_MODELS


class OpenAIProvider(LLMProvider):
    def __init__(self, model_name: str):
        self.model_name = model_name
        model_info = SUPPORTED_MODELS.get(model_name, {})
        self.model_id = model_info.get("id", model_name)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. Please set it in a .env file."
            )

        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.7,
                max_tokens=8192,
            )

            content = response.choices[0].message.content

            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0

            metrics = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "generation_time": time.time() - start_time,
            }
            return content, metrics

        except Exception as e:
            return str(e), {}
