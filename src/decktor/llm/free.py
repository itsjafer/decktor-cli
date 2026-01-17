import time
import g4f
from typing import Tuple, Dict, Any
from .base import LLMProvider

class FreeProvider(LLMProvider):
    def __init__(self, model_name: str = "gpt-4o-mini"):
        # Map 'free' to a reasonable default if passed specifically
        if model_name.lower() == "free":
            self.model_alias = "gpt-4o-mini"
        else:
            self.model_alias = model_name
            
        self.client = g4f.client.Client()

    def generate(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_alias,
                messages=[{"role": "user", "content": prompt}],
            )
            
            content = response.choices[0].message.content
            
            metrics = {
                "input_tokens": 0, # Not reliably available
                "output_tokens": 0, 
                "generation_time": time.time() - start_time
            }
            return content, metrics
        except Exception as e:
            return f"Error: {e}", {}
