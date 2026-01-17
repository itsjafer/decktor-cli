import os
import time
import google.generativeai as genai
from typing import Tuple, Dict, Any
from .base import LLMProvider
from decktor.models import SUPPORTED_MODELS

class GeminiProvider(LLMProvider):
    def __init__(self, model_name: str):
        self.model_name = model_name
        model_info = SUPPORTED_MODELS.get(model_name, {})
        # If model_name is not in SUPPORTED_MODELS, assume it's a direct ID or fallback
        self.model_id = model_info.get("id", model_name)
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in a .env file.")
        
        genai.configure(api_key=api_key)
        
        # Configure generation config
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
        }
        
        self.model = genai.GenerativeModel(
            model_name=self.model_id,
            generation_config=generation_config,
        )

    def generate(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        start_time = time.time()
        try:
            response = self.model.generate_content(prompt)
            content = response.text
            
            input_tokens = 0
            output_tokens = 0
            if response.usage_metadata:
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count
            
            metrics = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "generation_time": time.time() - start_time
            }
            return content, metrics
            
        except Exception as e:
            # We explicitly return the error string to be handled by the caller,
            # mirroring original behavior where exceptions printed and returned error msg
            return str(e), {}
