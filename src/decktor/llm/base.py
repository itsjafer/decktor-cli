from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """
        Generates content based on the prompt.
        
        Args:
            prompt (str): The input prompt for the LLM.
            
        Returns:
            Tuple[str, Dict[str, Any]]: A tuple containing the generated text content
                                        and a dictionary of usage metrics (e.g., token counts, time).
        """
        pass
