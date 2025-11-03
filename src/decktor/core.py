"""Core LLM interaction logic for Decktor."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from decktor.models import get_model_id
from decktor.utils import get_prompt


def get_llm_model(model_name: str, quantize: bool = True):
    """Get the LLM model instance based on the model name.

    Args:
        model_name (str): The name of the LLM model.
        quantize (bool): Whether to use 4-bit quantization for the model.

    Returns:
        An instance of the specified LLM model.
    """
    # load the tokenizer and the model
    model_id = get_model_id(model_name)

    # Configure 4-bit quantization
    if quantize:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,  # Use bfloat16 for computation
        )
    else:
        quantization_config = None

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map="auto",
    )

    return model, tokenizer


def improve_card(
    card: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_path: str,
    max_new_tokens: int = 8192,
) -> str:
    """Improve an Anki card using the specified LLM model and prompt template.

    Args:
        card (str): The original Anki card content.
        llm_model (str): The LLM model to use for improvement.
        prompt_template (str): The prompt template to guide the LLM.
        max_new_tokens (int): Maximum number of new tokens to generate.

    Returns:
        str: The improved Anki card content.
    """
    # prepare the model input
    prompt = get_prompt(card, prompt_path)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    return content
