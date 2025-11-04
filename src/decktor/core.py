"""Core LLM interaction logic for Decktor."""

import io
import os
import shutil
import tempfile
import time
import zipfile

import torch
import zstandard
from anki.collection import Collection
from bs4 import BeautifulSoup
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from decktor.models import get_model_id
from decktor.utils import make_prompt


def get_llm_model(model_name: str, quantize: bool = True):
    """Get the LLM model instance based on the model name.

    Args:
        model_name (str): The name of the LLM model.
        quantize (bool): Whether to use 4-bit quantization for the model.

    Returns:
        An instance of the specified LLM model.
    """
    print(f"Loading model: {model_name} with quantization={quantize}")

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
    prompt_template: str,
    max_new_tokens: int = 8192,
) -> tuple[str, dict]:
    """Improve an Anki card using the specified LLM model and prompt template.

    Args:
        card (str): The original Anki card content.
        model: The LLM model to use for improvement.
        tokenizer: The tokenizer for the model.
        prompt_template (str): The prompt template to guide the LLM.
        max_new_tokens (int): Maximum number of new tokens to generate.

    Returns:
        tuple[str, dict]: The improved Anki card content and performance metrics.
    """
    # Track timing
    start_time = time.time()

    # prepare the model input
    prompt = make_prompt(card, prompt_template)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Track tokenization time
    tokenization_time = time.time() - start_time
    input_token_count = model_inputs.input_ids.shape[1]

    # conduct text completion
    generation_start = time.time()
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    generation_time = time.time() - generation_start

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    output_token_count = len(output_ids)

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    # Calculate performance metrics
    total_time = time.time() - start_time
    tokens_per_second = output_token_count / generation_time if generation_time > 0 else 0

    metrics = {
        "total_time": total_time,
        "tokenization_time": tokenization_time,
        "generation_time": generation_time,
        "input_tokens": input_token_count,
        "output_tokens": output_token_count,
        "tokens_per_second": tokens_per_second,
    }

    return content, metrics


def _extract_db_from_apkg(apkg_stream: io.BytesIO, temp_dir: str) -> str:
    """Extracts the Anki database from the .apkg stream into a temporary directory.

    Args:
        apkg_stream (io.BytesIO): The .apkg file stream.
        temp_dir (str): The temporary directory to extract files into.

    Returns:
        str: The path to the extracted Anki database file.
    """
    db_path = ""

    with zipfile.ZipFile(apkg_stream, "r") as z:
        db_filename = next(
            (name for name in z.namelist() if name.startswith("collection.anki2")), None
        )
        if not db_filename:
            raise FileNotFoundError("Could not find 'collection.anki2' in the .apkg stream.")

        compressed_db_path = os.path.join(temp_dir, db_filename)

        print(f"Extracting database: {compressed_db_path}")
        z.extract(db_filename, temp_dir)

        # Handle .anki21b compressed databases
        if db_filename.endswith(".anki21b"):
            # We need to decompress it
            db_path_to_load = compressed_db_path.replace(".anki21b", ".anki21")
            dctx = zstandard.ZstdDecompressor()
            with open(compressed_db_path, "rb") as ifh, open(db_path_to_load, "wb") as ofh:
                dctx.copy_stream(ifh, ofh)
            db_path = db_path_to_load
        else:
            db_path = compressed_db_path

    return db_path


def read_apkg_cards(apkg_stream: io.BytesIO):
    """Loads an .apkg file and reads the front/back of its cards."""

    # Create a temporary directory to extract the .apkg
    temp_dir = tempfile.mkdtemp()
    db_path = ""

    db_path = _extract_db_from_apkg(apkg_stream, temp_dir)

    col = Collection(db_path)

    # col.find_cards("") returns a list of all card IDs (cids)
    card_ids = col.find_cards("")
    print(f"\nFound {len(card_ids)} total cards.")

    cards = []

    for cid in card_ids:
        card = col.get_card(cid)

        # Get the note (the data) from the card
        note = card.note()

        field_names = [f["name"] for f in note.note_type()["flds"]]

        field_data = dict(zip(field_names, note.fields))

        front_text_html = field_data.get("Front", "")
        front_text = BeautifulSoup(front_text_html, "html.parser").get_text().strip()
        back_text_html = field_data.get("Back", "")
        back_text = BeautifulSoup(back_text_html, "html.parser").get_text().strip()

        cards.append({"front": front_text, "back": back_text})

    col.close()

    # Clean up the temporary directory
    if os.path.exists(temp_dir):
        print(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)

    return cards
