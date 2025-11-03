"""Core LLM interaction logic for Decktor."""

import io
import os
import shutil
import tempfile
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
    prompt = make_prompt(card, prompt_template)
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


def read_apkg_cards(apkg_stream: io.BytesIO):
    """Loads an .apkg file and reads the front/back of its cards."""

    # Create a temporary directory to extract the .apkg
    temp_dir = tempfile.mkdtemp()
    db_path = ""

    try:
        with zipfile.ZipFile(apkg_stream, "r") as z:
            db_filename = next(
                (name for name in z.namelist() if name.startswith("collection.anki2")), None
            )
            if not db_filename:
                raise FileNotFoundError("Could not find 'collection.anki2[1]' in the .apkg stream.")

            compressed_db_path = os.path.join(temp_dir, db_filename)

            print(f"Extracting database: {compressed_db_path}")
            z.extract(db_filename, temp_dir)

            if db_filename.endswith(".anki21b"):
                # We need to decompress it
                db_path_to_load = compressed_db_path.replace(".anki21b", ".anki21")
                dctx = zstandard.ZstdDecompressor()
                with open(compressed_db_path, "rb") as ifh, open(db_path_to_load, "wb") as ofh:
                    dctx.copy_stream(ifh, ofh)
            db_path = os.path.join(temp_dir, db_path_to_load)

        col = Collection(db_path)

        # col.find_cards("") returns a list of all card IDs (cids)
        card_ids = col.find_cards("")
        print(f"\nFound {len(card_ids)} total cards.")

        for i, cid in enumerate(card_ids[:10]):
            card = col.get_card(cid)

            # Get the note (the data) from the card
            note = card.note()

            # Get the field names from the note's template (model)
            # This creates a list like ['Front', 'Back', 'Add. Info']
            field_names = [f["name"] for f in note.note_type()["flds"]]

            # note.fields is the list of data, e.g. ['Define cov matrix', '...matrix...']
            # We zip them together to create a clean dictionary
            field_data = dict(zip(field_names, note.fields))

            # Now you can access the data by name, safely
            # We use .get() to avoid errors if a field is missing
            front_text_html = field_data.get("Front", "")
            front_text = BeautifulSoup(front_text_html, "html.parser").get_text().strip()
            back_text_html = field_data.get("Back", "")
            back_text = BeautifulSoup(back_text_html, "html.parser").get_text().strip()

            print(f"\n--- Card {i + 1} (cid: {cid}) ---")
            print(f"[Field 'Front']: {front_text}")
            print(f"[Field 'Back']:  {back_text}")

        col.close()
    except Exception as e:
        print(f"Error reading .apkg file: {e}")
    # finally:
    #     # Clean up the temporary directory
    #     if os.path.exists(temp_dir):
    #         print(f"Cleaning up temporary directory: {temp_dir}")
    #         shutil.rmtree(temp_dir)
