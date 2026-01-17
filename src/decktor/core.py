"""Core LLM interaction logic for Decktor."""

import io
import json
import os
import shutil
import tempfile
import time
import zipfile

import google.generativeai as genai
from anki.collection import Collection
from anki.decks import DeckManager
from anki.models import ModelManager
from bs4 import BeautifulSoup

import re
from decktor.models import SUPPORTED_MODELS, get_model_id
from decktor.utils import make_prompt
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


def get_llm_model(model_name: str):
    """Get the LLM model instance based on the model name.

    Args:
        model_name (str): The name of the LLM model.

    Returns:
        An instance of the specified LLM model.
    """
    print(f"Loading model: {model_name}")

    # load the tokenizer and the model
    model_info = SUPPORTED_MODELS.get(model_name, {})
    model_id = model_info.get("id")

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
    
    model = genai.GenerativeModel(
        model_name=model_id,
        generation_config=generation_config,
    )
    return model


def improve_card(
    card: str,
    model: genai.GenerativeModel,
    prompt_template: str,
) -> tuple[str, dict]:
    """Improve an Anki card using the specified LLM model and prompt template.

    Args:
        card (str): The original Anki card content.
        model: The LLM model to use for improvement.
        prompt_template (str): The prompt template to guide the LLM.

    Returns:
        tuple[str, dict]: The improved Anki card content and performance metrics.
    """
    # Track timing
    start_time = time.time()
    
    # prepare the model input
    prompt = make_prompt(card, prompt_template)
    
    # API execution path
    try:
        response = model.generate_content(prompt)
        content = response.text
        
        # Simple metrics for API (token counts might need usage metadata access if available)
        # Gemini response usually has usage_metadata
        input_token_count = 0
        output_token_count = 0
        if response.usage_metadata:
            input_token_count = response.usage_metadata.prompt_token_count
            output_token_count = response.usage_metadata.candidates_token_count
        
        generation_time = time.time() - start_time # Approximate
        tokenization_time = 0
        
    except Exception as e:
        # Handle API errors
        print(f"API Error: {e}")
        return str(e), {}

    # Calculate performance metrics
    total_time = time.time() - start_time
    tokens_per_second = output_token_count / generation_time if generation_time > 0 else 0

    metrics = {
        "total_time": total_time,
        "tokenization_time": tokenization_time,
        "generation_time": generation_time,
        "total_input_tokens_unpadded": input_token_count,
        "total_output_tokens": output_token_count,
        "avg_total_time_per_card": total_time,
        "avg_gen_time_per_card": generation_time,
        "throughput_cards_per_second": 1 / total_time,
        "throughput_tokens_per_second": tokens_per_second,
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
        names = z.namelist()
        if "collection.anki21b" in names:
            db_filename = "collection.anki21b"
        elif "collection.anki2" in names:
            db_filename = "collection.anki2"
        else:
            # Fallback for nested folders or other variations
            db_filename = next(
                (name for name in names if name.endswith("collection.anki21b")), 
                next((name for name in names if name.endswith("collection.anki2")), None)
            )

        if not db_filename:
            raise FileNotFoundError("Could not find 'collection.anki2' or 'collection.anki21b' in the .apkg stream.")

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


def create_apkg(
    processed_cards: list[dict], deck_name: str = "DeckTor Improved Deck"
) -> io.BytesIO:
    """Creates a new .apkg file from a list of processed cards.

    Args:
        processed_cards: The list of card dicts from st.session_state.
        deck_name: The name for the new deck inside the .apkg.

    Returns:
        io.BytesIO: The in-memory .apkg (zip) file.
    """
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "collection.anki2")
    zip_path = os.path.join(temp_dir, "deck.apkg")

    col = Collection(db_path)

    deck_manager = DeckManager(col)
    deck_id = deck_manager.id(deck_name, create=True)

    model_manager = ModelManager(col)
    basic_model = model_manager.by_name("Basic")

    for card_data in processed_cards:
        status = card_data.get("status", "pending")

        content = card_data["original"]
        if status == "accepted":
            content = card_data["improved"]

        front = content.get("front", "")
        back = content.get("back", "")

        note = col.new_note(basic_model)
        note["Front"] = front
        note["Back"] = back

        col.add_note(note, deck_id)

    col.save()
    col.close()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(db_path, arcname="collection.anki2")

    with open(zip_path, "rb") as f:
        zip_bytes_io = io.BytesIO(f.read())

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    return zip_bytes_io


def process_apkg_with_resume(
    input_apkg: str,
    output_apkg: str,
    model_name: str,
    prompt_path: str,
    working_dir: str,
    batch_size: int = 10,
    preview: bool = False,
    limit: int | None = None,
):
    """
    Process an .apkg file using an LLM, with resume capability.

    Args:
        input_apkg: Path to the input .apkg file.
        output_apkg: Path to the output .apkg file.
        model_name: Name of the LLM model to use.
        prompt_path: Path to the prompt template file.
        working_dir: Directory to store intermediate files.
        working_dir: Directory to store intermediate files.
        batch_size: Number of cards to process before saving.
        preview: If True, do not save changes to disk (Dry Run).
        limit: Maximum number of cards to process.
    """
    console = Console()

    if preview:
        console.print(
            Panel.fit(
                "[bold yellow]PREVIEW MODE ENABLED[/bold yellow]\n"
                "Changes will NOT be saved to the database or output file.\n"
                "Showing pretty diffs of changes.",
                title="⚠️  DRY RUN  ⚠️",
                border_style="yellow",
            )
        )

    # 1. Setup Phase
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
        console.print(f"[bold green]Created working directory:[/bold green] {working_dir}")
        # Unzip everything
        console.print(f"Extracting [bold]{input_apkg}[/bold]...")
        with zipfile.ZipFile(input_apkg, "r") as zf:
            zf.extractall(working_dir)
    else:
        # Check if it looks like a valid unzipped deck
        if not any(os.path.exists(os.path.join(working_dir, f)) for f in ["collection.anki2", "collection.anki21", "collection.anki21b"]):
            console.print(
                f"[bold red]Error:[/bold red] Working directory {working_dir} exists but does not contain a valid collection file. "
                "Please use a different working directory or clear it."
            )
            return

        console.print(
            f"[bold yellow]Resuming[/bold yellow] from existing working directory: {working_dir}"
        )

    # 2. Database Setup
    # 2. Database Setup
    anki21b_path = os.path.join(working_dir, "collection.anki21b")
    anki2_path = os.path.join(working_dir, "collection.anki2")
    
    db_path = anki2_path

    if os.path.exists(anki21b_path):
        console.print(f"[bold green]Found compressed database:[/bold green] {anki21b_path}")
        # Decompress to collection.anki21
        db_path = os.path.join(working_dir, "collection.anki21")
        if not os.path.exists(db_path): # verify if we need to decompress
            console.print("Decompressing database...")
            dctx = zstandard.ZstdDecompressor()
            with open(anki21b_path, "rb") as ifh, open(db_path, "wb") as ofh:
                dctx.copy_stream(ifh, ofh)
    elif os.path.exists(anki2_path):
        db_path = anki2_path
    else:
        console.print(f"[bold red]Error:[/bold red] No valid collection file found in {working_dir}.")
        return

    col = Collection(db_path)
    console.print(f"Opened collection at {db_path}")

    # 3. Model Loading
    console.print(f"Loading model [bold]{model_name}[/bold]...")
    model = get_llm_model(model_name)
    
    with open(prompt_path, "r") as f:
        prompt_template = f.read()

    # 4. Processing Loop
    # 4. Processing Loop
    processed_tag = "decktor-processed"
    
    # Find all notes
    # We want to iterate notes, not cards, to avoid processing the same content twice if multiple cards share a note.
    all_nids = col.find_notes("")
    total_notes = len(all_nids)
    
    # Filter for unprocessed notes
    # We can check if the note has the tag
    unprocessed_nids = []
    for nid in all_nids:
        note = col.get_note(nid)
        if not note.has_tag(processed_tag):
            unprocessed_nids.append(nid)
            
    notes_to_process_count = len(unprocessed_nids)
    console.print(f"Found {total_notes} total notes. {notes_to_process_count} to process.")

    processed_count = 0
    
    # Apply limit
    if limit is not None:
        console.print(f"[yellow]Limiting processing to {limit} cards.[/yellow]")
        notes_to_process_count = min(notes_to_process_count, limit)
        unprocessed_nids = unprocessed_nids[:limit]

    # Batching logic
    batches = [unprocessed_nids[i:i + batch_size] for i in range(0, len(unprocessed_nids), batch_size)]

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Processing notes...", total=notes_to_process_count)

        for batch_nids in batches:
            batch_payload = []
            batch_notes = {} # Map nid -> note object

            for nid in batch_nids:
                try:
                    note = col.get_note(nid)
                    batch_notes[nid] = note
                    
                    # Heuristic: try to find "Front" and "Back" fields
                    field_names = [f['name'] for f in note.note_type()['flds']]
                    fields = dict(zip(field_names, note.fields))
                    
                    front_text = fields.get("Front", "")
                    back_text = fields.get("Back", "")
                    
                    # If fields are not named Front/Back, fallback to first/second
                    if not front_text and not back_text:
                         if len(note.fields) > 0:
                             front_text = note.fields[0]
                         if len(note.fields) > 1:
                             back_text = note.fields[1]

                    # Clean HTML for the LLM
                    soup_front = BeautifulSoup(front_text, "html.parser")
                    clean_front = soup_front.get_text(separator="\n").strip()
                    soup_back = BeautifulSoup(back_text, "html.parser")
                    clean_back = soup_back.get_text(separator="\n").strip()

                    batch_payload.append({
                        "id": nid,
                        "front": clean_front,
                        "back": clean_back
                    })
                except Exception as e:
                    console.print(f"[red]Failed to prepare note {nid} for batch: {e}[/red]")
            
            if not batch_payload:
                continue

            # Call LLM with batch
            max_retries = 3
            batch_success = False

            for attempt in range(max_retries):
                try:
                    cards_json = json.dumps(batch_payload, indent=2, ensure_ascii=False)
                    
                    # Construct the full prompt manually to handle {cards} vs {card}
                    if "{cards}" in prompt_template:
                        full_prompt = prompt_template.replace("{cards}", cards_json)
                    else:
                        # Fallback for old prompts
                        full_prompt = prompt_template.replace("{card}", cards_json)

                    # Use improve_card to handle generation, passing identity template
                    improved_text, _ = improve_card(
                        full_prompt, 
                        model, 
                        "{card}" 
                    )
                    
                    # Parse Batch Output
                    json_match = re.search(r"\{.*\}", improved_text, re.DOTALL)
                    if json_match:
                        try:
                            response_json = json.loads(json_match.group(0))
                            # Handle both {cards: [...]} and just [...] (unlikely if prompt is followed, but good for robustness)
                            if "cards" in response_json:
                                improved_cards = response_json["cards"]
                            elif isinstance(response_json, list):
                                improved_cards = response_json
                            else:
                                raise ValueError("Unexpected JSON format: missing 'cards' key or not a list")
                                
                            # Create a map for easy lookup
                            improved_map = {str(item["id"]): item for item in improved_cards}
                            
                            # Validate Batch
                            validation_error = False
                            for nid in batch_notes:
                                nid_str = str(nid)
                                if nid_str in improved_map:
                                    item = improved_map[nid_str]
                                    if item.get("changed", False):
                                        if not item.get("front") or not item.get("back"):
                                            console.print(f"[yellow]Validation failed for card {nid}: Empty front/back in response. Retrying...[/yellow]")
                                            validation_error = True
                                            break
                            
                            if validation_error:
                                continue # Retry loop

                            # If we got here, validation passed
                            for nid in batch_notes:
                                note = batch_notes[nid]
                                # nid is int, convert to str for lookup if JSON used strings
                                nid_str = str(nid)
                                
                                if nid_str in improved_map:
                                    item = improved_map[nid_str]
                                    if item.get("changed", False):
                                        new_front = item.get("front")
                                        if new_front is None:
                                            new_front = ""
                                        else:
                                            new_front = str(new_front)

                                        new_back = item.get("back")
                                        if new_back is None:
                                            new_back = ""
                                        else:
                                            new_back = str(new_back)
                                        
                                        # Update Note
                                        field_names = [f['name'] for f in note.note_type()['flds']]
                                        fields = dict(zip(field_names, note.fields))

                                        if "Front" in fields:
                                            note["Front"] = new_front
                                        else:
                                            note.fields[0] = new_front 
                                            
                                        if "Back" in fields:
                                            note["Back"] = new_back
                                        elif len(note.fields) > 1:
                                            note.fields[1] = new_back

                                # Always mark processed
                                if not preview:
                                    note.add_tag(processed_tag)
                                    col.update_note(note)
                                
                                processed_count += 1
                                progress.advance(task)
                                
                                # Preview logic
                                if (preview or limit is not None) and nid_str in improved_map:
                                    item = improved_map[nid_str]
                                    title_suffix = "(Dry Run)" if preview else "(Limit Applied)"
                                    table = Table(title=f"Card {nid} {title_suffix}", show_lines=True)
                                    table.add_column("Field", style="cyan", no_wrap=True)
                                    table.add_column("Original", style="magenta")
                                    table.add_column("Improved", style="green")

                                    # Show original vs new
                                    # Note: we didn't store original strictly, but we can grab field values or use batch_payload
                                    # For simplicity, just show what we have in item
                                    table.add_row("Front", "...", item.get("front", ""))
                                    table.add_row("Back", "...", item.get("back", ""))
                                    console.print(table)
                                    console.print("\n")

                            batch_success = True
                            break # Break retry loop on success

                        except (json.JSONDecodeError, ValueError) as e:
                             console.print(f"[yellow]Attempt {attempt+1}/{max_retries} failed: {e}. Retrying...[/yellow]")
                    else:
                        console.print(f"[yellow]Attempt {attempt+1}/{max_retries} failed: No JSON found. Retrying...[/yellow]")
                        # console.print(improved_text[:500])

                except Exception as e:
                    console.print(f"[red]Error processing batch attempt {attempt+1}: {e}[/red]")
            
            if not batch_success:
                 console.print(f"[red]Batch failed after {max_retries} attempts. Skipping improvement for these cards (keeping original).[/red]")
                 # We must still mark them as processed so we don't loop forever
                 if not preview:
                     for nid in batch_notes:
                         note = batch_notes[nid]
                         note.add_tag(processed_tag)
                         col.update_note(note)
                         progress.advance(task)

            # Save periodically
            if not preview:
                col.save()

    col.save()
    col.close()
    console.print(f"[bold green]Processing complete![/bold green] Processed {processed_count} notes.")

    # 4.1 Re-compress and Cleanup
    if not preview:
        # If we had a compressed database, we should re-compress it to ensure changes are saved in the format Anki expects (if it prefers compressed)
        # OR simply ensure we don't have conflicting files.
        # Anki 2.1.50+ prefers collection.anki21b.
        
        # If we have collection.anki21, let's compress it back to collection.anki21b
        anki21_path = os.path.join(working_dir, "collection.anki21")
        anki21b_path = os.path.join(working_dir, "collection.anki21b")
        anki2_path = os.path.join(working_dir, "collection.anki2")

        if os.path.exists(anki21_path):
            console.print("Re-compressing database to collection.anki21b...")
            cctx = zstandard.ZstdCompressor()
            with open(anki21_path, "rb") as ifh, open(anki21b_path, "wb") as ofh:
                cctx.copy_stream(ifh, ofh)
            
            # Remove the uncompressed file so it's not zipped
            os.remove(anki21_path)
            console.print("Removed uncompressed collection.anki21")

        # Also remove collection.anki2 if it exists and we have anki21b, to avoid ambiguity
        # (Unless we originally only had anki2, in which case we keep it. But we prefer the newer format if available)
        if os.path.exists(anki21b_path) and os.path.exists(anki2_path):
             os.remove(anki2_path)
             console.print("Removed legacy collection.anki2 to avoid conflicts.")

    # 5. Repackaging
    if preview:
        console.print("[bold yellow]Preview mode complete. No output file created.[/bold yellow]")
        col.close()
        return

    console.print(f"Creating output package [bold]{output_apkg}[/bold]...")
    with zipfile.ZipFile(output_apkg, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(working_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, working_dir)
                zf.write(file_path, arcname)

    console.print("[bold green]Done![/bold green]")
