"""Core LLM interaction logic for Decktor."""

import json
import os
import shutil
import time
import zipfile
import re
from typing import Optional, List, Dict, Any, Tuple

import zstandard
from anki.collection import Collection
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table
from bs4 import BeautifulSoup

from decktor.models import SUPPORTED_MODELS
from decktor.utils import make_prompt
from decktor.llm import LLMProvider, GeminiProvider, FreeProvider

load_dotenv()


def get_llm_model(model_name: str) -> LLMProvider:
    """Get the LLM model instance based on the model name.

    Args:
        model_name (str): The name of the LLM model.

    Returns:
        LLMProvider: An instance of the specified LLM provider.
    """
    print(f"Loading model: {model_name}")

    model_info = SUPPORTED_MODELS.get(model_name, {})
    model_type = model_info.get("type", "api")

    # If explicitly free type or name implies free
    if model_type == "free" or model_name.lower() == "free":
        return FreeProvider(model_info.get("id", "gpt-4o-mini")) 
    
    # Default to Gemini
    return GeminiProvider(model_name)


def improve_card(
    card: str,
    model: LLMProvider,
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
    # prepare the model input
    prompt = make_prompt(card, prompt_template)
    
    # Provider execution path
    content, metrics = model.generate(prompt)

    # Calculate additional derived metrics if needed, or pass through
    # For now, we trust the provider's metrics + wrapping logic if we want total throughput
    # But the provider already returns basic metrics. We can augment them here if we want 
    # throughput relative to this function call, but the provider's timing is close enough.
    
    # Augment with throughput if time > 0
    if metrics.get("generation_time", 0) > 0:
        metrics["throughput_cards_per_second"] = 1 / metrics["generation_time"]
    
    return content, metrics


def _setup_workspace(input_apkg: str, working_dir: str, console: Console) -> bool:
    """Extracts the .apkg to the working directory."""
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
        # console.print(f"[bold green]Created working directory:[/bold green] {working_dir}") # optional noise
        console.print(f"Extracting [bold]{input_apkg}[/bold]...")
        with zipfile.ZipFile(input_apkg, "r") as zf:
            zf.extractall(working_dir)
        return True
    else:
        # Validate existing workspace
        if not any(os.path.exists(os.path.join(working_dir, f)) for f in ["collection.anki2", "collection.anki21", "collection.anki21b"]):
            console.print(
                f"[bold red]Error:[/bold red] Working directory {working_dir} exists but does not contain a valid collection file. "
                "Please use a different working directory or clear it."
            )
            return False
        console.print(
            f"[bold yellow]Resuming[/bold yellow] from existing working directory: {working_dir}"
        )
        return True


def _get_collection_path(working_dir: str, console: Console) -> Optional[str]:
    """Identifies and prepares the Anki database file."""
    anki21b_path = os.path.join(working_dir, "collection.anki21b")
    anki2_path = os.path.join(working_dir, "collection.anki2")
    
    if os.path.exists(anki21b_path):
        console.print(f"[bold green]Found compressed database:[/bold green] {anki21b_path}")
        # Decompress to collection.anki21
        db_path = os.path.join(working_dir, "collection.anki21")
        if not os.path.exists(db_path): # verify if we need to decompress
            console.print("Decompressing database...")
            dctx = zstandard.ZstdDecompressor()
            with open(anki21b_path, "rb") as ifh, open(db_path, "wb") as ofh:
                dctx.copy_stream(ifh, ofh)
        return db_path
    elif os.path.exists(anki2_path):
        return anki2_path
    
    console.print(f"[bold red]Error:[/bold red] No valid collection file found in {working_dir}.")
    return None


def _clean_field_html(html_content: str) -> str:
    """Strips HTML from field content for LLM consumption."""
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(separator="\n").strip()


def _prepare_batch_for_llm(batch_nids: List[int], col: Collection, exclude_fields: List[str], console: Console) -> Tuple[List[Dict[str, Any]], Dict[int, Any]]:
    """Prepares a batch of notes for the LLM."""
    batch_payload = []
    batch_notes = {}
    
    exclude_set = set(exclude_fields or [])

    for nid in batch_nids:
        try:
            note = col.get_note(nid)
            batch_notes[nid] = note
            
            field_names = [f['name'] for f in note.note_type()['flds']]
            fields = dict(zip(field_names, note.fields))
            
            cleaned_fields = {}
            for name, value in fields.items():
                if name in exclude_set:
                    continue
                    
                cleaned_val = _clean_field_html(value)
                cleaned_fields[name] = cleaned_val

            payload_item = {
                "id": nid,
                **cleaned_fields
            }
            batch_payload.append(payload_item)

        except Exception as e:
            console.print(f"[red]Failed to prepare note {nid} for batch: {e}[/red]")
            
    return batch_payload, batch_notes


def _update_note_fields(note, item: Dict[str, Any]) -> bool:
    """Updates note fields based on LLM response."""
    updated_any = False
    field_names = [f['name'] for f in note.note_type()['flds']]
    
    for field_name in field_names:
        if field_name in item and item[field_name] is not None:
            try:
                note[field_name] = str(item[field_name])
                updated_any = True
            except KeyError:
                pass
    return updated_any


def _display_preview(console: Console, nid: int, note, item: Dict[str, Any], preview_mode: bool):
    """Displays a preview table of changes."""
    title_suffix = "(Dry Run)" if preview_mode else "(Limit Applied)"
    table = Table(title=f"Card {nid} {title_suffix}", show_lines=True)
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Original", style="magenta", overflow="fold")
    table.add_column("Improved", style="green", overflow="fold")

    field_names = [f['name'] for f in note.note_type()['flds']]
    note_fields = dict(zip(field_names, note.fields))
    
    all_keys = sorted(list(set(field_names) | set(k for k in item.keys() if k not in ["id", "changed", "reason"])))
    
    for key in all_keys:
        original_val = note_fields.get(key, "(N/A)")
        if isinstance(original_val, str) and "<" in original_val:
                original_val_clean = _clean_field_html(original_val)
        else:
                original_val_clean = original_val

        new_val = item.get(key, "")
        if new_val is None:
            new_val = ""
        
        if key not in note_fields:
            table.add_row(f"{key} (New)", "", f"[bold yellow]{new_val}[/bold yellow]")
        elif str(new_val) != "" and str(new_val) != str(original_val):
            table.add_row(key, str(original_val_clean), f"[bold green]{new_val}[/bold green]")
        else:
            table.add_row(key, str(original_val_clean), str(new_val))
    
    console.print(table)
    if "reason" in item:
        console.print(f"[italic]Reason: {item['reason']}[/italic]")
    console.print("\n")


def _repackage_apkg(working_dir: str, output_apkg: str, console: Console, preview: bool):
    """Repackages the working directory into an .apkg file."""
    if preview:
        console.print("[bold yellow]Preview mode complete. No output file created.[/bold yellow]")
        return

    # Handle database re-compression if needed
    anki21_path = os.path.join(working_dir, "collection.anki21")
    anki21b_path = os.path.join(working_dir, "collection.anki21b")
    anki2_path = os.path.join(working_dir, "collection.anki2")

    if os.path.exists(anki21_path):
        console.print("Re-compressing database to collection.anki21b...")
        cctx = zstandard.ZstdCompressor()
        with open(anki21_path, "rb") as ifh, open(anki21b_path, "wb") as ofh:
            cctx.copy_stream(ifh, ofh)
        os.remove(anki21_path)
        console.print("Removed uncompressed collection.anki21")

    if os.path.exists(anki21b_path) and os.path.exists(anki2_path):
        os.remove(anki2_path)
        console.print("Removed legacy collection.anki2 to avoid conflicts.")

    console.print(f"Creating output package [bold]{output_apkg}[/bold]...")
    with zipfile.ZipFile(output_apkg, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(working_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, working_dir)
                zf.write(file_path, arcname)


def process_apkg_with_resume(
    input_apkg: str,
    output_apkg: str,
    model_name: str,
    prompt_path: str,
    working_dir: str,
    batch_size: int = 10,
    preview: bool = False,
    limit: Optional[int] = None,
    exclude_fields: list[str] = None,
    reprocess_all: bool = False,
):
    """
    Process an .apkg file using an LLM, with resume capability.
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
    if not _setup_workspace(input_apkg, working_dir, console):
        return

    # 2. Database Setup
    db_path = _get_collection_path(working_dir, console)
    if not db_path:
        return

    col = Collection(db_path)
    console.print(f"Opened collection at {db_path}")

    # 3. Model Loading
    console.print(f"Loading model [bold]{model_name}[/bold]...")
    model = get_llm_model(model_name)
    
    with open(prompt_path, "r") as f:
        prompt_template = f.read()

    # 4. Processing Loop
    processed_tag = "decktor-processed"
    
    # Find all notes
    all_nids = col.find_notes("")
    total_notes = len(all_nids)
    
    unprocessed_nids = []
    for nid in all_nids:
        note = col.get_note(nid)
        if reprocess_all or not note.has_tag(processed_tag):
            unprocessed_nids.append(nid)
            
    notes_to_process_count = len(unprocessed_nids)
    console.print(f"Found {total_notes} total notes. {notes_to_process_count} to process.")

    processed_count = 0
    
    if limit is not None:
        console.print(f"[yellow]Limiting processing to {limit} cards.[/yellow]")
        notes_to_process_count = min(notes_to_process_count, limit)
        unprocessed_nids = unprocessed_nids[:limit]

    batches = [unprocessed_nids[i:i + batch_size] for i in range(0, len(unprocessed_nids), batch_size)]

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Processing notes...", total=notes_to_process_count)

        for batch_nids in batches:
            batch_payload, batch_notes = _prepare_batch_for_llm(batch_nids, col, exclude_fields, console)
            
            if not batch_payload:
                continue

            max_retries = 3
            batch_success = False

            for attempt in range(max_retries):
                try:
                    cards_json = json.dumps(batch_payload, indent=2, ensure_ascii=False)
                    
                    if "{cards}" in prompt_template:
                        full_prompt = prompt_template.replace("{cards}", cards_json)
                    else:
                        full_prompt = prompt_template.replace("{card}", cards_json)

                    improved_text, _ = improve_card(full_prompt, model, "{card}")
                    
                    json_match = re.search(r"\{.*\}", improved_text, re.DOTALL)
                    if json_match:
                        try:
                            response_json = json.loads(json_match.group(0))
                            if "cards" in response_json:
                                improved_cards = response_json["cards"]
                            elif isinstance(response_json, list):
                                improved_cards = response_json
                            elif isinstance(response_json, dict) and "cards" not in response_json:
                                # Edge case: single card object but wrapper expects list logic?
                                # Assume response_json is the map if it smells like one? No, unsafe.
                                # Let's assume the LLM followed instructions to return a list under "cards"
                                # If it returns just a dict, maybe it treated batch as one item?
                                improved_cards = [response_json] # Try treating as single list item
                            else:
                                raise ValueError("Unexpected JSON format")
                                
                            improved_map = {str(item.get("id")): item for item in improved_cards if "id" in item}
                            
                            for nid in batch_notes:
                                note = batch_notes[nid]
                                nid_str = str(nid)
                                
                                if nid_str in improved_map:
                                    item = improved_map[nid_str]
                                    if item.get("changed", False):
                                        _update_note_fields(note, item)
                                    
                                    if (preview or limit is not None):
                                         _display_preview(console, nid, note, item, preview)

                                if not preview:
                                    note.add_tag(processed_tag)
                                    col.update_note(note)
                                
                                processed_count += 1
                                progress.advance(task)

                            batch_success = True
                            break 

                        except (json.JSONDecodeError, ValueError) as e:
                            console.print(f"[yellow]Attempt {attempt+1}/{max_retries} failed: {e}. Retrying...[/yellow]")
                    else:
                        console.print(f"[yellow]Attempt {attempt+1}/{max_retries} failed: No JSON found. Retrying...[/yellow]")

                except Exception as e:
                    console.print(f"[red]Error processing batch attempt {attempt+1}: {e}[/red]")
            
            if not batch_success:
                 console.print(f"[red]Batch failed after {max_retries} attempts.[/red]")
                 if not preview:
                     for nid in batch_notes:
                         note = batch_notes[nid]
                         note.add_tag(processed_tag)
                         col.update_note(note)
                         progress.advance(task)

            if not preview:
                col.save()

    col.save()
    col.close()
    console.print(f"[bold green]Processing complete![/bold green] Processed {processed_count} notes.")

    # 5. Repackaging
    _repackage_apkg(working_dir, output_apkg, console, preview)
    console.print("[bold green]Done![/bold green]")
