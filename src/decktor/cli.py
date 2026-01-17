import click
import os

from decktor.core import process_apkg_with_resume
from decktor.models import SUPPORTED_MODELS


@click.group()
def main_group():
    """A simple tool to fix Anki cards using an LLM."""
    pass




@main_group.command(name="process")
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_path", type=click.Path(dir_okay=False))
@click.option(
    "--model",
    default="Gemini 2.5 Flash Lite",
    help="Name of the LLM model to use.",
)
@click.option(
    "--prompt",
    required=False,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the prompt template file. Defaults to cli_default.txt.",
)
@click.option(
    "--working-dir",
    default=".decktor_work",
    type=click.Path(file_okay=False),
    help="Directory to store intermediate files. Use the same directory to resume processing.",
)
@click.option(
    "--batch-size",
    default=10,
    help="Number of cards to process before saving progress.",
)
@click.option(
    "--preview",
    is_flag=True,
    default=False,
    help="Preview changes without saving to disk (Dry Run).",
)
@click.option(
    "--limit",
    default=None,
    type=int,
    help="Limit the number of cards to process.",
)
@click.option(
    "--exclude-fields",
    default=None,
    help="Comma-separated list of fields to exclude from the LLM prompt (e.g., 'Audio,Image').",
)
def process_command(
    input_path, output_path, model, prompt, working_dir, batch_size, preview, limit, exclude_fields
):
    if prompt is None:
        # Resolve default prompt path relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompt = os.path.join(current_dir, "prompts", "default.txt")
        if not os.path.exists(prompt):
            raise click.ClickException(f"Default prompt not found at {prompt}")

    exclude_fields_list = []
    if exclude_fields:
        exclude_fields_list = [f.strip() for f in exclude_fields.split(",")]

    """
    Process an .apkg file using an LLM.

    This command extracts the deck, iterates through cards, improves them using the specified LLM,
    and repackages the result. It supports resuming if interrupted by using the same --working-dir.
    """
    process_apkg_with_resume(
        input_apkg=input_path,
        output_apkg=output_path,
        model_name=model,
        prompt_path=prompt,
        working_dir=working_dir,
        batch_size=batch_size,
        preview=preview,
        limit=limit,
        exclude_fields=exclude_fields_list,
    )


if __name__ == "__main__":
    main_group()
