import click

from decktor.app import run_entrypoint as run_entrypoint
from decktor.download_models import main as download_models
from decktor.models import SUPPORTED_MODELS


@click.group()
def main_group():
    """A simple tool to fix Anki cards using an LLM."""
    pass


# We wrap the existing function to fit the new CLI structure
@main_group.command(name="run")
def run_command():
    """
    Runs the main Decktor logic to process Anki cards.
    """
    run_entrypoint()


@main_group.command(name="download-models")
@click.option(
    "--models",
    multiple=True,
    default=list(SUPPORTED_MODELS.keys()),
    help="List of model names to download. If not provided, all supported models will be downloaded.",
)
def download_models_command(models):
    """
    Downloads required LLM and Vision-Language models.
    """
    download_models(models)
