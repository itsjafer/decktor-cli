<div align="center">
  <img src="media/banner.png" width="100%" />
</div>

# DeckTor (CLI)

DeckTor is a command-line tool to improve your Anki decks using Google's Gemini API. It processes cards individually to fix errors, improve clarity, or add content, giving you granular control over your flashcards.

Note: This was almost entirely vibe-coded off the original repo.

## Quick Start
Don't want to install anything? Run DeckTor directly in your browser:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/itsjafer/decktor-cli/blob/main/notebooks/quickstart.ipynb)

## Features

- **Scheduling Preserved:** Preserves scheduling information of the cards by modifying them in-place.
- **Gemini API Powered:** Uses Google's Gemini models (Flash, Flash Lite) for fast and effective card processing.
- **Batch Processing:** Robust CLI for processing large decks with resume capability.
- **GPT4Free Support:** Includes a "GPT4Free (Free)" option powered by `g4f`, allowing usage without an API key (only for testing, not recommended for large scale processing).
- **Preview Mode:** Dry-run your changes before committing them to a new deck.
- **Resumable:** If the process is interrupted, simply run the command again to pick up where you left off.

## Requirements

- **Python 3.9+**
- **Google Gemini API Key:** Required for using Gemini models. Get yours [here](https://aistudio.google.com/api-keys). Cost estimate: Using the flash lite model, processing 5000 cards cost me about $0.40.
    - **GPT4Free** is also supported for usage without an API key, but not recommended for large scale processing.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/itsjafer/decktor-cli.git
   cd decktor-cli
   ```

2. **Set up a Virtual Environment (Recommended):**

   ```bash
   # Create a virtual environment
   python3 -m venv .venv
   
   # Activate it
   source .venv/bin/activate
   ```

3. **Install the package:**

   ```bash
   pip install .
   ```

4. **Configure Environment:**

   Create a `.env` file in the root directory and add your Gemini API key:

   ```bash
   GEMINI_API_KEY=your_actual_api_key_here
   ```

## Usage

The primary command is `decktor process`.

```bash
decktor process input.apkg output.apkg
```

### Options

- `--model`: Choose the model to use. Default: `Gemini 2.5 Flash Lite`.
    - Example: `--model "Gemini 2.5 Flash"`
    - Example: `--model free` (Free mode)
- `--batch-size`: Number of cards to process before saving progress. Default: `10`.
    - Example: `--batch-size 20`
- `--preview`: Run in "dry run" mode. No changes are written to disk; changes are shown in the terminal.
- `--limit`: Process only the first N cards. Useful for testing.
    - Example: `--limit 5`
- `--prompt`: Path to a custom prompt text file. The prompt should output JSON.
    - Example: `--prompt my_custom_prompt.txt`
- `--exclude-fields`: Comma-separated list of fields to **exclude** from the LLM context. These fields will be ignored by the LLM and preserved untouched in the output.
    - Example: `--exclude-fields "Audio,Image"`
- `--working-dir`: Directory for intermediate files (default: `.decktor_work`). **Keep this directory to resume if interrupted.**
- `--reprocess-all`: Reprocess all cards, ignoring the 'decktor-processed' tag. Useful if you want to force a re-run on an already processed deck.

### Example Workflow

The default prompt assumes you're feeding it basic front/back cards. If you have a different type of card, adjust your prompt accordingly (see src/prompts/default.txt).

1. **Use the default improvement prompt**
   ```bash
   decktor process my_deck.apkg improved_deck.apkg
   ``` 

1. **Preview changes on 5 cards:**
   ```bash
   decktor process my_deck.apkg improved_deck.apkg --preview --limit 5
   ```

2. **Process specific fields (filtering out Audio):**
   ```bash
   decktor process my_deck.apkg improved_deck.apkg --exclude-fields "Audio" --prompt my_prompt.txt
   ```

3. **Resume if interrupted:**
   (Run the exact same command again)
   ```bash
   decktor process my_deck.apkg improved_deck.apkg --exclude-fields "Audio"
   ```

## License

MIT License.
