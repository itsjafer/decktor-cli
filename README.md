<div align="center">
  <img src="media/banner.png" width="100%" />
</div>

# DeckTor (CLI)

DeckTor is a command-line tool to improve your Anki decks using Google's Gemini API. It processes cards individually to fix errors, improve clarity, or add content, giving you granular control over your flashcards.

## Features

- **Gemini API Powered:** Uses Google's Gemini models (Flash, Flash Lite) for fast and effective card processing.
- **Batch Processing:** Robust CLI for processing large decks with resume capability.
- **Preview Mode:** Dry-run your changes before committing them to a new deck.
- **Resumable:** If the process is interrupted, simply run the command again to pick up where you left off.

## Requirements

- **Python 3.10+**
- **Google Gemini API Key:** Required for using Gemini models.

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
- `--preview`: Run in "dry run" mode. No changes are written to disk; changes are shown in the terminal.
- `--limit`: Process only the first N cards. Useful for testing.
    - Example: `--limit 5`
- `--prompt`: Path to a custom prompt text file.
    - Example: `--prompt my_custom_prompt.txt`
- `--working-dir`: Directory for intermediate files (default: `.decktor_work`). **Keep this directory to resume if interrupted.**

### Example Workflow

1. **Preview changes on 5 cards:**
   ```bash
   decktor process my_deck.apkg improved_deck.apkg --preview --limit 5
   ```

2. **Process the full deck:**
   ```bash
   decktor process my_deck.apkg improved_deck.apkg
   ```

3. **Resume if interrupted:**
   (Run the exact same command again)
   ```bash
   decktor process my_deck.apkg improved_deck.apkg
   ```

## License

MIT License.
