import os
import subprocess
from io import BytesIO

import streamlit as st

from decktor.core import get_llm_model, improve_card, read_apkg_cards
from decktor.models import get_supported_model_names
from decktor.utils import get_prompt_template

DEFAULT_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "default.txt")


def load_prompt(path):
    try:
        return get_prompt_template(path)
    except Exception:
        return "Error: Could not load default prompt. Please paste your prompt."


@st.cache_data
def load_css(file_name="main.css"):
    """Loads a CSS file from the 'styles' directory."""
    styles_dir = os.path.join(os.path.dirname(__file__), "styles")
    css_path = os.path.join(styles_dir, file_name)
    print(f"Loading CSS from: {css_path}")
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def show_results(card: str, improved_card: str):
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.json(card)
    with col2:
        st.json(improved_card)


def run_entrypoint():
    subprocess.Popen(["streamlit", "run", "src/decktor/app.py", "--server.address=127.0.0.1"])


def run():
    """Run the Streamlit app for DeckTor."""
    st.set_page_config(
        page_title="DeckTor", page_icon="🤖", layout="centered", initial_sidebar_state="expanded"
    )

    load_css()

    if "improved_content" not in st.session_state:
        st.session_state.improved_content = ""
    if "original_content" not in st.session_state:
        st.session_state.original_content = ""

    with st.sidebar:
        st.header("⚙️ Configuration")

        # Step 1: Model Selection
        llm_model = st.selectbox(
            "**1. Select LLM Model**",
            get_supported_model_names(),
            help="Choose the language model to process your cards.",
        )

        st.divider()

        # Step 2: Prompt Settings
        st.header("📝 Prompt Settings")
        st.markdown(
            "The prompt template loaded by default and shown below is the recommended one. You can modify it as needed."
        )

        default_prompt_text = load_prompt(DEFAULT_PROMPT_PATH)

        # The 'key' links this to st.session_state.prompt_template
        st.text_area(
            "Prompt Template", value=default_prompt_text, height=300, key="prompt_template"
        )

    st.title("DeckTor")
    st.write("Your Anki Doctor. Improve your Anki cards with an LLM.")

    with st.container(border=True):
        anki_file = st.file_uploader("Upload your Anki .apkg file", type=["apkg"])

    # Action button
    fix_button = st.button("Improve Cards", type="primary")

    if fix_button:
        if anki_file is None:
            st.write("You need to upload an Anki file first!")
        else:
            print("Processing uploaded .apkg file...")
            anki_file_io = BytesIO(anki_file.read())
            cards = read_apkg_cards(apkg_stream=anki_file_io)

            with st.spinner("Loading LLM model..."):
                model, tokenizer = get_llm_model(llm_model)
            with st.spinner(f"Processing cards with model: **{llm_model}**"):
                for card in cards:
                    current_prompt = st.session_state.prompt_template

                    st.session_state.original_content = card

                    processed_card = improve_card(
                        str(st.session_state.original_content), model, tokenizer, current_prompt
                    )
                    st.session_state.improved_content = processed_card

                    show_results(str(card), processed_card)

            st.success("Processing complete!")


if __name__ == "__main__":
    run()
