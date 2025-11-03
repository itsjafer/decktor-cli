import os
import subprocess
from io import BytesIO

import streamlit as st

from decktor.core import get_llm_model, improve_card, read_apkg_cards
from decktor.models import get_supported_model_names
from decktor.utils import get_prompt_template

DEFAULT_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "default.txt")


@st.cache_data
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
            read_apkg_cards(apkg_stream=anki_file_io)
            exit(0)
            with st.spinner("Loading LLM model..."):
                model, tokenizer = get_llm_model(llm_model)
            with st.spinner(f"Processing **{anki_file.name}** with model: **{llm_model}**"):
                current_prompt = st.session_state.prompt_template

                # Store original content
                st.session_state.original_content = anki_file.read().decode("utf-8")

                improved_card = improve_card(
                    st.session_state.original_content, model, tokenizer, current_prompt
                )
                st.session_state.improved_content = improved_card
            st.success("Processing complete!")

    st.divider()
    st.header("Results")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Original Cards")
        st.text_area(
            "Original",
            value=st.session_state.original_content,
            height=400,
            disabled=True,
            label_visibility="collapsed",
        )

    with col2:
        st.markdown("#### Improved Cards")
        st.text_area(
            "Improved",
            value=st.session_state.improved_content,
            height=400,
            label_visibility="collapsed",
        )


if __name__ == "__main__":
    run()
