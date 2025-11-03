import os
import subprocess

import streamlit as st

from decktor.core import get_llm_model, improve_card
from decktor.models import get_supported_model_names
from decktor.utils import get_prompt_template

DEFAULT_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "default.txt")


def run_entrypoint():
    subprocess.Popen(["streamlit", "run", "src/decktor/app.py"])


def run():
    """Run the Streamlit app for DeckTor."""
    st.set_page_config(page_title="DeckTor", page_icon="🤖", layout="centered")

    st.title("DeckTor")
    st.write("Improve your Anki cards with an LLM.")

    # Load Anki txt file
    anki_file = st.file_uploader("**Step 1**: Upload your Anki .txt file", type=["txt"])

    llm_model = st.selectbox("**Step 2**: Select LLM Model", get_supported_model_names())

    # Prompt template section
    with st.expander("**(Optional)** Prompt Settings", expanded=False):
        prompt_path = st.text_input("Prompt path loaded by default", value=DEFAULT_PROMPT_PATH)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.text(
                "To update the prompt, change the path above and click the Update button on the right"
            )
        with col2:
            if st.button("Update prompt path"):
                st.success(f"Prompt path updated to: {prompt_path}")

        if st.button("Show prompt template"):
            try:
                prompt_template = get_prompt_template(prompt_path)
                st.code(prompt_template, language="text")
            except Exception as e:
                st.error(f"Error loading prompt template: {e}")

    # Action button
    fix_button = st.button("Improve Cards")

    if fix_button:
        if anki_file is None:
            st.write("You need to upload an Anki file first!")
        else:
            model, tokenizer = get_llm_model(llm_model)
            with st.spinner(f"Processing **{anki_file.name}** with model: **{llm_model}**"):
                improved_card = improve_card(
                    anki_file.read().decode("utf-8"), model, tokenizer, prompt_path
                )
                st.text_area("Improved Cards", value=improved_card, height=400)
            st.success("Processing complete!")


if __name__ == "__main__":
    run()
