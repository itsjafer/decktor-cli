import json
import logging
import os
import subprocess
from io import BytesIO

import streamlit as st
import torch
from PIL import Image

from decktor.core import get_llm_model, improve_card, read_apkg_cards
from decktor.models import get_supported_model_names
from decktor.utils import (
    get_prompt_template,
    img_to_bytes,
)

DEFAULT_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "default.txt")
logger = logging.getLogger(__name__)


def load_prompt(path):
    try:
        return get_prompt_template(path)
    except Exception:
        return "Error: Could not load default prompt. Please paste your prompt."


def initialize_session_state():
    """Initialize all session state variables."""
    if "improved_content" not in st.session_state:
        st.session_state.improved_content = ""
    if "original_content" not in st.session_state:
        st.session_state.original_content = ""
    if "processed_cards" not in st.session_state:
        st.session_state.processed_cards = []
    if "card_statuses" not in st.session_state:
        st.session_state.card_statuses = {}
    if "current_card_index" not in st.session_state:
        st.session_state.current_card_index = 0
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False
    if "total_cards" not in st.session_state:
        st.session_state.total_cards = 0
    if "cards_processed" not in st.session_state:
        st.session_state.cards_processed = 0
    if "quantization" not in st.session_state:
        st.session_state.quantization = False
    if "performance_metrics" not in st.session_state:
        st.session_state.performance_metrics = []
    if "thinking_mode" not in st.session_state:
        st.session_state.thinking_mode = False


def show_performance(metrics: dict):
    with st.expander("Performance Metrics", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tokens/sec", f"{metrics['throughput_tokens_per_second']:.1f}")
        with col2:
            st.metric("Time per card", f"{metrics['total_time']:.2f}s")
        with col3:
            st.metric("Generation Time", f"{metrics['generation_time']:.2f}s")
        col4, col5 = st.columns(2)
        with col4:
            st.metric("Input Tokens", metrics["total_input_tokens_unpadded"])
        with col5:
            st.metric("Output Tokens", metrics["total_output_tokens"])


@st.cache_data
def load_css(file_name="main.css"):
    """Loads a CSS file from the 'styles' directory."""
    styles_dir = os.path.join(os.path.dirname(__file__), "styles")
    css_path = os.path.join(styles_dir, file_name)
    print(f"Loading CSS from: {css_path}")
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def accept_card():
    idx = st.session_state.current_card_index
    st.session_state.processed_cards[idx]["status"] = "accepted"
    if idx < len(st.session_state.processed_cards) - 1:
        st.session_state.current_card_index = idx + 1


def reject_card():
    idx = st.session_state.current_card_index
    st.session_state.processed_cards[idx]["status"] = "rejected"
    if idx < len(st.session_state.processed_cards) - 1:
        st.session_state.current_card_index = idx + 1


def show_card_comparison(original_dict: dict, improved_dict: dict):
    """Show a side-by-side comparison of cards with visual indicators."""

    original_front = original_dict.get("front", "")
    original_back = original_dict.get("back", "")
    improved_front = improved_dict.get("front", original_front)
    improved_back = improved_dict.get("back", original_back)
    changed = improved_dict.get("changed", False)
    reason = improved_dict.get("reason", "")

    # Check if there are actual changes
    front_changed = original_front != improved_front
    back_changed = original_back != improved_back

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Original Card")
        with st.container(border=True):
            st.markdown("**Front:**")
            st.info(original_front if original_front else "_Empty_")
            st.markdown("**Back:**")
            st.info(original_back if original_back else "_Empty_")

    with col2:
        st.markdown("#### Improved Card")
        with st.container(border=True):
            st.markdown("**Front:**")
            if front_changed:
                st.success(improved_front if improved_front else "_Empty_")
            else:
                st.info(improved_front if improved_front else "_Empty_")

            st.markdown("**Back:**")
            if back_changed:
                st.success(improved_back if improved_back else "_Empty_")
                st.markdown(f"**Reason for change:** {reason}")
            else:
                st.info(improved_back if improved_back else "_Empty_")


def show_results(card: dict, improved_card: str):
    """Display original and improved card in a beautiful side-by-side comparison."""
    st.divider()

    original_dict = card
    improved_dict = json.loads(improved_card)

    # Add card to processed cards
    card_data = {
        "original": original_dict,
        "improved": improved_dict,
        "status": "pending",
    }
    st.session_state.processed_cards.append(card_data)
    st.session_state.cards_processed += 1

    # Show comparison
    try:
        show_card_comparison(original_dict, improved_dict)
    except Exception as e:
        st.error(f"Error displaying card comparison: {e}")


def run_entrypoint():
    subprocess.Popen(["streamlit", "run", "src/decktor/app.py", "--server.address=127.0.0.1"])


def clear_gpu_cache():
    """Clear GPU cache if available."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache")
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
            logger.info("Cleared MPS cache")
    except Exception as e:
        logger.warning(f"Failed to clear GPU cache: {e}")


def restart_app():
    """Restart the Streamlit app by clearing session state and GPU cache."""
    try:
        # Clear all session state keys properly
        for key in list(st.session_state.keys()):
            del st.session_state[key]

        # Reinitialize with defaults
        initialize_session_state()

        # Clear GPU memory
        clear_gpu_cache()
    except Exception as e:
        logger.error(f"Error during app restart: {e}")
        st.error(f"Failed to restart app: {e}")


def close_app():
    """Close the Streamlit app."""
    try:
        clear_gpu_cache()
        st.success("App closed successfully. You can close this browser tab.")
        st.stop()
    except Exception as e:
        logger.error(f"Error during app closure: {e}")
        st.error(f"Error closing app: {e}")


def run():
    """Run the Streamlit app for DeckTor."""
    # Load logo for page icon
    logo_path = os.path.join("media", "logo.png")
    logo_img = Image.open(logo_path)

    st.set_page_config(
        page_title="DeckTor", page_icon=logo_img, layout="wide", initial_sidebar_state="expanded"
    )

    load_css()
    initialize_session_state()

    with st.sidebar:
        st.markdown("## Controls")
        st.button("Restart App", type="secondary", use_container_width=True, on_click=restart_app)
        st.button("Close App", type="secondary", use_container_width=True, on_click=close_app)

        st.markdown("## Configuration")

        # Step 1: Model Selection
        llm_model = st.selectbox(
            "**Select LLM Model**",
            get_supported_model_names(),
            help="Choose the language model to process your cards.",
        )
        quantization = st.toggle(
            "Use 4-bit Quantization (saves memory, necessary for large models)",
            value=st.session_state.quantization,
            help="Enable 4-bit quantization to reduce memory usage at the cost of some model quality.",
            key="quantization",
        )
        thinking_mode = st.toggle(
            "Use Thinking Mode (slower but more accurate)",
            value=st.session_state.thinking_mode,
            help="Enable Thinking Mode to improve response quality at the cost of speed.",
            key="thinking_mode",
        )

        st.divider()

        # Step 2: Prompt Settings
        st.markdown("## Prompt Settings")
        st.caption(
            "The default prompt template is optimized for card improvement. Modify if needed."
        )

        default_prompt_text = load_prompt(DEFAULT_PROMPT_PATH)

        # The 'key' links this to st.session_state.prompt_template
        st.text_area(
            "Prompt Template",
            value=default_prompt_text,
            height=300,
            key="prompt_template",
            help="This template guides the LLM in improving your cards.",
        )

    # Main content area
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 10px;">
            <img src='data:image/png;base64,{img_to_bytes(logo_path)}' width="50" style="vertical-align: middle;">
            <h1 style="margin: 0; display: inline;">DeckTor</h1>
        </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("### Your Anki Doctor. Improve your flashcards with local AI")

    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "📝 Review Cards", "📊 Summary"])

    with tab1:
        # Upload section
        st.markdown("#### Step 1: Upload Your Deck")
        st.markdown("Export your deck from Anki as `.apkg` (without media)")

        with st.container():
            anki_file = st.file_uploader(
                "Choose your Anki deck file",
                type=["apkg"],
                help="Export from Anki: File → Export Deck (uncheck 'Include Media')",
            )

        st.markdown("#### Step 2: Process Cards")

        col1, col2 = st.columns([1, 1])

        with col1:
            fix_button = st.button(
                "Improve Cards",
                type="primary",
                use_container_width=True,
                disabled=anki_file is None,
            )

        # Show info message if no file uploaded
        if anki_file is None and not fix_button:
            st.info("Please upload an Anki deck file to get started")

        # Processing logic
        if fix_button:
            if anki_file is None:
                st.error("Please upload an Anki file first!")
            else:
                # Reset previous results
                st.session_state.processed_cards = []
                st.session_state.cards_processed = 0
                st.session_state.processing_complete = False

                st.markdown("---")
                st.markdown("### 🔄 Processing Your Cards")

                print("Processing uploaded .apkg file...")
                anki_file_io = BytesIO(anki_file.read())
                cards = read_apkg_cards(apkg_stream=anki_file_io)

                st.session_state.total_cards = len(cards)

                # Show progress info
                progress_container = st.container()
                with progress_container:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Cards", len(cards))
                    with col2:
                        cards_metric = st.empty()

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    model, tokenizer = get_llm_model(
                        llm_model, quantize=st.session_state.quantization
                    )

                st.success(
                    "Model loaded successfully! Once the processing is complete, you can review the changes in the 'Review Cards' tab."
                )

                # Reset performance metrics
                st.session_state.performance_metrics = []

                for i, card in enumerate(cards):
                    status_text.text(f"Processing card {i + 1} of {len(cards)}...")
                    cards_metric.metric("Processed", f"{i}/{len(cards)}")

                    st.session_state.original_content = card
                    current_prompt = st.session_state.prompt_template

                    processed_card, metrics = improve_card(
                        str(card),
                        model,
                        tokenizer,
                        current_prompt,
                        thinking_mode=st.session_state.thinking_mode,
                    )

                    st.session_state.improved_content = processed_card

                    # Show result with metrics
                    try:
                        show_results(card, processed_card)
                    except Exception as e:
                        print(f"Error displaying results for card {i + 1}: {e}")
                        continue

                    show_performance(metrics)

                    progress = (i + 1) / len(cards)
                    progress_bar.progress(progress)

                st.session_state.processing_complete = True
                status_text.text("Processing complete!")
                st.balloons()
                st.success(f"Successfully processed {len(cards)} cards!")
                st.info("Switch to the **Review Cards** tab to review and accept/reject changes")

    with tab2:
        # Interactive Review Tab
        if not st.session_state.processed_cards:
            st.info(
                "Cards will appear here as they are processed. Start processing in the Upload tab!"
            )
        else:
            # Header with progress
            total_cards = len(st.session_state.processed_cards)
            accepted = sum(
                1 for c in st.session_state.processed_cards if c.get("status") == "accepted"
            )
            rejected = sum(
                1 for c in st.session_state.processed_cards if c.get("status") == "rejected"
            )
            pending = total_cards - accepted - rejected

            st.markdown(f"### Review Cards ({accepted + rejected}/{total_cards} reviewed)")

            # Compact progress bar
            progress_pct = (accepted + rejected) / total_cards if total_cards > 0 else 0
            st.progress(
                progress_pct,
                text=f"{accepted} accepted ✅  |  {rejected} rejected ❌ |  {pending} pending ⏳",
            )

            # Card navigation - simplified
            col1, col2, col3 = st.columns([1, 2, 1])

            with col1:
                if st.button(
                    "⬅️ Previous",
                    disabled=st.session_state.current_card_index == 0,
                    use_container_width=True,
                ):
                    st.session_state.current_card_index -= 1

            with col2:
                # Jump to card selector
                card_options = [f"Card {i + 1}" for i in range(total_cards)]
                selected = st.selectbox(
                    "Jump to card:",
                    options=range(total_cards),
                    format_func=lambda x: f"Card {x + 1}",
                    index=st.session_state.current_card_index,
                    label_visibility="collapsed",
                )
                if selected != st.session_state.current_card_index:
                    st.session_state.current_card_index = selected

            with col3:
                if st.button(
                    "Next ➡️",
                    disabled=st.session_state.current_card_index >= total_cards - 1,
                    use_container_width=True,
                ):
                    st.session_state.current_card_index += 1

            # Current card display
            current_card = st.session_state.processed_cards[st.session_state.current_card_index]
            current_status = current_card.get("status", "pending")

            # Compact status indicator
            status_emoji = {"accepted": "✅", "rejected": "❌", "pending": ""}
            st.markdown(
                f"### {status_emoji.get(current_status, '')} Card {st.session_state.current_card_index + 1}"
            )

            # Show card comparison
            show_card_comparison(current_card["original"], current_card["improved"])

            col1, col2 = st.columns([1, 1])

            with col1:
                button_accept = st.button(
                    "**Accept**",
                    use_container_width=True,
                    key="accept_btn",
                    icon="✅",
                    on_click=accept_card,
                )

            with col2:
                button_reject = st.button(
                    "**Reject**",
                    use_container_width=True,
                    key="reject_btn",
                    icon="❌",
                    on_click=reject_card,
                )

            # Quick actions (collapsible)
            with st.expander("Quick Actions"):
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(
                        "Accept All Remaining", use_container_width=True, key="accept_all"
                    ):
                        for card in st.session_state.processed_cards:
                            if card.get("status") == "pending":
                                card["status"] = "accepted"
                with col2:
                    if st.button(
                        "Reject All Remaining", use_container_width=True, key="reject_all"
                    ):
                        for card in st.session_state.processed_cards:
                            if card.get("status") == "pending":
                                card["status"] = "rejected"


if __name__ == "__main__":
    run()
