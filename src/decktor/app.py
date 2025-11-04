import os
import subprocess
from io import BytesIO
import json

import streamlit as st
from PIL import Image

from decktor.core import get_llm_model, improve_card, read_apkg_cards
from decktor.models import get_supported_model_names
from decktor.utils import get_prompt_template, img_to_bytes

DEFAULT_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "default.txt")


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


@st.cache_data
def load_css(file_name="main.css"):
    """Loads a CSS file from the 'styles' directory."""
    styles_dir = os.path.join(os.path.dirname(__file__), "styles")
    css_path = os.path.join(styles_dir, file_name)
    print(f"Loading CSS from: {css_path}")
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def parse_card_json(card_str):
    """Parse card string into dict."""
    if isinstance(card_str, dict):
        return card_str
    return json.loads(card_str)



def show_card_comparison(original_dict, improved_dict):
    """Show a side-by-side comparison of cards with visual indicators."""
    
    # Parse if needed
    if isinstance(improved_dict, str):
        try:
            improved_dict = json.loads(improved_dict)
        except:
            improved_dict = {"front": improved_dict, "back": "", "changed": False}
    
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
            else:
                st.info(improved_back if improved_back else "_Empty_")
    
    # Show change indicator and reason
    if changed and reason:
        st.markdown("---")
        col1, col2 = st.columns([1, 4])
        with col1:
            if front_changed or back_changed:
                st.markdown("✅ **Changed**")
            else:
                st.markdown("**Unchanged**")
        with col2:
            st.caption(f"**Reason:** {reason}")
    elif not changed:
        st.markdown("---")
        st.markdown("**No changes needed** - Card already meets quality standards")


def show_results(card: str, improved_card: str):
    """Display original and improved card in a beautiful side-by-side comparison."""
    st.divider()
    
    # Parse the cards
    original_dict = parse_card_json(card)
    improved_dict = parse_card_json(improved_card)
    
    # Add card to processed cards
    card_data = {
        "original": original_dict,
        "improved": improved_dict,
        "status": "pending"
    }
    st.session_state.processed_cards.append(card_data)
    st.session_state.cards_processed += 1
    
    # Show comparison
    show_card_comparison(original_dict, improved_dict)


def run_entrypoint():
    subprocess.Popen(["streamlit", "run", "src/decktor/app.py", "--server.address=127.0.0.1"])


def run():
    """Run the Streamlit app for DeckTor."""
    # Load logo for page icon
    logo_path = os.path.join("media", "logo.png")
    logo_img = Image.open(logo_path)
    
    st.set_page_config(
        page_title="DeckTor",
        page_icon=logo_img,
        layout="wide",
        initial_sidebar_state="expanded"
    )

    load_css()
    initialize_session_state()

    with st.sidebar:
        st.markdown("## Configuration")

        # Step 1: Model Selection
        llm_model = st.selectbox(
            "**Select LLM Model**",
            get_supported_model_names(),
            help="Choose the language model to process your cards.",
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
            help="This template guides the LLM in improving your cards."
        )
        
        # Show processing stats in sidebar if processing is complete
        if st.session_state.processing_complete and st.session_state.processed_cards:
            st.divider()
            st.markdown("## 📊 Processing Summary")
            total = len(st.session_state.processed_cards)
            st.metric("Total Cards", total)
            st.metric("Cards Processed", st.session_state.cards_processed)
            
            if total > 0:
                progress_pct = (st.session_state.cards_processed / total) * 100
                st.progress(progress_pct / 100)

    # Main content area
    st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 10px;">
            <img src='data:image/png;base64,{img_to_bytes(logo_path)}' width="50" style="vertical-align: middle;">
            <h1 style="margin: 0; display: inline;">DeckTor</h1>
        </div>
    """, unsafe_allow_html=True)

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
                help="Export from Anki: File → Export Deck (uncheck 'Include Media')"
            )
        
        st.markdown("#### Step 2: Process Cards")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fix_button = st.button(
                "Improve Cards", 
                type="primary",
                use_container_width=True,
                disabled=anki_file is None
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

                with st.spinner(f"Loading model: **{llm_model}**..."):
                    model, tokenizer = get_llm_model(llm_model)
                
                st.success(f"Model loaded successfully!")
                
                # Process each card
                for idx, card in enumerate(cards):
                    status_text.text(f"Processing card {idx + 1} of {len(cards)}...")
                    cards_metric.metric("Processed", f"{idx + 1}/{len(cards)}")
                    
                    current_prompt = st.session_state.prompt_template
                    st.session_state.original_content = card

                    processed_card = improve_card(
                        str(st.session_state.original_content), 
                        model, 
                        tokenizer, 
                        current_prompt
                    )
                    st.session_state.improved_content = processed_card

                    # Update progress
                    progress = (idx + 1) / len(cards)
                    progress_bar.progress(progress)
                    
                    # Show result
                    show_results(str(card), processed_card)

                st.session_state.processing_complete = True
                status_text.text("Processing complete!")
                st.balloons()
                st.success(f"Successfully processed {len(cards)} cards!")
                st.info("Switch to the **Review Cards** tab to review and accept/reject changes")



if __name__ == "__main__":
    run()
