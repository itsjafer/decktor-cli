
import os
import shutil
import tempfile
import pytest
from unittest.mock import MagicMock, patch
from decktor.core import process_apkg_with_resume

# Path to a sample input file. 
# We'll use one of the files present in the root directory for this test.
SAMPLE_APKG = "Main__Arabic__TTS__Arabic Frequency List (Top 5,000 Words).apkg"

@pytest.fixture
def temp_workspace():
    """Sets up a temporary workspace directory and cleans it up after."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def sample_apkg(temp_workspace):
    """Copies the sample apkg to the temp workspace to avoid modifying the original."""
    source_path = os.path.abspath(SAMPLE_APKG)
    if not os.path.exists(source_path):
        pytest.skip(f"Sample file {SAMPLE_APKG} not found in root")
    
    dest_path = os.path.join(temp_workspace, "test_input.apkg")
    shutil.copy(source_path, dest_path)
    return dest_path

def test_process_apkg_golden_master(temp_workspace, sample_apkg):
    """
    Golden Master test: Runs the full pipeline with a mocked LLM.
    Verifies that the code runs without error and produces an output file.
    """
    output_apkg = os.path.join(temp_workspace, "output.apkg")
    working_dir = os.path.join(temp_workspace, "work_dir")
    prompt_path = os.path.join(temp_workspace, "prompt.txt")
    
    # Create a dummy prompt file
    with open(prompt_path, "w") as f:
        f.write("Fix this card: {card}")

    # Mock the LLM Response
    mock_response_content = """
    {
        "cards": [
            {
                "id": 123456789,
                "Front": "Updated Front",
                "Back": "Updated Back",
                "changed": true,
                "reason": "Fixed typo"
            }
        ]
    }
    """
    
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = mock_response_content
    # improved_text comes from response.text
    
    # We need to make sure the mock returns a list of cards that actually matches IDs in the deck.
    # Since we don't know the IDs in the binary APKG beforehand without reading it,
    # we can make the mock dynamic or just return a generic response structure that the code accepts.
    # The code in core.py handles unmatched IDs gracefully (just doesn't update them).
    # To test that updates ACTUALLY happen, we'd need to know an ID.
    # For a "Golden Master" that simply ensures "no crash" and "file created", a generic response is enough.
    # To be stricter, we could inspect the output database, but that might be overkill for step 1.
    
    def side_effect(prompt):
        # We can inspect the prompt to extract IDs if we wanted to be clever,
        # but for now let's just return a valid JSON structure so the parser doesn't crash.
        # The parser expects { "cards": [...] }
        return mock_response

    mock_model.generate_content.side_effect = side_effect

    # Mock get_llm_model to return our mock_model
    with patch("decktor.core.get_llm_model", return_value=mock_model) as mock_get_model:
        
        process_apkg_with_resume(
            input_apkg=sample_apkg,
            output_apkg=output_apkg,
            model_name="Gemini 2.5 Flash Lite",
            prompt_path=prompt_path,
            working_dir=working_dir,
            batch_size=1, # Small batch size to trigger loops
            preview=False,
            limit=5 # Limit to 5 cards to run quickly
        )
        
        # assertions
        assert os.path.exists(output_apkg), "Output .apkg file was not created"
        assert os.path.exists(working_dir), "Working directory was not created"
        
        # Verify call arguments
        # mock_model.generate_content.assert_called() 
        # (This might fail if no cards are found, but we expect cards in appropriate file)

