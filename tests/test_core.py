
import pytest
from unittest.mock import MagicMock
from decktor.core import _clean_field_html, _update_note_fields
from decktor.utils import make_prompt

def test_clean_field_html():
    """Test that HTML is correctly stripped from fields."""
    html = "<div>Hello <b>World</b></div>"
    expected = "Hello \nWorld"
    assert _clean_field_html(html) == expected
    
    html_simple = "Just text"
    assert _clean_field_html(html_simple) == "Just text"

def test_update_note_fields():
    """Test updating note fields from LLM response item."""
    # Mock an Anki note
    mock_note = MagicMock()
    # Mock the behavior of note['Field'] access
    data = {"Front": "Old Front", "Back": "Old Back"}
    
    def get_item(key):
        return data[key]
    
    def set_item(key, value):
        data[key] = value
        
    mock_note.__getitem__.side_effect = get_item
    mock_note.__setitem__.side_effect = set_item
    
    # Mock note_type structure
    mock_note.note_type.return_value = {
        "flds": [{"name": "Front"}, {"name": "Back"}]
    }
    
    # Update item
    update_item = {
        "Front": "New Front",
        "changed": True
    }
    
    updated = _update_note_fields(mock_note, update_item)
    
    assert updated is True
    assert data["Front"] == "New Front"
    assert data["Back"] == "Old Back" # Should remain unchanged

def test_update_note_fields_no_change():
    """Test that fields are not updated if keys don't match."""
    mock_note = MagicMock()
    data = {"Front": "Old Front"}
    mock_note.note_type.return_value = {"flds": [{"name": "Front"}]}
    
    update_item = {"OtherField": "Value"}
    
    updated = _update_note_fields(mock_note, update_item)
    
    assert updated is False

def test_make_prompt():
    """Test prompt interpolation."""
    template = "Fix: {card}"
    card_str = "My Card"
    assert make_prompt(card_str, template) == "Fix: My Card"

