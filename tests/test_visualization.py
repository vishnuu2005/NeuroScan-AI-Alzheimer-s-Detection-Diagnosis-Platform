import json
import pytest
from chatbot.formatters import format_stage_label, get_risk_tier, prepare_chart_dataset, CLINICAL_ORDER

def test_format_stage_label():
    assert format_stage_label("NonDemented") == "Non-Demented"
    assert format_stage_label("VeryMildDemented") == "Very Mild Dementia"
    assert format_stage_label("MildDemented") == "Mild Dementia"
    assert format_stage_label("ModerateDemented") == "Moderate Dementia"
    assert format_stage_label("Uncertain") == "Uncertain / Inconclusive"
    assert format_stage_label("UnknownStage") == "UnknownStage"

def test_get_risk_tier():
    assert get_risk_tier("NonDemented") == "low"
    assert get_risk_tier("VeryMildDemented") == "early"
    assert get_risk_tier("MildDemented") == "mild"
    assert get_risk_tier("ModerateDemented") == "moderate"
    assert get_risk_tier("Uncertain") == "unknown"
    assert get_risk_tier("Other") == "unknown"

def test_prepare_chart_dataset_empty():
    ds = prepare_chart_dataset({})
    assert ds["labels"] == []
    assert ds["values"] == []
    assert ds["bg_colors"] == []

def test_prepare_chart_dataset_ordering_and_highlight():
    raw_probs = {
        "ModerateDemented": 5.2,
        "NonDemented": 88.4,
        "VeryMildDemented": 4.1,
        "MildDemented": 2.3
    }
    
    ds = prepare_chart_dataset(raw_probs, predicted_stage="NonDemented")
    
    # Must be sorted in progressive clinical severity
    assert ds["labels"] == ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
    assert ds["display_labels"] == [
        "Non-Demented", "Very Mild Dementia", "Mild Dementia", "Moderate Dementia"
    ]
    assert ds["values"] == [88.4, 4.1, 2.3, 5.2]
    
    # NonDemented is winning class: border opacity should be full 1.0
    assert "1.0" in ds["border_colors"][0]
    # Non-winning classes have reduced opacity 0.6
    assert "0.6" in ds["border_colors"][1]

def test_chart_dataset_sanitization():
    raw_probs = {
        "<script>alert(1)</script>": 50.0,
        "NonDemented": 50.0
    }
    ds = prepare_chart_dataset(raw_probs)
    # Ensure json dump is serializable without exceptions
    dumped = json.dumps(ds)
    loaded = json.loads(dumped)
    assert len(loaded["values"]) == 2
