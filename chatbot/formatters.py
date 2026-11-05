"""
Visualization and Formatting Helpers for NeuroScan AI.
Provides clinical label formatting, risk tier assignment, and sanitized Chart.js datasets.
"""

CLINICAL_LABELS = {
    "NonDemented": "Non-Demented",
    "VeryMildDemented": "Very Mild Dementia",
    "MildDemented": "Mild Dementia",
    "ModerateDemented": "Moderate Dementia",
    "Uncertain": "Uncertain / Inconclusive"
}

TIER_COLORS = {
    "NonDemented": {"bg": "rgba(16, 185, 129, 0.75)", "border": "rgba(16, 185, 129, 1.0)"},
    "VeryMildDemented": {"bg": "rgba(245, 158, 11, 0.75)", "border": "rgba(245, 158, 11, 1.0)"},
    "MildDemented": {"bg": "rgba(249, 115, 22, 0.75)", "border": "rgba(249, 115, 22, 1.0)"},
    "ModerateDemented": {"bg": "rgba(239, 68, 68, 0.75)", "border": "rgba(239, 68, 68, 1.0)"},
    "Uncertain": {"bg": "rgba(100, 116, 139, 0.75)", "border": "rgba(100, 116, 139, 1.0)"}
}

CLINICAL_ORDER = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]

def format_stage_label(stage_key):
    """Returns patient-friendly clinical terminology for raw model class key."""
    return CLINICAL_LABELS.get(stage_key, stage_key)

def get_risk_tier(stage_key):
    """Categorizes Alzheimer stage into clinical risk classification."""
    mapping = {
        "NonDemented": "low",
        "VeryMildDemented": "early",
        "MildDemented": "mild",
        "ModerateDemented": "moderate",
        "Uncertain": "unknown"
    }
    return mapping.get(stage_key, "unknown")

def prepare_chart_dataset(all_probs, predicted_stage=None):
    """
    Transforms raw probability dictionary into ordered, colored dataset for Chart.js.
    Aligns classes in progressive clinical severity order.
    """
    if not all_probs:
        return {"labels": [], "display_labels": [], "values": [], "bg_colors": [], "border_colors": []}

    ordered_keys = [k for k in CLINICAL_ORDER if k in all_probs]
    # Include any extra keys not in standard order
    for k in all_probs:
        if k not in ordered_keys:
            ordered_keys.append(k)

    labels = []
    display_labels = []
    values = []
    bg_colors = []
    border_colors = []

    for key in ordered_keys:
        val = round(float(all_probs[key]), 2)
        labels.append(key)
        display_labels.append(format_stage_label(key))
        values.append(val)
        tier_cfg = TIER_COLORS.get(key, {"bg": "rgba(14, 165, 233, 0.7)", "border": "rgba(56, 189, 248, 1.0)"})
        
        # If this is the winning predicted class, brighten border
        if predicted_stage and key == predicted_stage:
            bg_colors.append(tier_cfg["bg"])
            border_colors.append(tier_cfg["border"])
        else:
            # Subtle opacity for non-predicted classes
            bg_colors.append(tier_cfg["bg"].replace("0.75", "0.45"))
            border_colors.append(tier_cfg["border"].replace("1.0", "0.6"))

    return {
        "labels": labels,
        "display_labels": display_labels,
        "values": values,
        "bg_colors": bg_colors,
        "border_colors": border_colors
    }
