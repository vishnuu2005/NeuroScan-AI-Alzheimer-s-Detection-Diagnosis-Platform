"""
Clinical AI consultation module for NeuroScan AI.
Integrates Groq LLaMA 3.3 70B with clinical system prompts, GDS staging guidelines,
and an offline clinical knowledge fallback engine.
"""

CLINICAL_SYSTEM_PROMPT = """You are NeuroScan AI's Clinical Consultation Assistant, specialized in neurodegenerative disorders and Alzheimer's disease staging.

CORE CLINICAL OBJECTIVES:
1. Provide empathetic, evidence-based, structured educational information regarding MRI scan classifications.
2. Emphasize that neuroimaging classification is assistive and NOT an independent medical diagnosis.
3. Strictly advise consultation with a board-certified neurologist or geriatrician for clinical assessment (including neuropsychological testing, amyloid PET, and CSF biomarkers).
4. Frame disease progression around standard clinical staging frameworks (such as the Global Deterioration Scale - GDS and Clinical Dementia Rating - CDR).
5. Suggest actionable questions for patients and families to bring to their next clinical consultation.

COMMUNICATION GUIDELINES:
- Use clean Markdown with clear headings (##, ###), bullet lists, and highlighted key terms (**bold**).
- Maintain an empathetic, objective, and reassuring tone.
- Never prescribe medication dosages or promise medical cures.
- If the user asks about non-medical matters, gently steer them back to neurological health.
"""

GDS_STAGES = {
    "NonDemented": {
        "title": "GDS Stage 1–2: Normal / Age-Associated Memory Impairment",
        "description": "No measurable cognitive decline. Subtle subjective complaints may occur, consistent with normal neuro-aging.",
        "symptoms": ["Occasional lapses in retrieving names or misplaced objects", "Normal functional independence maintained"],
        "recommendations": ["Adherence to Mediterranean/MIND dietary patterns", "Regular cardiovascular exercise (150 min/week)", "Cognitive engagement and sleep hygiene"]
    },
    "VeryMildDemented": {
        "title": "GDS Stage 3: Mild Cognitive Impairment (MCI)",
        "description": "Subtle deficits noticeable by close family or colleagues. Objective clinical testing reveals measurable memory impairment.",
        "symptoms": ["Difficulty remembering recent conversations or names of new acquaintances", "Trouble retaining material read", "Decreased organizational capacity in complex tasks"],
        "recommendations": ["Comprehensive neuropsychological evaluation", "Baseline cognitive screening (MoCA / MMSE)", "Establish routine calendar and organizational tools", "Address modifiable vascular risk factors"]
    },
    "MildDemented": {
        "title": "GDS Stage 4: Mild Dementia (Early-Stage Alzheimer's)",
        "description": "Clear-cut clinical deficits in complicated tasks such as managing personal finances or travelling alone to unfamiliar locations.",
        "symptoms": ["Knowledge of current and recent events impaired", "Difficulty with mental arithmetic (serial subtraction)", "Reduced emotional responsiveness and denial of deficits"],
        "recommendations": ["Consultation with a neurologist regarding pharmacological options (e.g. cholinesterase inhibitors)", "Advance care planning and healthcare proxy designations", "Home safety review (driving safety evaluation, medication management)"]
    },
    "ModerateDemented": {
        "title": "GDS Stage 5–6: Moderate to Moderately Severe Dementia",
        "description": "Substantial assistance required for daily activities. Significant disorientation to time, place, or season.",
        "symptoms": ["Inability to recall major aspects of current life (address, telephone number)", "Disorientation to date and season", "Difficulty selecting appropriate clothing for weather"],
        "recommendations": ["Close caregiver supervision and respite planning", "Structured daily routines and fall prevention measures", "Discussion of non-pharmacological behavioural strategies"]
    },
    "Uncertain": {
        "title": "Inconclusive / Equivocal Neuroimaging Finding",
        "description": "The AI model confidence was beneath the certainty threshold. Anatomical slice presentation requires professional radiological review.",
        "symptoms": ["Inconclusive single-slice findings", "Potential motion artifact or slice orientation variance"],
        "recommendations": ["Review volumetric 3D T1-weighted MRI with a neuroradiologist", "Correlate with formal cognitive screening (MoCA)", "Repeat imaging if clinically indicated"]
    }
}

def generate_local_clinical_response(user_message, stage="NonDemented", confidence=90.0, all_probs=None):
    """
    Offline clinical decision-support fallback.
    Answers patient and clinician inquiries using structured GDS guidelines when external LLMs are unreachable.
    """
    msg_lower = user_message.lower()
    stage_key = stage if stage in GDS_STAGES else "NonDemented"
    stage_data = GDS_STAGES[stage_key]
    stage_title = stage_data["title"]

    # Intent 1: Symptoms and clinical signs
    if any(w in msg_lower for w in ["symptom", "signs", "behavior", "memory", "forget", "notice"]):
        symptoms_md = "\n".join([f"- **{s}**" for s in stage_data["symptoms"]])
        return (
            f"## Clinical Symptom Profile: {stage_title}\n\n"
            f"Based on the classified neuroimaging stage (**{stage_key}**, {confidence:.1f}% confidence), characteristic clinical observations include:\n\n"
            f"{symptoms_md}\n\n"
            f"### Next Clinical Step\n"
            f"- Correlate these observations with a standardized cognitive assessment (such as the MoCA or MMSE).\n"
            f"- Document changes over 6-month intervals with a primary care physician or neurologist."
        )

    # Intent 2: Treatments and therapeutics
    if any(w in msg_lower for w in ["treat", "cure", "medicine", "medication", "drug", "pill", "therapy"]):
        return (
            f"## Therapeutic & Management Overview: {stage_title}\n\n"
            f"**Important:** Pharmacological treatments must always be prescribed and monitored by a qualified physician.\n\n"
            f"### Current Evidence-Based Approaches\n"
            f"- **Cholinesterase Inhibitors:** Commonly evaluated in mild-to-moderate dementia to support neurotransmitter function.\n"
            f"- **NMDA Receptor Antagonists:** Considered in moderate stages to regulate glutamate activity.\n"
            f"- **Lifestyle Interventions:** Intensive cardiovascular management, Mediterranean/MIND dietary intervention, and targeted cognitive rehabilitation.\n\n"
            f"### Recommended Questions for Your Neurologist\n"
            f"1. *Are disease-modifying therapies or clinical trials appropriate for my specific biomarker profile?*\n"
            f"2. *What non-pharmacological interventions are most beneficial at this stage?*"
        )

    # Intent 3: Lifestyle, prevention, and brain health
    if any(w in msg_lower for w in ["lifestyle", "diet", "food", "exercise", "prevent", "sleep"]):
        return (
            f"## Brain Health & Neuroprotective Guidance\n\n"
            f"Regardless of disease staging, evidence demonstrates that active lifestyle interventions support neurological resilience:\n\n"
            f"- **Aerobic Exercise:** 150 minutes per week of moderate-intensity activity stimulates BDNF (brain-derived neurotrophic factor).\n"
            f"- **MIND / Mediterranean Diet:** Emphasize leafy greens, berries, whole grains, nuts, and omega-3 fatty acids; minimize saturated fats and refined sugars.\n"
            f"- **Sleep Architecture:** Consistent 7–8 hours of restorative sleep aids glymphatic clearance of neurotoxic metabolites.\n"
            f"- **Cognitive Reserve:** Engage in novel intellectual challenges, multilingual activities, and continuous learning."
        )

    # Intent 4: Caregiver and family support
    if any(w in msg_lower for w in ["caregiver", "family", "help", "support", "home", "safety", "burnout"]):
        return (
            f"## Caregiver Support & Safety Framework\n\n"
            f"Caring for an individual navigating cognitive change requires a structured and compassionate support plan:\n\n"
            f"- **Home Safety:** Ensure adequate lighting, eliminate trip hazards, and evaluate medication dispensing systems.\n"
            f"- **Communication Strategies:** Speak in clear, calm sentences, allow ample time for responses, and validate emotional experiences.\n"
            f"- **Caregiver Respite:** Connect with local Alzheimer's Association chapters and support groups to prevent caregiver fatigue.\n"
            f"- **Legal & Financial Planning:** Establish advance healthcare directives, power of attorney, and care preferences early."
        )

    # Default: Comprehensive clinical overview
    rec_md = "\n".join([f"- {r}" for r in stage_data["recommendations"]])
    return (
        f"## Clinical Consultation: {stage_title}\n\n"
        f"**Neuroimaging Stage:** `{stage_key}` ({confidence:.1f}% confidence)\n\n"
        f"### Pathological & Anatomical Context\n"
        f"{stage_data['description']}\n\n"
        f"### Clinical Recommendations\n"
        f"{rec_md}\n\n"
        f"### Clinical Notice\n"
        f"This platform is an assistive AI research tool. MRI classifications require clinical correlation with blood biomarkers (p-tau217), cognitive testing, and neurological examination."
    )

