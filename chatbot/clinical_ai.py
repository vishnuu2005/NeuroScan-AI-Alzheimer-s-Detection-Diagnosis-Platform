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
