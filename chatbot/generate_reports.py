import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, Frame, PageTemplate
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus.flowables import Flowable

# ── Color Palette ──────────────────────────────────────────────────────────────
DARK        = colors.HexColor("#0f172a")
DARK_MID    = colors.HexColor("#1e293b")
SLATE       = colors.HexColor("#334155")
MUTED       = colors.HexColor("#64748b")
LIGHT_GRAY  = colors.HexColor("#f1f5f9")
BORDER      = colors.HexColor("#e2e8f0")
WHITE       = colors.white

STAGE_INFO = {
    "NonDemented": {
        "description": "No signs of dementia were detected in this MRI scan. Brain structure and volume appear within normal parameters.",
        "gds": "GDS Stage 1–2",
        "gds_sub": "No Cognitive Impairment",
        "symptoms": "No significant cognitive symptoms are associated with this classification.",
        "recommendations": [
            "Continue regular health check-ups and annual brain health monitoring.",
            "Maintain an active lifestyle — 150 min/week of aerobic exercise is recommended.",
            "Follow a Mediterranean-style diet rich in antioxidants and omega-3 fatty acids.",
            "Stay socially and mentally engaged to build cognitive reserve over time.",
            "Schedule follow-up MRI scans annually or as advised by your physician.",
        ],
        "prognosis": "No dementia indicators were found. Maintaining a healthy lifestyle significantly reduces future risk and supports long-term cognitive wellbeing.",
        "accent":       colors.HexColor("#059669"),
        "accent_light": colors.HexColor("#d1fae5"),
        "accent_mid":   colors.HexColor("#6ee7b7"),
        "risk":  "LOW RISK",
        "risk_icon": "✓",
    },
    "VeryMildDemented": {
        "description": "Very early-stage cognitive changes have been detected. Memory lapses may be subtle and not yet impacting daily functioning significantly.",
        "gds": "GDS Stage 3",
        "gds_sub": "Mild Cognitive Impairment",
        "symptoms": "Slight forgetfulness, occasional word-finding difficulties, minor short-term memory lapses.",
        "recommendations": [
            "Consult a neurologist for a comprehensive cognitive assessment (MMSE, MoCA).",
            "Begin memory exercises and structured cognitive training programmes.",
            "Maintain a consistent daily routine to support memory retention.",
            "Discuss early intervention options and medications with your treating physician.",
            "Schedule follow-up MRI scans every 6–12 months to monitor progression.",
        ],
        "prognosis": "Early intervention at this stage can significantly slow progression. Prognosis is most favourable with prompt medical attention and lifestyle modification.",
        "accent":       colors.HexColor("#d97706"),
        "accent_light": colors.HexColor("#fef3c7"),
        "accent_mid":   colors.HexColor("#fcd34d"),
        "risk":  "EARLY STAGE",
        "risk_icon": "⚠",
    },
    "MildDemented": {
        "description": "Mild cognitive decline has been detected. The patient may experience noticeable memory and thinking difficulties affecting day-to-day activities.",
        "gds": "GDS Stage 4",
        "gds_sub": "Mild Dementia",
        "symptoms": "Memory loss affecting daily life, confusion with dates or recent events, difficulty with problem-solving, mood changes.",
        "recommendations": [
            "Seek immediate consultation with a neurologist or geriatric specialist.",
            "Discuss FDA-approved medications: Donepezil, Rivastigmine (cholinesterase inhibitors).",
            "Arrange regular caregiver support and structured daily supervision.",
            "Implement home safety measures — labels, reminders, structured environment.",
            "Explore occupational therapy to maintain daily living skills.",
            "Connect with the Alzheimer's Association for resources and local support groups.",
        ],
        "prognosis": "With appropriate care and medication, daily functioning can be maintained for several years. Caregiver involvement is important for quality of life.",
        "accent":       colors.HexColor("#ea580c"),
        "accent_light": colors.HexColor("#ffedd5"),
        "accent_mid":   colors.HexColor("#fdba74"),
        "risk":  "MILD STAGE",
        "risk_icon": "⚠",
    },
    "ModerateDemented": {
        "description": "Moderate cognitive decline has been detected. Significant assistance with daily activities is required. Specialist review is strongly advised.",
        "gds": "GDS Stage 5–6",
        "gds_sub": "Moderate Dementia",
        "symptoms": "Significant memory loss, disorientation, difficulty recognising familiar people, behavioural changes, and wandering risk.",
        "recommendations": [
            "Immediate specialist consultation is strongly advised.",
            "Full-time caregiver support or memory care facility assessment recommended.",
            "Implement comprehensive home safety modifications — door alarms, GPS tracking.",
            "Explore memantine (Namenda) for moderate-to-severe Alzheimer's management.",
            "Discuss advanced care planning and legal/financial arrangements promptly.",
            "Connect family members with caregiver support networks and respite care services.",
        ],
        "prognosis": "Significant care and supervision are required. The focus shifts to quality of life, comfort, and safety for both the patient and their family.",
        "accent":       colors.HexColor("#dc2626"),
        "accent_light": colors.HexColor("#fee2e2"),
        "accent_mid":   colors.HexColor("#fca5a5"),
        "risk":  "MODERATE STAGE",
        "risk_icon": "!",
    },
    "Unknown / Low Confidence": {
        "description": "The model could not confidently classify this scan. The image quality or scan type may not be optimal for analysis.",
        "gds": "Inconclusive",
        "gds_sub": "Unable to Determine",
        "symptoms": "Unable to determine — please consult a medical professional for interpretation.",
        "recommendations": [
            "Upload a clearer, higher-quality MRI scan for improved results.",
            "Consult a radiologist or neurologist for professional interpretation.",
            "Ensure the scan is a standard brain MRI (axial or coronal plane).",
        ],
        "prognosis": "Inconclusive result. Professional medical evaluation is essential for accurate assessment.",
        "accent":       colors.HexColor("#6b7280"),
        "accent_light": colors.HexColor("#f3f4f6"),
        "accent_mid":   colors.HexColor("#d1d5db"),
        "risk":  "INCONCLUSIVE",
        "risk_icon": "?",
    },
}

ABOUT_ALZHEIMERS = (
    "Alzheimer's disease is a progressive neurological disorder and the most common cause of dementia, "
    "accounting for 60–80% of all dementia cases worldwide. It causes brain cells to degenerate and die, leading to "
    "a continuous decline in thinking, behavioural, and social skills that affects a person's ability to function independently."
    "\n\n"
    "The disease progresses through defined stages — from no impairment through very mild, mild, moderate, and severe decline. "
    "Early detection through neuroimaging (MRI, PET) and cognitive assessments (MMSE, MoCA) allows for earlier intervention, "
    "which can slow progression and significantly improve quality of life."
    "\n\n"
    "Current FDA-approved treatments include cholinesterase inhibitors (Donepezil, Rivastigmine, Galantamine) for mild-to-moderate "
    "stages, and memantine for moderate-to-severe stages. In 2023, Lecanemab (Leqembi) became the first disease-modifying therapy "
    "approved, shown to slow cognitive decline in early Alzheimer's by clearing amyloid plaques from the brain."
    "\n\n"
    "Lifestyle interventions — regular aerobic exercise, cognitive engagement, Mediterranean diet, social activity, and quality "
    "sleep — have demonstrated meaningful benefit in reducing risk and slowing progression."
)


# ── Custom Flowables ───────────────────────────────────────────────────────────

class ColorBar(Flowable):
    """A simple coloured horizontal bar used as a decorative separator."""
    def __init__(self, width, height=3, color=DARK, radius=1.5):
        super().__init__()
        self.bar_width  = width
        self.bar_height = height
        self.color      = color
        self.bar_radius = radius

    def draw(self):
        self.canv.setFillColor(self.color)
        self.canv.roundRect(0, 0, self.bar_width, self.bar_height,
                            self.bar_radius, stroke=0, fill=1)

    def wrap(self, *args):
        return self.bar_width, self.bar_height + 4


class BadgeFlowable(Flowable):
    """Pill-shaped badge."""
    def __init__(self, text, bg_color, text_color=WHITE, font_size=8, h=16):
        super().__init__()
        self.text       = text
        self.bg_color   = bg_color
        self.text_color = text_color
        self.font_size  = font_size
        self.h          = h
        self.w          = len(text) * font_size * 0.65 + 14

    def draw(self):
        c = self.canv
        c.setFillColor(self.bg_color)
        c.roundRect(0, 0, self.w, self.h, self.h / 2, stroke=0, fill=1)
        c.setFillColor(self.text_color)
        c.setFont("Helvetica-Bold", self.font_size)
        c.drawCentredString(self.w / 2, (self.h - self.font_size) / 2 + 1, self.text)

    def wrap(self, *args):
        return self.w, self.h


# ── Page decorators ────────────────────────────────────────────────────────────

def make_page_decorator(accent_color, total_pages_ref):
    """Returns an onPage callback that draws header/footer chrome."""
    def on_page(canv, doc):
        W, H = letter
        margin = 0.75 * inch

        # ── Top accent strip ──
        canv.saveState()
        canv.setFillColor(DARK)
        canv.rect(0, H - 36, W, 36, stroke=0, fill=1)

        # Brand text
        canv.setFillColor(WHITE)
        canv.setFont("Helvetica-Bold", 11)
        canv.drawString(margin, H - 23, "NeuroScan AI")
        canv.setFont("Helvetica", 8)
        canv.setFillColor(colors.HexColor("#94a3b8"))
        canv.drawString(margin + 88, H - 23, "Alzheimer's MRI Analysis Report")

        # Page number (top right)
        canv.setFont("Helvetica", 7.5)
        canv.drawRightString(W - margin, H - 23,
                             f"Page {doc.page}")

        # Accent line under header
        canv.setFillColor(accent_color)
        canv.rect(0, H - 38, W, 2, stroke=0, fill=1)
        canv.restoreState()

        # ── Bottom footer ──
        canv.saveState()
        canv.setFillColor(LIGHT_GRAY)
        canv.rect(0, 0, W, 28, stroke=0, fill=1)
        canv.setFillColor(colors.HexColor("#94a3b8"))
        canv.setFont("Helvetica-Oblique", 6.5)
        canv.drawCentredString(
            W / 2, 10,
            "MEDICAL DISCLAIMER: For educational and informational purposes only. "
            "Not a clinical diagnosis. Consult a qualified neurologist for professional evaluation."
        )
        canv.restoreState()

    return on_page


# ── Main generator ─────────────────────────────────────────────────────────────

def generate_report(prediction: str,
                    confidence: float,
                    all_probs: dict,
                    output_path: str) -> str:

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    stage  = STAGE_INFO.get(prediction, STAGE_INFO["Unknown / Low Confidence"])
    accent = stage["accent"]
    accent_light = stage["accent_light"]
    accent_mid   = stage["accent_mid"]
    now    = datetime.now().strftime("%d %B %Y  ·  %I:%M %p")

    # Inner margins (header + footer consume space)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch + 38,    # below header strip
        bottomMargin=0.5 * inch + 28,  # above footer strip
    )

    on_page = make_page_decorator(accent, {})

    # ── Style definitions ──────────────────────────────────────────────────────
    def S(name, **kw):
        base = dict(fontName="Helvetica", fontSize=9.5, textColor=SLATE,
                    leading=15, spaceAfter=4)
        base.update(kw)
        return ParagraphStyle(name, **base)

    LABEL   = S("lbl",  fontName="Helvetica-Bold", fontSize=7,  textColor=MUTED,
                letterSpacing=0.8, spaceAfter=2)
    H1      = S("h1",   fontName="Helvetica-Bold", fontSize=13, textColor=DARK,
                spaceBefore=18, spaceAfter=6)
    H2      = S("h2",   fontName="Helvetica-Bold", fontSize=10, textColor=DARK,
                spaceBefore=12, spaceAfter=5)
    BODY    = S("body", fontName="Helvetica", fontSize=9.5, textColor=SLATE,
                leading=15, spaceAfter=6, alignment=TA_JUSTIFY)
    BULLET  = S("blt",  fontName="Helvetica", fontSize=9.5, textColor=SLATE,
                leading=15, leftIndent=12, spaceAfter=5)
    CAPTION = S("cap",  fontName="Helvetica-Oblique", fontSize=7.5,
                textColor=MUTED, alignment=TA_CENTER)
    META    = S("meta", fontName="Helvetica", fontSize=8, textColor=MUTED,
                alignment=TA_RIGHT)

    story = []
    W = 7 * inch  # usable width

    # ── 1. Report Meta block ───────────────────────────────────────────────────
    meta_data = [[
        Paragraph(f'<font color="#64748b" size="8">Report Date</font><br/>'
                  f'<b>{now}</b>', S("md", fontSize=8, textColor=SLATE)),
        Paragraph(f'<font color="#64748b" size="8">Reference ID</font><br/>'
                  f'<b>NSA-{datetime.now().strftime("%Y%m%d-%H%M%S")}</b>',
                  S("md", fontSize=8, textColor=SLATE)),
        Paragraph(f'<font color="#64748b" size="8">Model Version</font><br/>'
                  f'<b>v2.1.0-stable</b>', S("md", fontSize=8, textColor=SLATE)),
    ]]
    meta_tbl = Table(meta_data, colWidths=[W/3]*3)
    meta_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,-1), LIGHT_GRAY),
        ("BOX",         (0,0), (-1,-1), 0.5, BORDER),
        ("INNERGRID",   (0,0), (-1,-1), 0.5, BORDER),
        ("PADDING",     (0,0), (-1,-1), 10),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("LINEABOVE",   (0,0), (-1,0),  2, accent),
    ]))
    story.append(meta_tbl)
    story.append(Spacer(1, 16))

    # ── 2. Classification Summary Cards ───────────────────────────────────────
    story.append(Paragraph("CLASSIFICATION SUMMARY", LABEL))
    story.append(Spacer(1, 4))

    # Card 1: Prediction
    accent_hex = "#" + "".join(f"{int(v*255):02x}" for v in [accent.red, accent.green, accent.blue])
    card_pred = Table(
        [[Paragraph("CLASSIFICATION", LABEL)],
         [Paragraph(f'<b>{prediction}</b>',
                    S("cls", fontName="Helvetica-Bold", fontSize=15, textColor=accent, leading=20))]],
        colWidths=[2*inch]
    )
    # Card 2: Confidence
    conf_color = accent if confidence >= 70 else colors.HexColor("#f59e0b")
    card_conf = Table(
        [[Paragraph("CONFIDENCE", LABEL)],
         [Paragraph(f'<font size="20"><b>{confidence:.1f}%</b></font>',
                    S("cnf", fontName="Helvetica-Bold", fontSize=20, textColor=DARK, leading=26))]],
        colWidths=[1.5*inch]
    )
    # Card 3: GDS
    card_gds = Table(
        [[Paragraph("GDS STAGING", LABEL)],
         [Paragraph(f'<b>{stage["gds"]}</b><br/>'
                    f'<font size="8" color="#64748b">{stage["gds_sub"]}</font>',
                    S("gds", fontName="Helvetica-Bold", fontSize=10, textColor=DARK, leading=14))]],
        colWidths=[2*inch]
    )
    # Card 4: Risk badge
    card_risk = Table(
        [[Paragraph("RISK LEVEL", LABEL)],
         [Paragraph(f'<b>{stage["risk"]}</b>',
                    S("rsk", fontName="Helvetica-Bold", fontSize=11, textColor=accent, leading=16))]],
        colWidths=[1.5*inch]
    )

    def card_style(t, bg=WHITE):
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), bg),
            ("PADDING",    (0,0), (-1,-1), 12),
            ("VALIGN",     (0,0), (-1,-1), "TOP"),
        ]))
        return t

    for c in [card_pred, card_conf, card_gds, card_risk]:
        card_style(c)

    # Combine cards in a row
    gap = 0.1 * inch
    cards_row = Table(
        [[card_pred, card_conf, card_gds, card_risk]],
        colWidths=[2.05*inch, 1.55*inch, 2.05*inch, 1.35*inch],
        hAlign="LEFT"
    )
    cards_row.setStyle(TableStyle([
        ("BOX",       (0,0), (0,0), 0.5, BORDER),
        ("BOX",       (1,0), (1,0), 0.5, BORDER),
        ("BOX",       (2,0), (2,0), 0.5, BORDER),
        ("BOX",       (3,0), (3,0), 0.5, BORDER),
        ("LINEABOVE", (0,0), (0,0), 2.5, accent),
        ("LINEABOVE", (1,0), (1,0), 2.5, DARK),
        ("LINEABOVE", (2,0), (2,0), 2.5, MUTED),
        ("LINEABOVE", (3,0), (3,0), 2.5, accent),
        ("LEFTPADDING",  (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 0),
        ("TOPPADDING",   (0,0), (-1,-1), 0),
        ("BOTTOMPADDING",(0,0), (-1,-1), 0),
        ("COLPADDING",   (0,0), (-1,-1), 4),
    ]))
    story.append(cards_row)
    story.append(Spacer(1, 20))

    # ── 3. Probability Distribution ───────────────────────────────────────────
    story.append(ColorBar(W, height=1, color=BORDER))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Class Probability Distribution", H1))

    # Header row
    th_style = S("th", fontName="Helvetica-Bold", fontSize=8, textColor=WHITE)
    prob_rows = [[
        Paragraph("Classification",      th_style),
        Paragraph("Probability",         S("th2", fontName="Helvetica-Bold", fontSize=8, textColor=WHITE, alignment=TA_CENTER)),
        Paragraph("Confidence Bar",      th_style),
    ]]

    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
    for i, (cls, prob) in enumerate(sorted_probs):
        is_top = (cls == prediction)
        filled = int(round(prob / 5))   # out of 20 chars
        empty  = 20 - filled
        bar_str = ("█" * filled + "░" * empty)

        name_para = Paragraph(
            f'<b>{"▶  " if is_top else "     "}{cls}</b>' if is_top else f'     {cls}',
            S("td", fontName="Helvetica-Bold" if is_top else "Helvetica",
              fontSize=9, textColor=accent if is_top else SLATE)
        )
        pct_para = Paragraph(
            f'<b>{prob:.1f}%</b>' if is_top else f'{prob:.1f}%',
            S("tp2", fontName="Helvetica-Bold" if is_top else "Helvetica",
              fontSize=9, textColor=accent if is_top else SLATE, alignment=TA_CENTER)
        )
        bar_para = Paragraph(
            f'<font face="Courier" size="8.5" color="{"#059669" if is_top else "#cbd5e1"}">{bar_str}</font>',
            S("bp")
        )
        prob_rows.append([name_para, pct_para, bar_para])

    prob_tbl = Table(prob_rows, colWidths=[2.4*inch, 0.9*inch, 3.7*inch])
    row_styles = [
        ("BACKGROUND",  (0,0), (-1,0),  DARK_MID),
        ("BOX",         (0,0), (-1,-1), 0.5, BORDER),
        ("INNERGRID",   (0,0), (-1,-1), 0.3, BORDER),
        ("PADDING",     (0,0), (-1,-1), 9),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("LINEBELOW",   (0,0), (-1,0),  1.5, accent),
        # Highlight top row
        ("BACKGROUND",  (0,1), (-1,1),  accent_light),
        ("LINEABOVE",   (0,1), (-1,1),  1, accent),
        ("LINEBELOW",   (0,1), (-1,1),  1, accent),
    ]
    # Alternating rows for non-top
    for r in range(2, len(prob_rows)):
        bg = WHITE if r % 2 == 0 else LIGHT_GRAY
        row_styles.append(("BACKGROUND", (0,r), (-1,r), bg))

    prob_tbl.setStyle(TableStyle(row_styles))
    story.append(prob_tbl)
    story.append(Spacer(1, 20))

    # ── 4. Clinical Assessment ────────────────────────────────────────────────
    story.append(ColorBar(W, height=1, color=BORDER))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Clinical Assessment", H1))

    # Description box
    desc_tbl = Table(
        [[Paragraph(stage["description"], BODY)]],
        colWidths=[W]
    )
    desc_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), LIGHT_GRAY),
        ("BOX",        (0,0), (-1,-1), 0.5, BORDER),
        ("LINEBEFORE", (0,0), (0,-1), 3,   accent),
        ("PADDING",    (0,0), (-1,-1), 12),
    ]))
    story.append(desc_tbl)
    story.append(Spacer(1, 14))

    # Two-column: Symptoms | Prognosis
    sym_content = [
        [Paragraph("OBSERVED SYMPTOMS", LABEL)],
        [Spacer(1, 4)],
        [Paragraph(stage["symptoms"], BODY)],
    ]
    sym_tbl = Table(sym_content, colWidths=[3.35*inch])
    sym_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,-1), WHITE),
        ("BOX",         (0,0), (-1,-1), 0.5, BORDER),
        ("LINEABOVE",   (0,0), (-1,0),  2.5, accent),
        ("PADDING",     (0,0), (-1,-1), 11),
        ("VALIGN",      (0,0), (-1,-1), "TOP"),
    ]))

    prog_content = [
        [Paragraph("PROGNOSIS", LABEL)],
        [Spacer(1, 4)],
        [Paragraph(stage["prognosis"], BODY)],
    ]
    prog_tbl = Table(prog_content, colWidths=[3.35*inch])
    prog_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,-1), accent_light),
        ("BOX",         (0,0), (-1,-1), 0.5, BORDER),
        ("LINEABOVE",   (0,0), (-1,0),  2.5, accent),
        ("PADDING",     (0,0), (-1,-1), 11),
        ("VALIGN",      (0,0), (-1,-1), "TOP"),
    ]))

    two_col = Table([[sym_tbl, prog_tbl]], colWidths=[3.35*inch, 3.35*inch],
                    hAlign="LEFT")
    two_col.setStyle(TableStyle([
        ("LEFTPADDING",   (0,0), (-1,-1), 0),
        ("RIGHTPADDING",  (0,0), (-1,-1), 0),
        ("TOPPADDING",    (0,0), (-1,-1), 0),
        ("BOTTOMPADDING", (0,0), (-1,-1), 0),
        ("COLPADDING",    (1,0), (1,0),  10),
    ]))
    story.append(two_col)
    story.append(Spacer(1, 18))

    # ── 5. Recommendations ────────────────────────────────────────────────────
    story.append(Paragraph("Clinical Recommendations", H1))

    rec_rows = [[
        Paragraph("#", S("rh", fontName="Helvetica-Bold", fontSize=8, textColor=WHITE, alignment=TA_CENTER)),
        Paragraph("Recommendation", S("rh2", fontName="Helvetica-Bold", fontSize=8, textColor=WHITE)),
    ]]
    for i, rec in enumerate(stage["recommendations"], 1):
        num_para = Paragraph(str(i), S("rn", fontName="Helvetica-Bold", fontSize=9,
                                        textColor=accent, alignment=TA_CENTER))
        rec_para = Paragraph(rec, S("rv", fontSize=9.5, textColor=SLATE, leading=14))
        rec_rows.append([num_para, rec_para])

    rec_tbl = Table(rec_rows, colWidths=[0.45*inch, 6.55*inch])
    rec_row_styles = [
        ("BACKGROUND", (0,0), (-1,0),  DARK_MID),
        ("BOX",        (0,0), (-1,-1), 0.5, BORDER),
        ("INNERGRID",  (0,0), (-1,-1), 0.3, BORDER),
        ("PADDING",    (0,0), (-1,-1), 9),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("LINEBELOW",  (0,0), (-1,0),  1.5, accent),
    ]
    for r in range(1, len(rec_rows)):
        bg = WHITE if r % 2 != 0 else LIGHT_GRAY
        rec_row_styles.append(("BACKGROUND", (0,r), (-1,r), bg))
        rec_row_styles.append(("LINEBEFORE", (0,r), (0,r), 0, WHITE))  # no inner left
    rec_tbl.setStyle(TableStyle(rec_row_styles))
    story.append(rec_tbl)
    story.append(Spacer(1, 20))

    # ── 6. About Alzheimer's ──────────────────────────────────────────────────
    story.append(ColorBar(W, height=1, color=BORDER))
    story.append(Spacer(1, 10))
    story.append(Paragraph("About Alzheimer's Disease", H1))

    for para_text in ABOUT_ALZHEIMERS.split("\n\n"):
        story.append(Paragraph(para_text.strip(), BODY))
        story.append(Spacer(1, 4))

    story.append(Spacer(1, 12))

    # ── 7. Extended Disclaimer ────────────────────────────────────────────────
    disc_box = Table(
        [[Paragraph(
            "<b>MEDICAL DISCLAIMER</b>  This report is generated by an AI model for educational and "
            "informational purposes only. It is based on a single 2D MRI slice and does not constitute "
            "a clinical diagnosis. Results may not be accurate. Always consult a qualified neurologist "
            "or medical professional for proper diagnosis, interpretation, and treatment planning.",
            S("disc", fontName="Helvetica", fontSize=8, textColor=MUTED,
              leading=12, alignment=TA_JUSTIFY)
        )]],
        colWidths=[W]
    )
    disc_box.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), LIGHT_GRAY),
        ("BOX",        (0,0), (-1,-1), 0.5, BORDER),
        ("LINEBEFORE", (0,0), (0,-1), 3,   MUTED),
        ("PADDING",    (0,0), (-1,-1), 10),
    ]))
    story.append(disc_box)

    # ── Build ─────────────────────────────────────────────────────────────────
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return output_path


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_probs = {
        "NonDemented":       5.1,
        "VeryMildDemented": 12.4,
        "MildDemented":     68.3,
        "ModerateDemented": 14.2,
    }
    out = generate_report(
        prediction="MildDemented",
        confidence=68.3,
        all_probs=test_probs,
        output_path="/mnt/user-data/outputs/neuroscan_report.pdf"
    )
    print(f"Report written to: {out}")