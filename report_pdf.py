"""
report_pdf.py -- render a one-page clinical-style PDF from the agent's output:
the BUS image, the benign/malignant decision with confidence, BI-RADS estimate,
structured findings, the generated report text, and a confidence/caveat block
(including a review flag when evidence is uncertain or the case looks out-of-
distribution). Research demo output -- not a diagnostic device.
"""
from datetime import datetime
from fpdf import FPDF

_REPL = {"→": "->", "←": "<-", "⚠": "!", "•": "-",
         "’": "'", "‘": "'", "“": '"', "”": '"',
         "–": "-", "—": "-", "≥": ">=", "≤": "<=",
         "●": "*", "…": "..."}


def _s(t):
    """Make text safe for the latin-1 core PDF font."""
    t = str(t)
    for k, v in _REPL.items():
        t = t.replace(k, v)
    return t.encode("latin-1", "replace").decode("latin-1")

NAVY = (18, 33, 58)
INK = (30, 41, 59)
MUT = (110, 122, 145)
RED = (200, 60, 60)
GREEN = (30, 140, 90)
AMBER = (200, 140, 20)
LINE = (210, 216, 228)


def _risk_color(decision, p):
    return RED if decision == "malignant" else GREEN


def render_report(out_path, *, case_id, image_path, diagnosis, findings=None,
                  report_text="", bi_rads=None, rationale="", review_flag=False,
                  review_reason="", heatmap_path=None, possible_findings=None):
    # sanitise all dynamic text for the latin-1 core font
    case_id = _s(case_id); report_text = _s(report_text)
    rationale = _s(rationale); review_reason = _s(review_reason)
    bi_rads = _s(bi_rads) if bi_rads else bi_rads
    findings = [(_s(t), p) for t, p in (findings or [])]
    possible_findings = [(_s(t), p) for t, p in (possible_findings or [])]

    pdf = FPDF(format="A4", unit="mm")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    W = pdf.w - 2 * pdf.l_margin

    # header
    pdf.set_fill_color(*NAVY); pdf.rect(0, 0, pdf.w, 22, "F")
    pdf.set_xy(pdf.l_margin, 6); pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 6, "Breast Ultrasound - AI Diagnostic Report", ln=1)
    pdf.set_x(pdf.l_margin); pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, "Agentic multimodal pipeline  -  research demo, not for clinical use", ln=1)
    pdf.ln(8)

    # case line
    pdf.set_text_color(*MUT); pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, f"Case: {case_id}      Generated: {datetime.now():%Y-%m-%d %H:%M}", ln=1)
    pdf.ln(2)

    top_y = pdf.get_y()
    # image (left) -- size from the real aspect ratio, capped in height
    img_w, img_h = 78, 60
    try:
        from PIL import Image as _PILImage
        iw, ih = _PILImage.open(image_path).size
        img_h = img_w * ih / iw
        if img_h > 72:                       # keep tall images from dominating
            img_h = 72; img_w = img_h * iw / ih
        pdf.image(image_path, x=pdf.l_margin, y=top_y, w=img_w, h=img_h)
    except Exception:
        img_h = 60

    # Grad-CAM attention overlay below the main image
    if heatmap_path:
        hy = top_y + img_h + 3
        hw = 52
        try:
            pdf.image(heatmap_path, x=pdf.l_margin, y=hy, w=hw, h=hw)
            pdf.set_xy(pdf.l_margin, hy + hw + 0.5)
            pdf.set_text_color(*MUT); pdf.set_font("Helvetica", "I", 7.5)
            pdf.cell(hw, 4, "Model attention (Grad-CAM)", align="C")
            img_h = img_h + 3 + hw + 5             # extend left-column height
        except Exception:
            pass

    # decision panel (right)
    rx = pdf.l_margin + img_w + 8
    rw = W - img_w - 8
    d = diagnosis
    col = _risk_color(d["decision"], d["p_malignant"])
    pdf.set_xy(rx, top_y)
    pdf.set_text_color(*MUT); pdf.set_font("Helvetica", "B", 9)
    pdf.cell(rw, 5, "PREDICTION", ln=2)
    pdf.set_x(rx); pdf.set_text_color(*col); pdf.set_font("Helvetica", "B", 22)
    pdf.cell(rw, 11, d["decision"].upper(), ln=2)
    pdf.set_x(rx); pdf.set_text_color(*INK); pdf.set_font("Helvetica", "", 10)
    pdf.cell(rw, 6, f"P(malignant) = {d['p_malignant']:.2f}   "
                    f"confidence {d['confidence']*100:.0f}%", ln=2)
    if bi_rads:
        pdf.set_x(rx); pdf.cell(rw, 6, f"BI-RADS: {bi_rads}", ln=2)
    mode = "image + descriptors" if d.get("used_descriptors") else "image only"
    pdf.set_x(rx); pdf.set_text_color(*MUT); pdf.set_font("Helvetica", "", 8.5)
    pdf.cell(rw, 5, f"evidence: {mode}", ln=2)
    # per-stream bars
    pdf.ln(1)
    for name, val in d.get("streams", {}).items():
        pdf.set_x(rx); pdf.set_text_color(*MUT); pdf.set_font("Helvetica", "", 8)
        pdf.cell(26, 4.5, name)
        bx, by = pdf.get_x(), pdf.get_y() + 1.2
        bw = rw - 40
        pdf.set_fill_color(*LINE); pdf.rect(bx, by, bw, 2.4, "F")
        pdf.set_fill_color(*col); pdf.rect(bx, by, bw * float(val), 2.4, "F")
        pdf.set_xy(bx + bw + 2, pdf.get_y()); pdf.cell(10, 4.5, f"{val:.2f}", ln=2)

    pdf.set_y(max(pdf.get_y(), top_y + img_h) + 6)

    def section(title):
        pdf.set_draw_color(*LINE); pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
        pdf.ln(2); pdf.set_text_color(*NAVY); pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, title, ln=1)

    # review flag
    if review_flag:
        pdf.set_fill_color(255, 244, 224); pdf.set_draw_color(*AMBER)
        y0 = pdf.get_y()
        pdf.set_xy(pdf.l_margin, y0)
        pdf.set_text_color(*AMBER); pdf.set_font("Helvetica", "B", 10)
        pdf.multi_cell(W, 6, f"  Flagged for radiologist review - {review_reason}",
                       border=1, fill=True)
        pdf.ln(3)

    # findings -- asserted (classifier, high precision) then possible (retrieval)
    if findings or possible_findings:
        section("Predicted findings (interpretation)")
        pdf.set_text_color(*INK); pdf.set_font("Helvetica", "", 10)
        for term, prob in findings:
            pdf.cell(0, 5.5, f"  -  {term}  ({prob:.2f})", ln=1)
        if not findings:
            pdf.set_text_color(*MUT)
            pdf.cell(0, 5.5, "  (no findings asserted with high confidence)", ln=1)
        if possible_findings:
            pdf.ln(1); pdf.set_text_color(*MUT); pdf.set_font("Helvetica", "I", 9)
            pdf.cell(0, 5, "Possible (seen in similar cases):", ln=1)
            pdf.set_font("Helvetica", "", 9.5)
            for term, sim in possible_findings:
                pdf.cell(0, 5, f"  ~  {term}  (similarity {sim:.2f})", ln=1)
        pdf.ln(1)

    # report text
    if report_text:
        section("Generated report")
        pdf.set_text_color(*INK); pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(W, 5.5, report_text)
        pdf.ln(1)

    # rationale
    if rationale:
        section("Decision rationale")
        pdf.set_text_color(*MUT); pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(W, 5, rationale)

    # footer
    pdf.set_y(-15); pdf.set_text_color(*MUT); pdf.set_font("Helvetica", "I", 7.5)
    pdf.multi_cell(0, 4, "Generated by an agentic research pipeline (BiomedCLIP retrieval + "
                         "ViT malignancy model + descriptor fusion + LLM report refinement). "
                         "For research only; not a medical device and not for diagnostic use.")
    pdf.output(out_path)
    return out_path
