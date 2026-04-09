#!/usr/bin/env python3
"""
insert_figures.py  v3
Finds every [IMAGE PLACEHOLDER] in the docx — including inside Word text boxes
(shapes) — and replaces it with the correct figure at a print-ready size.

Run in WSL (venv active):
    cp /mnt/c/Users/kenne/Downloads/Multi_Modal_Cheating_Detection_IEEE_Paper.docx ~/invigilai/
    python3 insert_figures.py

Output: Multi_Modal_Cheating_Detection_IEEE_Paper_WITH_FIGURES.docx
        (also copied to /mnt/c/Users/kenne/Downloads/)
"""

import re, sys, copy, shutil
from pathlib import Path
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from lxml import etree

# ── Locate source docx ────────────────────────────────────────────────────────
CANDIDATES = [
    Path.home() / "invigilai" / "Multi_Modal_Cheating_Detection_IEEE_Paper_FINAL.docx",
    Path.home() / "invigilai" / "Multi_Modal_Cheating_Detection_IEEE_Paper.docx",
    Path("/mnt/c/Users/kenne/Downloads/Multi_Modal_Cheating_Detection_IEEE_Paper_FINAL.docx"),
    Path("/mnt/c/Users/kenne/Downloads/Multi_Modal_Cheating_Detection_IEEE_Paper.docx"),
]
SRC = next((c for c in CANDIDATES if c.exists()), None)
if SRC is None:
    print("ERROR: docx not found. Copy it first:")
    print("  cp /mnt/c/Users/kenne/Downloads/Multi_Modal_...docx ~/invigilai/")
    sys.exit(1)

DEST      = SRC.parent / (SRC.stem + "_WITH_FIGURES.docx")
CHARTS    = Path(__file__).parent / "paper_charts"
WIN_DEST  = Path("/mnt/c/Users/kenne/Downloads") / DEST.name

# ── Figure map: (caption-pattern, image-file, width-inches) ──────────────────
# Width guide (IEEE two-column):
#   single column ≈ 3.3"   |   double column ≈ 6.8"
FIGURE_MAP = [
    # ── Architecture (full-width, double-column) ──────────────────────────────
    (r"fig\.?\s*1\b|efficientnet|mbconv",
        "paper_fig1_efficientnet_architecture.png", 6.3),
    # ── Training curves ───────────────────────────────────────────────────────
    (r"fig\.?\s*2\b|training.*accur",
        "paper_fig2_training_accuracy.png",         3.2),
    (r"fig\.?\s*3\b|training.*loss",
        "paper_fig3_training_loss.png",             3.2),
    # ── Confusion matrix ──────────────────────────────────────────────────────
    (r"fig\.?\s*4\b|confusion.matrix",
        "paper_fig4_confusion_matrix.png",          3.0),
    # ── PR / ROC ──────────────────────────────────────────────────────────────
    (r"fig\.?\s*5\b|precision.recall curve",
        "paper_fig5_pr_curve.png",                  3.0),
    (r"fig\.?\s*6\b|roc curve",
        "paper_fig6_roc_curve.png",                 3.0),
    # ── Ablation / per-modality ───────────────────────────────────────────────
    (r"fig\.?\s*7\b|per.modality|modality.*f1|ablation",
        "paper_fig7_per_modality_f1.png",           3.2),
    # ── NORMAL Grad-CAM / SHAP ────────────────────────────────────────────────
    (r"fig\.?\s*8\b|normal case 1|shap summary.*right",
        "paper_fig8_normal_case1.png",              3.2),
    (r"fig\.?\s*9\b|normal case 2|diffuse activation",
        "paper_fig9_normal_case2.png",              3.2),
    (r"fig\.?\s*10\b|normal case 3|eye and mouth|tab switching",
        "paper_fig10_normal_case3.png",             3.2),
    (r"fig\.?\s*11\b|normal case 4|keystroke dynamics",
        "paper_fig11_normal_case4.png",             3.2),
    # ── SUSPICIOUS Grad-CAM / SHAP ────────────────────────────────────────────
    (r"fig\.?\s*12\b|suspicious case 1|gaze deviation",
        "paper_fig12_suspicious_gaze.png",          3.2),
    (r"fig\.?\s*13\b|suspicious case 2|audio anomaly",
        "paper_fig13_suspicious_audio.png",         3.2),
    (r"fig\.?\s*14\b|suspicious case 3|keystroke anomaly|copy.paste",
        "paper_fig14_suspicious_keystroke.png",     3.2),
    # ── Dashboard / UI figures ────────────────────────────────────────────────
    (r"fig\.?\s*15\b|monitoring dashboard|active sessions grid",
        "paper_fig15_dashboard.png",               6.3),
    (r"fig\.?\s*16\b|risk score heatmap|temporal heatmap",
        "paper_fig16_risk_heatmap.png",            6.3),
    (r"fig\.?\s*17\b|alert panel|flagged sessions|alert feed",
        "paper_fig17_alert_panel.png",             6.3),
    (r"fig\.?\s*18\b|session replay|evidence timeline",
        "paper_fig18_session_replay.png",          6.3),
]


def match_figure(text: str):
    t = text.lower()
    for pattern, fname, width in FIGURE_MAP:
        if re.search(pattern, t):
            p = CHARTS / fname
            if p.exists():
                return p, width
    return None


def _get_para_text(p_elem):
    return "".join((t.text or "") for t in p_elem.iter(qn("w:t")))


def _add_image_paragraph(doc, img_path, width_in):
    """
    Append a centred image paragraph at the END of the doc body,
    return its lxml element (caller will move it to the right place).
    """
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(img_path), width=Inches(width_in))
    elem = p._element
    # Detach from body so we can re-attach elsewhere
    doc.element.body.remove(elem)
    return elem


def _replace_elem(old_elem, new_elem):
    """Replace old_elem with new_elem in its parent."""
    parent = old_elem.getparent()
    idx    = list(parent).index(old_elem)
    parent.remove(old_elem)
    parent.insert(idx, new_elem)


def _blank_caption_siblings(p_elem, n=5):
    """
    After inserting an image, clean up the following paragraphs:
    - Delete 'Replace with actual image' lines entirely
    - Delete duplicate captions (same text as the one right after the image)
    - Keep only ONE clean Fig. N caption
    """
    parent   = p_elem.getparent()
    siblings = list(parent)
    try:
        start = siblings.index(p_elem) + 1
    except ValueError:
        return

    first_caption_seen = False
    for sib in siblings[start: start + n]:
        tag = sib.tag.split("}")[-1] if "}" in sib.tag else sib.tag
        if tag != "p":
            break
        txt = _get_para_text(sib).strip()
        low = txt.lower()

        # Always delete "Replace with actual image"
        if low.startswith("replace with"):
            for t in sib.iter(qn("w:t")):
                t.text = ""
            continue

        # Keep first Fig. caption, delete any duplicate of it
        if re.match(r"fig\.?\s*\d", low):
            if first_caption_seen:
                # duplicate — delete it
                for t in sib.iter(qn("w:t")):
                    t.text = ""
            else:
                first_caption_seen = True


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Source  : {SRC.name}")
    print(f"Output  : {DEST.name}")
    print(f"Charts  : {CHARTS}/")
    print()

    if not CHARTS.exists():
        print("ERROR: paper_charts/ not found — run generate_paper_figures.py first.")
        sys.exit(1)

    doc = Document(str(SRC))

    # ── Collect ALL placeholder paragraph elements (incl. text boxes) ─────────
    placeholders = []
    for p_elem in doc.element.body.iter(qn("w:p")):
        txt = _get_para_text(p_elem)
        if "[image placeholder]" in txt.lower():
            # Build context: this para + next 4 siblings
            parent   = p_elem.getparent()
            siblings = list(parent)
            try:
                idx = siblings.index(p_elem)
            except ValueError:
                idx = 0
            window = siblings[idx: idx + 5]
            context = " ".join(_get_para_text(s) for s in window
                               if s.tag == qn("w:p"))
            placeholders.append((p_elem, context))

    print(f"Found {len(placeholders)} [IMAGE PLACEHOLDER](s) in document.\n")

    replaced = 0
    for p_elem, context in placeholders:
        result = match_figure(context)
        if result is None:
            print(f"  [!] No match for: {context[:80]!r}")
            continue

        img_path, width = result
        img_elem = _add_image_paragraph(doc, img_path, width)
        _replace_elem(p_elem, img_elem)
        _blank_caption_siblings(img_elem)
        print(f"  [OK] {img_path.name}  ({width}\")")
        replaced += 1

    print(f"\nReplaced: {replaced} / {len(placeholders)} placeholder(s).")

    if replaced == 0:
        print("\nWARNING: Nothing was replaced.")
        print("The placeholders may use a non-standard character — try running:")
        print("  python3 insert_figures.py --debug")
        print("to see the raw text of each paragraph in the document.")
        sys.exit(1)

    doc.save(str(DEST))
    print(f"\nSaved: {DEST}")

    try:
        WIN_DEST.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(DEST), str(WIN_DEST))
        print(f"Copied to: {WIN_DEST}")
    except Exception as e:
        print(f"(Windows copy failed: {e})")

    print("\nDone.")


# ── Debug mode ────────────────────────────────────────────────────────────────
def debug():
    """Print every paragraph in the doc that might be a placeholder."""
    doc = Document(str(SRC))
    print(f"Scanning {SRC.name} ...\n")
    for i, p_elem in enumerate(doc.element.body.iter(qn("w:p"))):
        txt = _get_para_text(p_elem).strip()
        if txt and ("placeholder" in txt.lower() or "image" in txt.lower()
                    or "fig." in txt.lower() or "replace" in txt.lower()):
            print(f"  [{i:04d}] {txt[:120]!r}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--debug":
        debug()
    else:
        main()
