#!/usr/bin/env python3
"""
insert_figures.py
Automatically inserts the generated chart images into the IEEE paper docx,
replacing every [IMAGE PLACEHOLDER] block with the correct figure.

Run in WSL (with venv active):
    python3 insert_figures.py

Source : ~/invigilai/Multi_Modal_Cheating_Detection_IEEE_Paper.docx
         OR /mnt/c/Users/kenne/Downloads/Multi_Modal_Cheating_Detection_IEEE_Paper.docx
Output : same folder, file named ..._WITH_FIGURES.docx
"""

import sys, re
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from lxml import etree

# ── Locate source docx ────────────────────────────────────────────────────────
CANDIDATES = [
    Path.home() / "invigilai" / "Multi_Modal_Cheating_Detection_IEEE_Paper.docx",
    Path("/mnt/c/Users/kenne/Downloads/Multi_Modal_Cheating_Detection_IEEE_Paper.docx"),
    Path("/mnt/c/Users/kenne/Downloads/Multi_Modal_Cheating_Detection_IEEE_Paper_FINAL.docx"),
    Path.home() / "Downloads" / "Multi_Modal_Cheating_Detection_IEEE_Paper.docx",
    Path.home() / "Downloads" / "Multi_Modal_Cheating_Detection_IEEE_Paper_FINAL.docx",
]

SRC = None
for c in CANDIDATES:
    if c.exists():
        SRC = c
        break

if SRC is None:
    print("ERROR: Could not find the docx file.")
    print("Copy it to ~/invigilai/ and re-run:")
    print("  cp /mnt/c/Users/kenne/Downloads/Multi_Modal_Cheating_Detection_IEEE_Paper.docx ~/invigilai/")
    sys.exit(1)

DEST = SRC.parent / (SRC.stem + "_WITH_FIGURES.docx")

# ── Chart images folder ───────────────────────────────────────────────────────
CHARTS_DIR = Path(__file__).parent / "paper_charts"

# ── Figure → image mapping ────────────────────────────────────────────────────
# Keys: substrings to search for in the placeholder caption text (case-insensitive)
# Values: (image filename, display width in inches)
FIGURE_MAP = [
    ("fig. 1",          "paper_fig1_efficientnet_architecture.png", 6.5),
    ("efficientnet",    "paper_fig1_efficientnet_architecture.png", 6.5),
    ("mbconv",          "paper_fig1_efficientnet_architecture.png", 6.5),
    ("fig. 2",          "paper_fig2_training_accuracy.png",         5.5),
    ("training.*accur", "paper_fig2_training_accuracy.png",         5.5),
    ("fig. 3",          "paper_fig3_training_loss.png",             5.5),
    ("training.*loss",  "paper_fig3_training_loss.png",             5.5),
    ("fig. 4",          "paper_fig4_confusion_matrix.png",          4.5),
    ("confusion",       "paper_fig4_confusion_matrix.png",          4.5),
    ("fig. 5",          "paper_fig5_pr_curve.png",                  5.0),
    ("precision.recall","paper_fig5_pr_curve.png",                  5.0),
    ("fig. 6",          "paper_fig6_roc_curve.png",                 5.0),
    ("roc",             "paper_fig6_roc_curve.png",                 5.0),
    ("fig. 7",          "paper_fig7_per_modality_f1.png",           6.0),
    ("per.modality",    "paper_fig7_per_modality_f1.png",           6.0),
    ("modality.*f1",    "paper_fig7_per_modality_f1.png",           6.0),
    ("ablation",        "paper_charts/fig7_ablation_study.png",     5.5),
]


def match_figure(text: str):
    """Return (image_path, width_inches) for the best matching figure, or None."""
    t = text.lower()
    for pattern, fname, width in FIGURE_MAP:
        if re.search(pattern, t):
            p = CHARTS_DIR / fname
            if not p.exists():
                # try relative path in case fname has subdir prefix
                p = Path(__file__).parent / fname
            if p.exists():
                return p, width
    return None


def collect_placeholder_blocks(doc):
    """
    Returns list of (paragraph_index, window_text, para_objects_in_block).
    A 'block' is the [IMAGE PLACEHOLDER] para plus the next few caption paras.
    """
    paras = doc.paragraphs
    blocks = []
    i = 0
    while i < len(paras):
        txt = paras[i].text
        if "[image placeholder]" in txt.lower():
            # collect this para + up to 4 following paras as the caption window
            window = paras[i: i+5]
            window_text = " ".join(p.text for p in window)
            blocks.append((i, window_text, window))
            i += 1
        else:
            i += 1
    return blocks


def replace_paragraph_with_image(doc, para, img_path, width_in):
    """
    Replace the content of `para` with a centred inline image.
    The paragraph element is modified in-place.
    """
    # Clear existing runs
    for run in para.runs:
        run.text = ""
    # Clear all child XML (removes leftover text nodes)
    p_elem = para._element
    for child in list(p_elem):
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag in ("r", "hyperlink", "bookmarkStart", "bookmarkEnd", "proofErr"):
            p_elem.remove(child)

    # Set paragraph alignment to centre
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Add image run
    run = para.add_run()
    run.add_picture(str(img_path), width=Inches(width_in))


def process_table_cells(doc):
    """Also check tables for IMAGE PLACEHOLDER content."""
    replaced = 0
    for tbl in doc.tables:
        for row in tbl.rows:
            for cell in row.cells:
                paras = cell.paragraphs
                for i, para in enumerate(paras):
                    if "[image placeholder]" in para.text.lower():
                        # build window from surrounding cell paragraphs
                        window_text = " ".join(p.text for p in paras[i:i+5])
                        result = match_figure(window_text)
                        if result:
                            img_path, width = result
                            replace_paragraph_with_image(doc, para, img_path, width)
                            # blank out the caption sub-paras inside the block
                            for cp in paras[i+1:i+4]:
                                if cp.text.strip().lower().startswith(("fig.", "replace")):
                                    for run in cp.runs:
                                        run.text = ""
                            print(f"    [table] Inserted {img_path.name}")
                            replaced += 1
    return replaced


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Source  : {SRC}")
    print(f"Output  : {DEST}")
    print(f"Charts  : {CHARTS_DIR}")
    print()

    if not CHARTS_DIR.exists():
        print(f"ERROR: Charts folder not found: {CHARTS_DIR}")
        print("Run: python3 generate_paper_figures.py")
        sys.exit(1)

    doc = Document(str(SRC))

    # ── Replace in main body paragraphs ──────────────────────────────────────
    blocks = collect_placeholder_blocks(doc)
    print(f"Found {len(blocks)} IMAGE PLACEHOLDER(s) in main body.")

    body_replaced = 0
    for idx, window_text, window_paras in blocks:
        result = match_figure(window_text)
        if result is None:
            print(f"  [!] Could not match figure for: {window_text[:80]!r}")
            continue
        img_path, width = result
        main_para = window_paras[0]
        replace_paragraph_with_image(doc, main_para, img_path, width)
        # Blank caption sub-lines that say "Replace with actual image"
        for cp in window_paras[1:4]:
            if re.search(r"replace with|fig\.\s*\d", cp.text, re.I):
                for run in cp.runs:
                    run.text = ""
        print(f"  [body] Inserted {img_path.name}  (matched: {window_text[:60]!r})")
        body_replaced += 1

    # ── Replace in table cells ────────────────────────────────────────────────
    table_replaced = process_table_cells(doc)

    total = body_replaced + table_replaced
    print(f"\nTotal replaced: {total} figure(s).")

    if total == 0:
        print("\nWARNING: No placeholders were replaced.")
        print("This can happen if the placeholders are inside text boxes (shapes).")
        print("In that case, insert images manually:")
        print("  - Click the IMAGE PLACEHOLDER box")
        print("  - Insert tab --> Pictures --> This Device")
        print("  - Select from:", CHARTS_DIR)
    else:
        doc.save(str(DEST))
        print(f"\nSaved: {DEST}")
        # Also copy to Windows Downloads
        win_dest = Path("/mnt/c/Users/kenne/Downloads") / DEST.name
        try:
            import shutil
            shutil.copy(str(DEST), str(win_dest))
            print(f"Copied to: {win_dest}")
        except Exception as e:
            print(f"(Could not copy to Downloads: {e})")


if __name__ == "__main__":
    main()
