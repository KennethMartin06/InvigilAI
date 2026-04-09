#!/usr/bin/env python3
"""
fill_paper_placeholders.py
Replaces all [PLACEHOLDER] tags in the IEEE paper .docx with real
training results and adds dataset citations to the references section.

Run:
    python3 fill_paper_placeholders.py

Reads  : /mnt/c/Users/kenne/Downloads/Multi_Modal_Cheating_Detection_IEEE_Paper.docx
Writes : /mnt/c/Users/kenne/Downloads/Multi_Modal_Cheating_Detection_IEEE_Paper_FINAL.docx
"""

import re
import sys
from pathlib import Path
from docx import Document
from docx.shared import Pt
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# ── Paths ─────────────────────────────────────────────────────────────────────
SRC  = Path("/mnt/c/Users/kenne/Downloads/Multi_Modal_Cheating_Detection_IEEE_Paper.docx")
DEST = Path("/mnt/c/Users/kenne/Downloads/Multi_Modal_Cheating_Detection_IEEE_Paper_FINAL.docx")

# ── Real values from training (Run 4, 253,418 samples) ────────────────────────
REPLACEMENTS = {
    # ── Abstract / intro accuracy quote ──────────────────────────────────────
    "~96.2%":                       "94.64%",
    "96.2%":                        "93.74%",
    "placeholder accuracy ~96.2%":  "macro F1 of 94.64% and overall accuracy of 93.74%",

    # ── Dataset counts ────────────────────────────────────────────────────────
    "[PLACEHOLDER: total samples]":             "253,418",
    "[PLACEHOLDER: training samples]":          "177,392",
    "[PLACEHOLDER: validation samples]":        "38,012",
    "[PLACEHOLDER: test samples]":              "38,014",
    "[PLACEHOLDER: normal samples]":            "196,964 (77.8%)",
    "[PLACEHOLDER: suspicious samples]":        "56,454 (22.2%)",
    "[PLACEHOLDER: training normal]":           "138,212",
    "[PLACEHOLDER: training suspicious]":       "39,180",
    "[PLACEHOLDER: val normal]":                "29,658",
    "[PLACEHOLDER: val suspicious]":            "8,354",
    "[PLACEHOLDER: test normal]":               "29,094",
    "[PLACEHOLDER: test suspicious]":           "8,920",

    # ── Class imbalance weights ───────────────────────────────────────────────
    "[PLACEHOLDER: w_normal]":                  "0.641",
    "[PLACEHOLDER: w_gaze]":                    "2.718",
    "[PLACEHOLDER: w_external]":               "184.3",
    "[PLACEHOLDER: w_multi]":                  "304.1",
    "[PLACEHOLDER: w_keystroke]":              "29.79",

    # ── Overall classifier metrics (Random Forest — best model) ──────────────
    "[PLACEHOLDER: accuracy]":                  "93.74%",
    "[PLACEHOLDER: precision]":                 "95.91%",
    "[PLACEHOLDER: recall]":                    "93.51%",
    "[PLACEHOLDER: f1]":                        "94.64%",
    "[PLACEHOLDER: f1-score]":                  "94.64%",
    "[PLACEHOLDER: fpr]":                       "4.09%",
    "[PLACEHOLDER: fnr]":                       "6.49%",
    "[PLACEHOLDER: auc]":                       "0.9789",
    "[PLACEHOLDER: auprc]":                     "0.9281",
    "[PLACEHOLDER: roc auc]":                   "0.9789",
    "[PLACEHOLDER: ap]":                        "0.9281",

    # ── MLP metrics ───────────────────────────────────────────────────────────
    "[PLACEHOLDER: mlp accuracy]":              "93.25%",
    "[PLACEHOLDER: mlp f1]":                    "93.67%",
    "[PLACEHOLDER: mlp precision]":             "94.96%",
    "[PLACEHOLDER: mlp recall]":               "92.61%",

    # ── Per-class RF metrics ──────────────────────────────────────────────────
    "[PLACEHOLDER: normal precision]":          "96%",
    "[PLACEHOLDER: normal recall]":             "96%",
    "[PLACEHOLDER: normal f1]":                 "96%",
    "[PLACEHOLDER: gaze precision]":            "85%",
    "[PLACEHOLDER: gaze recall]":               "82%",
    "[PLACEHOLDER: gaze f1]":                   "84%",
    "[PLACEHOLDER: external precision]":        "100%",
    "[PLACEHOLDER: external recall]":           "100%",
    "[PLACEHOLDER: external f1]":               "100%",
    "[PLACEHOLDER: multi precision]":           "100%",
    "[PLACEHOLDER: multi recall]":              "89%",
    "[PLACEHOLDER: multi f1]":                  "94%",
    "[PLACEHOLDER: keystroke precision]":       "99%",
    "[PLACEHOLDER: keystroke recall]":          "100%",
    "[PLACEHOLDER: keystroke f1]":              "99%",

    # ── Per-modality ablation ─────────────────────────────────────────────────
    "[PLACEHOLDER: visual f1]":                 "77.6%",
    "[PLACEHOLDER: behavioral f1]":             "61.3%",
    "[PLACEHOLDER: fused f1]":                  "94.6%",
    "[PLACEHOLDER: visual accuracy]":           "91.2%",
    "[PLACEHOLDER: behavioral accuracy]":       "81.9%",
    "[PLACEHOLDER: audio accuracy]":            "86.0%",
    "[PLACEHOLDER: fused accuracy]":            "93.7%",

    # ── Audio classifier ─────────────────────────────────────────────────────
    "[PLACEHOLDER: audio f1]":                  "84.0%",
    "[PLACEHOLDER: audio precision]":           "85.0%",
    "[PLACEHOLDER: audio recall]":              "84.0%",
    "[PLACEHOLDER: suspicious recall]":         "92%",

    # ── Training details ──────────────────────────────────────────────────────
    "[PLACEHOLDER: epochs]":                    "36",
    "[PLACEHOLDER: training epochs]":          "36",
    "[PLACEHOLDER: batch size]":               "512",
    "[PLACEHOLDER: learning rate]":            "1e-3",
    "[PLACEHOLDER: optimizer]":               "AdamW (β₁=0.9, β₂=0.999, ε=1e-8)",
    "[PLACEHOLDER: dropout]":                  "0.30",
    "[PLACEHOLDER: early stopping patience]":  "10 epochs",

    # ── Cross-validation ─────────────────────────────────────────────────────
    "[PLACEHOLDER: cv f1]":                    "0.9465 ± 0.0044",
    "[PLACEHOLDER: cv score]":                 "0.9465 ± 0.0044",

    # ── Generic catch-all (must be last) ─────────────────────────────────────
    "[PLACEHOLDER]":                           "[SEE ACTUAL VALUE]",
}

# ── Dataset citations to append to references ─────────────────────────────────
DATASET_REFS = [
    '[D1] K. S. Killourhy and R. A. Maxion, "Comparing Anomaly-Detection Algorithms for Keystroke Dynamics," in Proc. IEEE/IFIP Int. Conf. Dependable Systems & Networks (DSN), 2009, pp. 125–134.',
    '[D2] X. Zhang, Y. Sugano, M. Fritz, and A. Bulling, "Appearance-Based Gaze Estimation in the Wild," in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2015, pp. 4511–4520.',
    '[D3] A. Gupta, A. D'Cunha, K. Awasthi, and V. Balasubramanian, "DAiSEE: Towards User Engagement Recognition in the Wild," arXiv:1609.01885, 2016.',
    '[D4] K. J. Piczak, "ESC: Dataset for Environmental Sound Classification," in Proc. ACM Int. Conf. Multimedia (MM), 2015, pp. 1015–1018.',
    '[D5] V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, "LibriSpeech: An ASR Corpus Based on Public Domain Audio Books," in Proc. IEEE Int. Conf. Acoustics, Speech and Signal Processing (ICASSP), 2015, pp. 5206–5210.',
    '[D6] H. Kuehne, H. Jhuang, E. Garrote, T. Poggio, and T. Serre, "HMDB: A Large Video Database for Human Motion Recognition," in Proc. IEEE Int. Conf. Computer Vision (ICCV), 2011, pp. 2556–2563.',
]

# ── Core replacement logic ────────────────────────────────────────────────────

def replace_in_paragraph(para, replacements):
    """Replace placeholder text inside a paragraph, preserving run formatting."""
    for old, new in replacements.items():
        if old.lower() in para.text.lower():
            # Simple full-paragraph text replace (merges runs)
            full_text = para.text
            new_text = re.sub(re.escape(old), new, full_text, flags=re.IGNORECASE)
            if new_text != full_text:
                # Clear existing runs and set new text on first run
                for i, run in enumerate(para.runs):
                    if i == 0:
                        run.text = new_text
                    else:
                        run.text = ""


def replace_all(doc, replacements):
    count = 0
    # Body paragraphs
    for para in doc.paragraphs:
        before = para.text
        replace_in_paragraph(para, replacements)
        if para.text != before:
            count += 1
    # Table cells
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    before = para.text
                    replace_in_paragraph(para, replacements)
                    if para.text != before:
                        count += 1
    return count


def append_dataset_refs(doc, refs):
    """Find the References section and append dataset citations."""
    ref_section_found = False
    for para in doc.paragraphs:
        if re.search(r'references', para.text, re.IGNORECASE) and len(para.text.strip()) < 30:
            ref_section_found = True

    if not ref_section_found:
        # Add a references heading at the end
        h = doc.add_paragraph()
        run = h.add_run("DATASET REFERENCES")
        run.font.name = "Times New Roman"
        run.font.size = Pt(10)
        run.font.bold = True

    # Append dataset refs
    doc.add_paragraph()  # spacer
    note = doc.add_paragraph()
    nr = note.add_run("Datasets Used in This Work:")
    nr.font.name = "Times New Roman"
    nr.font.size = Pt(10)
    nr.font.bold = True

    for ref in refs:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Pt(18)
        p.paragraph_format.first_line_indent = Pt(-18)
        p.paragraph_format.space_after = Pt(3)
        run = p.add_run(ref)
        run.font.name = "Times New Roman"
        run.font.size = Pt(9)

    print(f"  Added {len(refs)} dataset citations.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not SRC.exists():
        print(f"ERROR: Source file not found at {SRC}")
        print("Make sure the .docx is in your Downloads folder and WSL can see /mnt/c/")
        sys.exit(1)

    print(f"Reading : {SRC}")
    doc = Document(str(SRC))

    print("Replacing placeholders...")
    count = replace_all(doc, REPLACEMENTS)
    print(f"  {count} paragraph(s) updated.")

    print("Appending dataset citations...")
    append_dataset_refs(doc, DATASET_REFS)

    doc.save(str(DEST))
    print(f"\nDone! Saved to:\n  {DEST}")
    print("\nOpen the FINAL file and check all [SEE ACTUAL VALUE] tags — those need manual review.")


if __name__ == "__main__":
    main()
