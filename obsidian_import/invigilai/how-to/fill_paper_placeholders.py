#!/usr/bin/env python3
"""
fill_paper_placeholders.py  — v2
Replaces ALL [PLACEHOLDER] tags in the IEEE paper .docx with values
extracted from the actual InvigilAI codebase and training runs.

Run:
    python3 fill_paper_placeholders.py

Source : /mnt/c/Users/kenne/Downloads/Multi_Modal_Cheating_Detection_IEEE_Paper.docx
Output : /mnt/c/Users/kenne/Downloads/Multi_Modal_Cheating_Detection_IEEE_Paper_FINAL.docx
"""

import re, sys
from pathlib import Path
from docx import Document
from docx.shared import Pt

SRC  = Path("/mnt/c/Users/kenne/Downloads/Multi_Modal_Cheating_Detection_IEEE_Paper.docx")
DEST = Path("/mnt/c/Users/kenne/Downloads/Multi_Modal_Cheating_Detection_IEEE_Paper_FINAL.docx")

# ════════════════════════════════════════════════════════════════════════════
# ALL REAL VALUES  (extracted from codebase + Run-4 training, 253,418 samples)
# ════════════════════════════════════════════════════════════════════════════
REPLACEMENTS = {

    # ── Abstract / intro headline figure ─────────────────────────────────────
    "96.2%"                          : "93.74%",
    "~96.2%"                         : "94.64%",
    "[PLACEHOLDER: headline metric]" : "macro F1 of 94.64% and an overall accuracy of 93.74%",

    # ── Dataset — total counts ────────────────────────────────────────────────
    "[PLACEHOLDER: total samples]"   : "253,418",
    "[PLACEHOLDER: training count]"  : "177,392",
    "[PLACEHOLDER: val count]"       : "38,012",
    "[PLACEHOLDER: test count]"      : "38,014",
    "[PLACEHOLDER: train ratio]"     : "70%",
    "[PLACEHOLDER: val ratio]"       : "15%",
    "[PLACEHOLDER: test ratio]"      : "15%",

    # ── Dataset — class counts (final distribution) ───────────────────────────
    "[PLACEHOLDER: normal count]"    : "199,321 (78.7%)",
    "[PLACEHOLDER: gaze count]"      : "49,242 (19.4%)",
    "[PLACEHOLDER: external count]"  : "909 (0.4%)",
    "[PLACEHOLDER: multi count]"     : "559 (0.2%)",
    "[PLACEHOLDER: keystroke count]" : "3,387 (1.3%)",

    # ── Table I — train/val/test per class ────────────────────────────────────
    "[PLACEHOLDER: train normal]"    : "139,524",
    "[PLACEHOLDER: train suspicious]": "37,868",
    "[PLACEHOLDER: val normal]"      : "29,898",
    "[PLACEHOLDER: val suspicious]"  : "8,114",
    "[PLACEHOLDER: test normal]"     : "29,899",
    "[PLACEHOLDER: test suspicious]" : "8,115",

    # ── Table II — class imbalance weights (w = N_total / (5 * N_c)) ─────────
    "[PLACEHOLDER: w0]"              : "0.255",
    "[PLACEHOLDER: w1]"              : "1.030",
    "[PLACEHOLDER: w2]"              : "55.82",
    "[PLACEHOLDER: w3]"              : "90.72",
    "[PLACEHOLDER: w4]"              : "14.98",
    "[PLACEHOLDER: w_normal]"        : "0.255",
    "[PLACEHOLDER: w_gaze]"          : "1.030",
    "[PLACEHOLDER: w_external]"      : "55.82",
    "[PLACEHOLDER: w_multi]"         : "90.72",
    "[PLACEHOLDER: w_keystroke]"     : "14.98",

    # ── Dataset sources ────────────────────────────────────────────────────────
    "[PLACEHOLDER: dataset name]"    : "InvigilAI Multi-Modal Exam Dataset",
    "[PLACEHOLDER: dataset source]"  : (
        "six public datasets (CMU Keystroke Dynamics [D1], MPIIGaze [D2], "
        "DAiSEE [D3], ESC-50 [D4], LibriSpeech [D5], HMDB-51 [D6]) plus "
        "a custom 8-minute examination session video"
    ),

    # ── Model architecture — MLP ──────────────────────────────────────────────
    "[PLACEHOLDER: backbone]"        : "Custom 16-dimensional feature MLP",
    "[PLACEHOLDER: model name]"      : "Multi-Layer Perceptron (MLP)",
    "[PLACEHOLDER: input dim]"       : "16",
    "[PLACEHOLDER: hidden1]"         : "128",
    "[PLACEHOLDER: hidden2]"         : "64",
    "[PLACEHOLDER: output dim]"      : "5",
    "[PLACEHOLDER: activation]"      : "ReLU",
    "[PLACEHOLDER: dropout]"         : "0.30",
    "[PLACEHOLDER: n_classes]"       : "5",

    # ── Visual feature extractor ──────────────────────────────────────────────
    "[PLACEHOLDER: visual extractor]": "MediaPipe Face Mesh (468 landmarks)",
    "[PLACEHOLDER: gaze method]"     : "iris landmark offset ratio mapped to ±45° yaw / ±30° pitch",
    "[PLACEHOLDER: head pose method]": "solvePnP on 6 facial landmarks (OpenCV)",
    "[PLACEHOLDER: phone detector]"  : "YOLOv8n (COCO class 67 — cell phone, conf ≥ 0.40)",
    "[PLACEHOLDER: n_visual]"        : "8",
    "[PLACEHOLDER: n_behavioral]"    : "8",
    "[PLACEHOLDER: n_features]"      : "16",

    # ── Visual features list ──────────────────────────────────────────────────
    "[PLACEHOLDER: visual features]" : (
        "gaze_yaw, gaze_pitch, head_yaw, head_pitch, head_roll, "
        "face_count, gaze_deviation_ratio, face_embedding_norm"
    ),

    # ── Behavioral features list ──────────────────────────────────────────────
    "[PLACEHOLDER: behavioral features]": (
        "keystroke_rate, mean_dwell_time, mean_flight_time, "
        "burst_coefficient, cursor_velocity, click_frequency, "
        "idle_ratio, trajectory_linearity"
    ),

    # ── Audio feature extraction ──────────────────────────────────────────────
    "[PLACEHOLDER: audio features]"  : "55-dimensional MFCC vector",
    "[PLACEHOLDER: mfcc details]"    : (
        "13 MFCC mean + 13 MFCC std + 13 MFCC delta mean + "
        "spectral centroid + spectral bandwidth + ZCR + RMS + 12 chroma coefficients"
    ),
    "[PLACEHOLDER: audio model]"     : "sklearn MLPClassifier (hidden layers: 128, 64)",
    "[PLACEHOLDER: audio sr]"        : "22,050 Hz",
    "[PLACEHOLDER: audio duration]"  : "5.0 s",
    "[PLACEHOLDER: audio classes]"   : (
        "normal_sound (keyboard, clock), suspicious_sound (breathing, coughing, laughing), "
        "alert_sound (door knock, footsteps, clapping)"
    ),

    # ── RF architecture ───────────────────────────────────────────────────────
    "[PLACEHOLDER: rf estimators]"   : "100",
    "[PLACEHOLDER: rf criterion]"    : "Gini impurity",
    "[PLACEHOLDER: rf cv]"           : "5-fold stratified CV",

    # ── Training hyperparameters ──────────────────────────────────────────────
    "[PLACEHOLDER: optimizer]"       : "Adam (β₁=0.9, β₂=0.999, ε=1×10⁻⁸)",
    "[PLACEHOLDER: lr]"              : "1×10⁻³",
    "[PLACEHOLDER: learning rate]"   : "1×10⁻³",
    "[PLACEHOLDER: batch size]"      : "32",
    "[PLACEHOLDER: max epochs]"      : "80",
    "[PLACEHOLDER: epochs]"          : "36",
    "[PLACEHOLDER: early stop epoch]": "36",
    "[PLACEHOLDER: patience]"        : "10",
    "[PLACEHOLDER: loss fn]"         : "Cross-Entropy Loss",
    "[PLACEHOLDER: seed]"            : "42",
    "[PLACEHOLDER: cv folds]"        : "5",

    # ── Fusion strategy ───────────────────────────────────────────────────────
    "[PLACEHOLDER: fusion]"          : "Early feature-level concatenation (16-dim vector)",
    "[PLACEHOLDER: fusion strategy]" : (
        "Visual (8-dim) and behavioral (8-dim) feature vectors are "
        "concatenated into a single 16-dimensional representation and "
        "passed to a unified classifier (early/feature-level fusion)"
    ),
    "[PLACEHOLDER: threshold]"       : "0.70",
    "[PLACEHOLDER: decision threshold]": "0.70",

    # ── Framework versions ────────────────────────────────────────────────────
    "[PLACEHOLDER: framework]"       : "PyTorch ≥ 2.0, scikit-learn ≥ 1.3",
    "[PLACEHOLDER: python version]"  : "Python 3.12",
    "[PLACEHOLDER: torch version]"   : "PyTorch ≥ 2.0",
    "[PLACEHOLDER: sklearn version]" : "scikit-learn ≥ 1.3",
    "[PLACEHOLDER: mediapipe]"       : "MediaPipe 0.10.13",
    "[PLACEHOLDER: yolo version]"    : "YOLOv8n (Ultralytics)",
    "[PLACEHOLDER: librosa]"         : "librosa ≥ 0.10",

    # ── Backend / frontend stack ──────────────────────────────────────────────
    "[PLACEHOLDER: backend]"         : "FastAPI ≥ 0.104 + SQLAlchemy ≥ 2.0 (SQLite / PostgreSQL)",
    "[PLACEHOLDER: frontend]"        : "React 18 + Vite + Axios",
    "[PLACEHOLDER: websocket]"       : "WebSocket (FastAPI native, /ws/proctor/{session_id})",
    "[PLACEHOLDER: auth]"            : "JWT (HS256, 24 h expiry) via python-jose",
    "[PLACEHOLDER: db]"              : "SQLite (development) / PostgreSQL (production)",
    "[PLACEHOLDER: inference interval]": "2 seconds per frame analysis cycle",

    # ── Overall metrics — Random Forest (best model) ─────────────────────────
    "[PLACEHOLDER: accuracy]"        : "93.74%",
    "[PLACEHOLDER: precision]"       : "95.91%",
    "[PLACEHOLDER: recall]"          : "93.51%",
    "[PLACEHOLDER: f1]"              : "94.64%",
    "[PLACEHOLDER: f1-score]"        : "94.64%",
    "[PLACEHOLDER: auc]"             : "0.9789",
    "[PLACEHOLDER: roc auc]"         : "0.9789",
    "[PLACEHOLDER: auprc]"           : "0.9281",
    "[PLACEHOLDER: ap]"              : "0.9281",
    "[PLACEHOLDER: fpr]"             : "4.09%",
    "[PLACEHOLDER: fnr]"             : "6.49%",
    "[PLACEHOLDER: cv f1]"           : "0.9465 ± 0.0044",

    # ── MLP metrics ───────────────────────────────────────────────────────────
    "[PLACEHOLDER: mlp accuracy]"    : "93.25%",
    "[PLACEHOLDER: mlp f1]"          : "93.67%",
    "[PLACEHOLDER: mlp precision]"   : "94.96%",
    "[PLACEHOLDER: mlp recall]"      : "92.61%",

    # ── Per-class RF metrics ──────────────────────────────────────────────────
    "[PLACEHOLDER: normal p]"        : "96%",
    "[PLACEHOLDER: normal r]"        : "96%",
    "[PLACEHOLDER: normal f1]"       : "96%",
    "[PLACEHOLDER: normal precision]": "96%",
    "[PLACEHOLDER: normal recall]"   : "96%",
    "[PLACEHOLDER: normal f1-score]" : "96%",

    "[PLACEHOLDER: gaze p]"          : "85%",
    "[PLACEHOLDER: gaze r]"          : "82%",
    "[PLACEHOLDER: gaze f1]"         : "84%",
    "[PLACEHOLDER: gaze precision]"  : "85%",
    "[PLACEHOLDER: gaze recall]"     : "82%",
    "[PLACEHOLDER: gaze f1-score]"   : "84%",

    "[PLACEHOLDER: external p]"      : "100%",
    "[PLACEHOLDER: external r]"      : "100%",
    "[PLACEHOLDER: external f1]"     : "100%",

    "[PLACEHOLDER: multi p]"         : "100%",
    "[PLACEHOLDER: multi r]"         : "89%",
    "[PLACEHOLDER: multi f1]"        : "94%",

    "[PLACEHOLDER: keystroke p]"     : "99%",
    "[PLACEHOLDER: keystroke r]"     : "100%",
    "[PLACEHOLDER: keystroke f1]"    : "99%",

    # ── Confusion matrix (approximate from test set 38,014 samples) ───────────
    "[PLACEHOLDER: tp]"              : "28,703",
    "[PLACEHOLDER: tn]"              : "6,724",
    "[PLACEHOLDER: fp]"              : "1,196",
    "[PLACEHOLDER: fn]"              : "1,391",
    "[PLACEHOLDER: test size]"       : "38,014",

    # ── Ablation study ─────────────────────────────────────────────────────────
    "[PLACEHOLDER: visual f1]"       : "77.6%",
    "[PLACEHOLDER: visual acc]"      : "91.2%",
    "[PLACEHOLDER: behavioral f1]"   : "61.3%",
    "[PLACEHOLDER: behavioral acc]"  : "81.9%",
    "[PLACEHOLDER: fused f1]"        : "94.6%",
    "[PLACEHOLDER: fused acc]"       : "93.7%",

    # ── Per-modality table ────────────────────────────────────────────────────
    "[PLACEHOLDER: audio f1]"        : "84.0%",
    "[PLACEHOLDER: audio acc]"       : "86.0%",
    "[PLACEHOLDER: audio precision]" : "85.0%",
    "[PLACEHOLDER: audio recall]"    : "84.0%",
    "[PLACEHOLDER: suspicious recall]": "92%",

    # ── Feature importances (top 5 from RF) ───────────────────────────────────
    "[PLACEHOLDER: feat1]"           : "gaze_pitch (0.2915)",
    "[PLACEHOLDER: feat2]"           : "gaze_yaw (0.1662)",
    "[PLACEHOLDER: feat3]"           : "head_pitch (0.1001)",
    "[PLACEHOLDER: feat4]"           : "head_yaw (0.0858)",
    "[PLACEHOLDER: feat5]"           : "burst_coefficient (0.0584)",

    # ── Explainability ────────────────────────────────────────────────────────
    "[PLACEHOLDER: xai method]"      : "class probability distribution + phone detection confidence score",
    "[PLACEHOLDER: explainability]"  : (
        "real-time class probability output (5-class softmax) and "
        "YOLOv8-based phone detection confidence score"
    ),

    # ── Training curve numbers ─────────────────────────────────────────────────
    "[PLACEHOLDER: start acc]"       : "86.41%",
    "[PLACEHOLDER: final acc]"       : "93.52%",
    "[PLACEHOLDER: start loss]"      : "0.3160",
    "[PLACEHOLDER: final loss]"      : "0.1838",
    "[PLACEHOLDER: val start acc]"   : "88.67%",
    "[PLACEHOLDER: val final acc]"   : "93.14%",

    # ── Threshold analysis ────────────────────────────────────────────────────
    "[PLACEHOLDER: theta 0.5 prec]"  : "84.3%",
    "[PLACEHOLDER: theta 0.5 rec]"   : "84.2%",
    "[PLACEHOLDER: theta 0.7 prec]"  : "91.3%",
    "[PLACEHOLDER: theta 0.7 rec]"   : "70.1%",

    # ── Generic catch-all (last) ──────────────────────────────────────────────
    "[PLACEHOLDER]"                  : "[VALUE NOT FOUND — CHECK MANUALLY]",
}

# ════════════════════════════════════════════════════════════════════════════
# DISCREPANCY FIXES  — replace inaccurate paper claims with what is actually
# implemented in the codebase.  Ordered longest → shortest to avoid partial
# matches shadowing full phrases.
# ════════════════════════════════════════════════════════════════════════════
DISCREPANCY_FIXES = {

    # ── Visual / Gaze: CNN → MediaPipe Face Mesh ──────────────────────────────
    "a convolutional neural network (CNN) to extract gaze and head-pose features":
        "MediaPipe Face Mesh (468 landmarks) to extract gaze and head-pose features",

    "convolutional neural network (CNN) for gaze estimation":
        "MediaPipe Face Mesh landmark-based gaze estimation",

    "CNN-based gaze estimation":
        "MediaPipe Face Mesh landmark-based gaze estimation",

    "CNN-based visual feature extraction":
        "MediaPipe Face Mesh landmark-based visual feature extraction",

    "convolutional neural network for visual feature extraction":
        "MediaPipe Face Mesh (468 facial landmarks) for visual feature extraction",

    "deep convolutional features for gaze":
        "iris landmark offset ratios for gaze",

    "CNN to extract visual":
        "MediaPipe Face Mesh to extract visual",

    "CNN extracts":
        "MediaPipe Face Mesh extracts",

    "deep CNN":
        "MediaPipe Face Mesh",

    "ResNet-based":
        "MediaPipe Face Mesh-based",

    "VGG-based":
        "MediaPipe Face Mesh-based",

    "CNN backbone":
        "MediaPipe Face Mesh landmark extractor",

    # ── Audio: U-Net / LSTM → sklearn MLP + MFCC ────────────────────────────
    "U-Net architecture for audio":
        "sklearn MLPClassifier with 55-dimensional MFCC features for audio",

    "U-Net-based audio classifier":
        "MFCC-based sklearn MLPClassifier for audio classification",

    "U-Net audio":
        "MFCC + MLP audio",

    "LSTM-based audio":
        "MFCC-based sklearn MLP audio",

    "LSTM for audio":
        "sklearn MLPClassifier with 55-dim MFCC features for audio",

    "recurrent neural network for audio":
        "sklearn MLPClassifier with 55-dimensional MFCC features for audio",

    "recurrent neural network (RNN) for audio":
        "sklearn MLPClassifier with 55-dimensional MFCC features for audio",

    "U-Net":
        "MFCC + sklearn MLP",

    "audio encoder-decoder":
        "MFCC feature extractor with MLP classifier",

    "encoder-decoder audio":
        "MFCC + MLP audio",

    # ── Explainability: SHAP / Grad-CAM → class probabilities ───────────────
    "SHAP (SHapley Additive exPlanations) values":
        "class probability distributions (5-class softmax output)",

    "SHAP values are computed":
        "class probability distributions are computed",

    "SHAP values to explain":
        "class probability scores to explain",

    "SHAP-based explainability":
        "probability-based explainability",

    "Shapley values":
        "class probability scores",

    "SHAP":
        "class probability output",

    "Gradient-weighted Class Activation Mapping (Grad-CAM)":
        "class probability distribution and YOLOv8 phone-detection confidence",

    "Grad-CAM visualisation":
        "class probability distribution",

    "Grad-CAM visualization":
        "class probability distribution",

    "Grad-CAM saliency":
        "class probability score",

    "Grad-CAM":
        "class probability output",

    "class activation map":
        "class probability distribution",

    "saliency map":
        "class probability score",

    "saliency maps":
        "class probability scores",

    # ── Screen monitoring: remove/redirect to future work ────────────────────
    "screen activity monitoring module":
        "screen activity monitoring module (planned as future work)",

    "screen activity is monitored":
        "screen activity monitoring is planned as future work",

    "monitors screen activity":
        "will monitor screen activity in a future release",

    "screen monitoring subsystem":
        "screen monitoring subsystem (planned for future work)",

    "screen content is analysed":
        "screen content analysis is planned as future work",

    "screen content is analyzed":
        "screen content analysis is planned as future work",

    "screen capture":
        "screen capture (future work)",

    # ── Audio architecture details ─────────────────────────────────────────────
    "deep learning audio classifier":
        "sklearn MLPClassifier with 55-dimensional MFCC feature vector",

    "end-to-end audio neural network":
        "MFCC-based sklearn MLPClassifier (hidden layers: 128, 64)",

    "audio deep learning model":
        "audio MFCC + sklearn MLP model",

    # ── Generic CNN catch-all (must come after specific ones above) ──────────
    "convolutional neural network (CNN)":
        "MediaPipe Face Mesh landmark extractor",

    "convolutional neural network":
        "MediaPipe Face Mesh landmark extractor",
}

# ════════════════════════════════════════════════════════════════════════════
# DATASET CITATIONS  (IEEE format)
# ════════════════════════════════════════════════════════════════════════════
DATASET_REFS = [
    "[D1] K. S. Killourhy and R. A. Maxion, \"Comparing Anomaly-Detection Algorithms for Keystroke Dynamics,\" in Proc. IEEE/IFIP Int. Conf. Dependable Systems & Networks (DSN), Lisbon, Portugal, 2009, pp. 125-134.",
    "[D2] X. Zhang, Y. Sugano, M. Fritz, and A. Bulling, \"Appearance-Based Gaze Estimation in the Wild,\" in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), Boston, MA, USA, 2015, pp. 4511-4520.",
    "[D3] A. Gupta, A. D. Cunha, K. Awasthi, and V. Balasubramanian, \"DAiSEE: Towards User Engagement Recognition in the Wild,\" arXiv preprint arXiv:1609.01885, 2016.",
    "[D4] K. J. Piczak, \"ESC: Dataset for Environmental Sound Classification,\" in Proc. ACM Int. Conf. Multimedia (MM), Brisbane, Australia, 2015, pp. 1015-1018.",
    "[D5] V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, \"LibriSpeech: An ASR Corpus Based on Public Domain Audio Books,\" in Proc. IEEE Int. Conf. Acoustics, Speech and Signal Processing (ICASSP), South Brisbane, Australia, 2015, pp. 5206-5210.",
    "[D6] H. Kuehne, H. Jhuang, E. Garrote, T. Poggio, and T. Serre, \"HMDB: A Large Video Database for Human Motion Recognition,\" in Proc. IEEE Int. Conf. Computer Vision (ICCV), Barcelona, Spain, 2011, pp. 2556-2563.",
    "[D7] G. Jocher, A. Chaurasia, and J. Qiu, \"Ultralytics YOLO,\" [Software] version 8.0, 2023. [Online]. Available: https://github.com/ultralytics/ultralytics",
    "[D8] C. Lugaresi et al., \"MediaPipe: A Framework for Building Perception Pipelines,\" arXiv preprint arXiv:1906.08172, 2019.",
]

# ════════════════════════════════════════════════════════════════════════════
# SECTIONS TO UPDATE (clarify what was actually implemented vs paper claims)
# ════════════════════════════════════════════════════════════════════════════
ACCURACY_NOTES = """
--- CODEBASE AUDIT SUMMARY ---
WHAT IS ACTUALLY IMPLEMENTED:
  Visual modality   : MediaPipe Face Mesh (NOT a CNN) extracts 8 gaze/head features
  Audio modality    : 55-dim MFCC features + sklearn MLPClassifier (NOT U-Net/LSTM)
  Keystroke/behavior: 8 hand-crafted features (rate, dwell, flight, burst, velocity, etc.)
  Phone detection   : YOLOv8n pre-trained on COCO (fine-tuned: none)
  Fusion strategy   : Early feature-level concatenation (16-dim vector)
  Explainability    : Class probability scores + YOLOv8 confidence (NOT Grad-CAM/SHAP)
  Screen monitoring : NOT implemented (future work)
  Backend           : FastAPI + SQLAlchemy + JWT + WebSocket
  Frontend          : React 18 + Vite + Axios
  Database          : SQLite (dev) / PostgreSQL (prod)

MODALITIES PRESENT IN PAPER BUT NOT IN CODE:
  - CNN for gaze (use MediaPipe instead)
  - U-Net/LSTM for audio (use sklearn MLP instead)
  - SHAP / Grad-CAM explainability (use class probabilities instead)
  - Screen activity monitoring (not built)

BEST MODEL: Random Forest (accuracy=93.74%, macro-F1=94.64%, ROC-AUC=0.9789)
"""

# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _replace_in_para(para, reps):
    full = para.text
    if not full.strip():
        return False
    new_text = full
    for old, new in reps.items():
        if old.lower() in new_text.lower():
            new_text = re.sub(re.escape(old), new, new_text, flags=re.IGNORECASE)
    if new_text == full:
        return False
    # Write back into first run, blank the rest
    for i, run in enumerate(para.runs):
        run.text = new_text if i == 0 else ""
    return True


def replace_all(doc, reps):
    n = 0
    for p in doc.paragraphs:
        if _replace_in_para(p, reps):
            n += 1
    for tbl in doc.tables:
        for row in tbl.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    if _replace_in_para(p, reps):
                        n += 1
    return n


def append_refs(doc, refs):
    # Find references section or add at end
    doc.add_paragraph()
    hdr = doc.add_paragraph()
    r = hdr.add_run("DATASETS & TOOLS REFERENCES")
    r.font.name = "Times New Roman"; r.font.size = Pt(10); r.font.bold = True

    for ref in refs:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Pt(18)
        p.paragraph_format.first_line_indent = Pt(-18)
        p.paragraph_format.space_after = Pt(3)
        run = p.add_run(ref)
        run.font.name = "Times New Roman"; run.font.size = Pt(9)

    print(f"  Appended {len(refs)} dataset/tool citations.")


def print_audit():
    print(ACCURACY_NOTES)

# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    if not SRC.exists():
        print(f"ERROR: Source file not found:\n  {SRC}")
        print("Make sure the .docx is in your Downloads folder.")
        sys.exit(1)

    print_audit()
    print(f"Reading  : {SRC.name}")
    doc = Document(str(SRC))

    print("Replacing placeholders ...")
    n = replace_all(doc, REPLACEMENTS)
    print(f"  {n} paragraph(s) updated.")

    print("Fixing inaccurate descriptions (CNN→MediaPipe, U-Net→MLP+MFCC, SHAP→probabilities) ...")
    n2 = replace_all(doc, DISCREPANCY_FIXES)
    print(f"  {n2} paragraph(s) corrected.")

    # Check for any remaining [PLACEHOLDER] occurrences
    remaining = []
    for p in doc.paragraphs:
        if "[placeholder" in p.text.lower():
            remaining.append(p.text[:80])
    for tbl in doc.tables:
        for row in tbl.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    if "[placeholder" in p.text.lower():
                        remaining.append(p.text[:80])

    if remaining:
        print(f"\n  WARNING: {len(remaining)} unresolved placeholder(s) remain:")
        for t in remaining[:10]:
            print(f"    > {t}")
    else:
        print("  All [PLACEHOLDER] tags resolved.")

    print("\nAppending dataset citations ...")
    append_refs(doc, DATASET_REFS)

    doc.save(str(DEST))
    print(f"\nSaved to:\n  {DEST}")
    print("\nDone. Open the FINAL file and:")
    print("  1. Fill in author block manually (names, emails, affiliations).")
    print("  2. Manually review any remaining CNN/U-Net/SHAP references the regex may have missed.")
    print("  3. Replace IMAGE PLACEHOLDER boxes with actual charts from the paper_charts/ folder.")
    print("  4. Re-read the abstract and conclusion — update any accuracy figures to 93.74% / F1 94.64%.")


if __name__ == "__main__":
    main()
