---
title: InvigilAI — Project Index
project: invigilai
tags: [index, ml, multimodal, proctoring, academic-integrity]
status: active
---

# InvigilAI — Multi-Modal Cheating Detection

Multi-modal AI-based academic integrity system integrating video, audio, screen, and keystroke modalities with explainable AI for real-time cheating detection during online exams.

## Map

### [[docs/README|Project README]]
Top-level architecture, feature list, installation.

### [[docs/paper|IEEE Paper]]
Full writeup — literature review, methodology, results, references.
Figures: [[docs/paper-figures|paper-figures/]]

## Architecture

- [[architecture/config|config.py]] — every knob in the system (360 lines)
- [[architecture/main|main.py]] — end-to-end pipeline orchestrator
- [[architecture/requirements|requirements.txt]] — tech stack
- [[architecture/backend-overview|Real-Time Backend Overview]]
- [[architecture/frontend-overview|Frontend Overview]]

## Learnings (reusable ML patterns)

### Core
- [[learnings/training-patterns|Training Patterns]] — focal loss, mixup, SWA, warmup
- [[learnings/evaluation-playbook|ML Evaluation Playbook]] — fairness, FGSM, OOD, drift, conformal
- [[learnings/preprocessing-template|Tabular Preprocessing Template]] — GAN, SMOTE-ENN, NoisyStudent

### Data augmentation / synthesis
- [[learnings/gan_synthetic|Conditional WGAN-GP]] (gan_synthetic.py)
- [[learnings/noisy_student|NoisyStudent Self-Training]] (noisy_student.py)
- [[learnings/temporal_augment|Temporal Augmentation]] (temporal_augment.py)
- [[learnings/smote_enn|SMOTE-ENN]] (smote_enn.py)
- [[learnings/augment|Minority Augmentation]] (augment.py)

### Architectures
- [[learnings/architectures/transformer_model|Transformer Encoder]]
- [[learnings/architectures/tcn_model|Temporal CNN]]
- [[learnings/architectures/gnn_model|Graph Neural Network]]
- [[learnings/architectures/vit_model|Vision Transformer]]

### Monitoring
- [[learnings/monitoring/degradation_tracker|Degradation Tracker]]
- [[learnings/monitoring/drift_detector|Concept Drift (PSI)]]
- [[learnings/monitoring/advanced_metrics|Advanced Metrics]]

### Feature engineering
- [[learnings/feature-engineering/advanced_features|Advanced Features]] — blink, fixation, bigrams
- [[learnings/feature-engineering/cross_modal|Cross-Modal Attention]] — gaze ↔ keystroke

### Other
- [[learnings/domain_adaptation|DANN Domain Adaptation]]
- [[learnings/ensemble_advanced|Weighted Voting + Cascade]]
- [[learnings/ood_detector|OOD Detector]]
- [[learnings/tune|Optuna Tuning]]
- [[learnings/model_utils|Shared Utilities]] — Wilson CIs, Mahalanobis, cascade

## How-to

- [[how-to/fast-training|Fast Training (Presentation Mode)]] — ~20 min on CPU
- [[how-to/train_demo|train_demo.py]] source
- [[how-to/insert_figures|insert_figures.py]] — inject figures into IEEE docx
- [[how-to/fill_paper_placeholders|fill_paper_placeholders.py]] — programmatic number insertion

## Tech stack highlights

- PyTorch (MLP, Transformer, TCN, GNN, ViT, GAN)
- scikit-learn (RF, SVM, stacking, metrics)
- LightGBM + XGBoost (gradient boosting)
- imbalanced-learn (Borderline-SMOTE + ENN)
- SHAP (tree explainer)
- Optuna (hyperparameter search)
- MediaPipe (face + gaze landmarks)
- ONNX Runtime (production inference)
- Flask + SocketIO (backend)
- React + Vite (frontend)

## Results (current)

| Metric | Value |
|--------|-------|
| Test accuracy (fused) | ~96.2% |
| AUC-ROC | ~0.987 |
| AUPRC | ~0.981 |
| False positive rate | ~3.1% |
| False negative rate | ~2.4% |

## Related notes

Tags to browse: `#ml` `#multimodal` `#proctoring` `#academic-integrity` `#training` `#evaluation`
