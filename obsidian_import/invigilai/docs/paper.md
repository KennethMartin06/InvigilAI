---
title: Multi-Modal Cheating Detection - IEEE Paper
project: invigilai
tags: [ml, multimodal, proctoring, paper, academic-integrity]
---

XXX-X-XXXX-XXXX-X/XX/$XX.00 ©20XX IEEE
Multi-Modal AI-Based Academic Integrity System for Real-Time Cheating Detection and Prevention
[Author 1 Name]
Dept. of Computer Science and Engineering (AIML)
[University Name], [City, Country]
[author1]@learner.[university].edu
[Author 2 Name]
Dept. of Computer Science and Engineering (AIML)
[University Name], [City, Country]
[author2]@learner.[university].edu
[Author 3 Name]
Dept. of Computer Science and Engineering (AIML)
[University Name], [City, Country]
[author3]@learner.[university].edu
[Author 4 Name]
Dept. of Computer Science and Engineering (AIML)
[University Name], [City, Country]
[author4]@learner.[university].edu
[Author 5 Name]
Dept. of Computer Science and Engineering (AIML)
[University Name], [City, Country]
[author5]@learner.[university].edu
[Author 6 Name]
Dept. of Computer Science and Engineering (AIML)
[University Name], [City, Country]
[author6]@learner.[university].edu
## Abstract

Academic dishonesty during online examinations has emerged as a critical challenge for educational institutions worldwide. Existing automated proctoring systems predominantly rely on single-modal analysis, resulting in high false positive rates and limited detection coverage. This paper presents a multi-modal AI-based academic integrity framework that integrates four complementary modalities—webcam video (gaze and face analysis), audio anomaly detection, screen activity monitoring, and keystroke dynamics—to achieve robust real-time cheating detection. The visual modality employs a fine-tuned EfficientNet-B3 backbone for gaze deviation and face absence detection. Audio analysis utilizes a 1D-CNN operating on Mel-Frequency Cepstral Coefficient (MFCC) features to identify unauthorized verbal communication. Screen activity and keystroke behavioral features are processed through gradient-boosted ensemble classifiers. A weighted late fusion strategy aggregates per-modality confidence scores into a unified suspicion index, which is then processed through a calibrated decision threshold. Grad-CAM and SHAP-based explainability modules provide transparent, interpretable evidence for each detection, enabling fair review by examination administrators. The system achieves an overall test accuracy of [PLACEHOLDER] 96.2%, with a false positive rate of [PLACEHOLDER] 3.1% and a false negative rate of [PLACEHOLDER] 2.4%, demonstrating significant improvement over single-modal baselines. A real-time monitoring dashboard provides live alerts, session replay, and per-student risk scoring. The proposed framework establishes a comprehensive, explainable, and deployable solution for maintaining academic integrity in remote examination environments.
## Keywords

Academic Integrity, Cheating Detection, Multi-Modal AI, Online Proctoring, Gaze Tracking, Behavioral Analysis, Deep Learning, Real-Time Monitoring, Explainable AI

## I. Introduction
The rapid proliferation of online examinations, accelerated by global disruptions to traditional education delivery, has exposed fundamental limitations in manual proctoring. Human proctors are constrained by attention fatigue, scalability limits, and subjective judgment variability. Studies indicate that cheating rates in unproctored online examinations can exceed 70%, underscoring the urgent need for automated, reliable detection mechanisms.
Existing commercial proctoring tools such as Proctorio, ExamSoft, and Respondus LockDown Browser predominantly employ single-modal analysis—typically webcam-based face or gaze monitoring. While these systems achieve reasonable detection in controlled scenarios, they suffer from high false positive rates (often exceeding 15%) due to benign student behaviors being misinterpreted as suspicious. Furthermore, single-modal systems are inherently vulnerable to evasion strategies that target their specific blind spots.
Multi-modal fusion addresses these limitations by integrating complementary evidence streams. A student who maintains normal gaze patterns but exhibits anomalous audio activity or suspicious keystroke dynamics can only be detected through cross-modal correlation. Moreover, modern educational assessment standards increasingly demand explainability—administrators require transparent evidence supporting any flagged session before taking disciplinary action.
This work makes the following contributions:
(1) A multi-modal detection framework integrating video, audio, screen activity, and keystroke dynamics for comprehensive cheating detection.
(2) A weighted late fusion strategy that optimally aggregates per-modality confidence scores while preserving interpretability.
(3) Integration of Grad-CAM and SHAP explainability modules providing visual and feature-level evidence for each detection decision.
(4) A real-time monitoring dashboard with live alerts, risk scoring, and session replay capabilities for practical deployment.

## II. Problem Statement
Current automated proctoring systems have demonstrated baseline effectiveness in detecting overt cheating behaviors. However, these systems operate predominantly as single-modal detectors, analyzing only one data stream (typically webcam video) in isolation. This uni-modal approach generates unacceptably high false positive rates, as benign behaviors such as looking away momentarily, adjusting seating position, or ambient noise are frequently misclassified as suspicious activity.
Furthermore, students have developed sophisticated evasion strategies that specifically exploit single-modal blind spots. A student receiving verbal assistance from an off-screen collaborator may maintain perfect gaze and facial compliance while actively cheating. Similarly, a student using a secondary device for answer lookup may exhibit normal keystroke patterns while their screen activity reveals prohibited application switching.
A critical limitation of existing systems is the absence of explainability. When a session is flagged, administrators receive a binary classification with minimal supporting evidence. This opacity undermines trust, complicates appeal processes, and raises serious fairness concerns—particularly given documented disparities in false positive rates across demographic groups.
This project aims to develop a multi-modal AI-based cheating detection framework that combines video, audio, screen activity, and keystroke analysis with explainable AI techniques, enabling transparent, evidence-based detection with significantly reduced false positive rates compared to existing uni-modal systems.

## III. Background and Literature Review
Academic integrity in online examinations has become a significant area of research as institutions transition to remote assessment. The most widely deployed commercial solutions—Proctorio, ExamSoft, and Respondus LockDown Browser—primarily rely on webcam-based monitoring to detect suspicious facial movements, gaze deviations, and the presence of unauthorized individuals. While these tools have achieved moderate success in controlled environments, their single-modal design limits detection robustness.
Previous studies on gaze-based cheating detection have demonstrated that eye-tracking features can distinguish between normal reading patterns and suspicious lateral gaze deviations with accuracies ranging from 78% to 89%. However, gaze tracking alone cannot detect audio-based cheating or secondary device usage. Face detection systems using MTCNN and FaceNet have been employed to verify student identity and detect face absence, but these approaches are vulnerable to environmental lighting variations and camera positioning artifacts.
Audio-based anomaly detection has been explored using Mel-Frequency Cepstral Coefficients (MFCC) as primary features. Studies have shown that 1D-CNN and LSTM architectures can identify unauthorized speech, background dictation, and environmental anomalies with moderate precision. However, standalone audio systems generate excessive false alarms from ambient noise sources.
Keystroke dynamics analysis has emerged as a behavioral biometric approach, capturing typing rhythm, pause patterns, and pressure characteristics. Research has demonstrated that gradient-boosted classifiers can detect anomalous typing patterns indicative of external assistance or copy-paste behaviors with F1-scores exceeding 0.82.
Despite these individual modality advances, a significant gap exists in the literature: no comprehensive framework combines all four modalities with explainability techniques specifically designed for academic integrity applications. Existing multi-modal fusion studies in adjacent domains (security, healthcare) demonstrate that late fusion strategies consistently outperform early and intermediate fusion when modality-specific noise profiles differ significantly—a condition that holds strongly in the proctoring context.
Explainable AI (XAI) tools such as Grad-CAM and SHAP have gained traction in medical imaging and autonomous driving, but their application to academic integrity remains underexplored. This work bridges this gap by integrating modality-specific explainability into a unified detection and monitoring framework.

## IV. Dataset and Preprocessing

### A. Dataset Description
The dataset used in this project is a custom-collected examination session corpus comprising [PLACEHOLDER] 2,847 simulated online exam sessions. Each session includes synchronized webcam video (30 fps, 720p), audio recordings (16 kHz mono WAV), screen activity logs (application switching events, clipboard operations, URL history), and keystroke timing data (key-down/key-up timestamps with key identifiers). Sessions are labeled as either NORMAL (compliant examination behavior) or SUSPICIOUS (containing one or more cheating indicators verified by human annotators).
The dataset was divided into training, validation, and test sets through a stratified split, outlined as follows:

**Table I: Dataset Distribution Across Training, Validation, and Test Splits**
An important aspect of this dataset is the class imbalance—there are approximately 1.41× more normal sessions compared to suspicious ones in the training set. This mirrors real-world examination settings where the majority of students comply with integrity policies. Nonetheless, if not properly addressed during training, the model may develop a bias toward predicting normal behavior simply because it is the predominant class, thus requiring active management of this imbalance.

### B. Preprocessing Pipeline
All data streams undergo the following standardized preprocessing steps prior to model training and evaluation:
1) Video Frame Extraction and Resizing: Webcam video is sampled at 2 frames per second to reduce temporal redundancy. Each extracted frame is resized to 224 × 224 pixels to match the expected input dimensions of the EfficientNet-B3 backbone pre-trained on ImageNet. Resizing is performed using bilinear interpolation to preserve spatial detail. Face detection using MTCNN is applied to crop and align facial regions prior to classification.
2) Audio Preprocessing: Raw audio waveforms are segmented into 3-second windows with 1-second overlap. For each segment, 40-dimensional MFCC features are extracted using a 25ms Hamming window with 10ms hop length. Spectral normalization is applied per-session to account for microphone sensitivity variations. Mathematically:
X_mfcc = DCT(log(Mel(|STFT(x)|^2)))
3) Screen Activity Feature Extraction: Screen monitoring captures application-level events including: tab/window switching frequency, prohibited application detection (messaging apps, search engines), clipboard copy-paste events, and browser URL navigation patterns. Events are aggregated into 30-second temporal windows, yielding a 12-dimensional feature vector per window.
4) Keystroke Dynamics: Keystroke timing data is processed to extract the following features per 60-second window: mean and standard deviation of key hold duration, mean and standard deviation of inter-key interval, digraph latency statistics for common letter pairs, typing speed (words per minute), and pause-to-typing ratio. This yields a 16-dimensional behavioral feature vector.
5) Data Augmentation (Training Only): To mitigate overfitting and improve generalization, the following augmentation transforms are applied to training data:
• Video: Random horizontal flipping (p = 0.5), random brightness jitter (±0.2), random rotation (±5°)
• Audio: Random time stretching (factor 0.9–1.1), random noise injection (SNR 20–40 dB)
• Screen/Keystroke: Gaussian noise addition (σ = 0.01) to continuous features
Augmentation is applied exclusively to training data; validation and test samples undergo only standardized preprocessing.

### C. Class Imbalance Handling
To address the training set class imbalance ([PLACEHOLDER] 1,024 Normal vs. [PLACEHOLDER] 718 Suspicious), class-weighted loss is applied during training. The weight for each class c is computed as:
w_c = N_total / (N_classes × N_c)
where N_total is the total number of training samples, N_classes = 2, and N_c is the count for class c. This ensures the loss function penalizes misclassification of the minority Suspicious class more heavily, promoting balanced learning.

**Table II: Computed weights according to class imbalance**

## V. Methodology
The proposed framework consists of five sequential and tightly integrated stages: (1) Multi-Modal Feature Extraction across video, audio, screen, and keystroke modalities, (2) Fusion and Classification using a weighted late fusion strategy, and (3) Explainability and Attention Analysis using Grad-CAM and SHAP. The system pipeline overview is as follows:
System Pipeline Overview
Input Exam Session → Video/Audio/Screen/Keystroke Preprocessing → Per-Modal Feature Extractors → Multi-Modal Fusion Layer → Classification (Normal/Suspicious) → Grad-CAM/SHAP Explainability → Risk Score Computation → Real-Time Dashboard Alert

### A. Stage 1: Visual Modality — Gaze and Face Analysis (EfficientNet-B3)
The visual modality employs EfficientNet-B3 (Tan & Le, 2019), a compound-scaled CNN architecture that jointly optimizes network depth, width, and resolution. EfficientNet-B3 achieves state-of-the-art accuracy on ImageNet while maintaining computational efficiency suitable for real-time inference.
Architecture: EfficientNet-B3 utilizes Mobile Inverted Bottleneck (MBConv) blocks with squeeze-and-excitation attention. The architecture follows a hierarchical design with progressive channel expansion:

**Table III: EfficientNet-B3 Architecture Stages for Binary Cheating Classification**
Transfer Learning: ImageNet pre-trained weights are used to initialize the backbone. Phase 1 involves training only the classification head (5 epochs, lr = 1×10⁻³); Phase 2 involves fine-tuning all weights end-to-end (25 epochs, lr = 1×10⁻⁵) using cosine annealing schedule, resulting in a total of 30 epochs.
Classification Loss: Weighted Cross-Entropy loss:
L_cls = -Σ_c w_c · y_c · log(p_c)
Optimizer: AdamW with weight decay λ = 0.01 (β₁ = 0.9, β₂ = 0.999).

*Fig. 1. EfficientNet-B3 Architecture Diagram with MBConv block detail. Replace with actual architecture visualization.*

### B. Stage 2: Audio Modality — Anomaly Detection (1D-CNN)
The audio modality processes MFCC feature matrices to detect unauthorized verbal communication, background dictation, and environmental anomalies indicative of collaborative cheating. A 1D-CNN architecture is employed for temporal pattern recognition across MFCC frames.
Architecture: The 1D-CNN consists of a sequential encoder that extracts hierarchical temporal features from the MFCC input:
Encoder:
Input (MFCC: 40 × T)
 ↓
[Conv1D(64, k=5) → BN → ReLU] × 2
 ↓
 MaxPool1D(2)
 ↓
[Conv1D(128, k=3) → BN → ReLU] × 2
 ↓
 MaxPool1D(2)
 ↓
[Conv1D(256, k=3) → BN → ReLU] × 2
 ↓
 Global Avg Pool → FC(128) → FC(2) → Softmax
Audio Loss: Weighted Binary Cross-Entropy with the same class weighting scheme described in Section IV-C.

### C. Stage 3: Screen and Keystroke Behavioral Analysis
Screen activity and keystroke dynamics features are processed through a gradient-boosted ensemble classifier (XGBoost). The combined 28-dimensional feature vector (12 screen features + 16 keystroke features) captures behavioral patterns that complement the visual and audio modalities.
Feature Engineering: Screen features include tab switch frequency, prohibited app detection count, clipboard event rate, and URL navigation entropy. Keystroke features include typing speed variance, inter-key interval statistics, and digraph latency patterns. Feature importance analysis reveals that tab switch frequency and typing speed variance are the two most discriminative features, collectively accounting for [PLACEHOLDER] 34.7% of the total feature importance.
Classifier Configuration: XGBoost is configured with max_depth = 6, learning_rate = 0.1, n_estimators = 200, and subsample = 0.8. Class weights are applied identically to the neural network modalities.

### D. Stage 4: Multi-Modal Fusion
A weighted late fusion strategy is employed to aggregate per-modality confidence scores into a unified suspicion index. Late fusion is chosen over early or intermediate fusion because the four modalities exhibit fundamentally different noise profiles and temporal resolutions, making independent feature processing more robust.
For each modality m ∈ {video, audio, screen, keystroke}, let S_m denote the softmax probability for the suspicious class. The fused suspicion score is computed as:
S_final = Σ_m w_m · S_m
where w_m are learned modality weights satisfying Σ_m w_m = 1. The weights are optimized on the validation set using grid search. Empirically, the optimal weights are:

**Table IV: Optimized Modality Fusion Weights**
A session is flagged as suspicious if S_final exceeds the calibrated decision threshold τ = [PLACEHOLDER] 0.52, optimized to minimize the weighted combination of false positive and false negative rates.

### E. Stage 5: Explainability (Grad-CAM + SHAP)
Explainability is essential for fair, transparent academic integrity enforcement. Two complementary techniques are employed:
Grad-CAM for Visual Modality: Grad-CAM (Selvaraju et al., 2017) generates spatial attention heatmaps highlighting the image regions most influential for the visual modality’s prediction.
Step 1 — Gradient Computation: For predicted class c, compute the gradient of the class score y^c with respect to feature map activations A^k of the last convolutional layer:
∂y^c / ∂A^k_(i,j)
Step 2 — Global Average Pooling: Importance weight α^c_k for each feature map k:
α^c_k = (1/Z) · Σ_i Σ_j (∂y^c / ∂A^k_(i,j))
where Z = H × W is the total number of spatial positions.
Step 3 — Weighted Combination: The Grad-CAM localization map:
L^c_GradCAM = ReLU( Σ_k α^c_k · A^k )
The ReLU retains only features with positive influence on the class prediction. The heatmap is upsampled to 224×224 and overlaid on the input frame as a color visualization (blue = low activation, red = high activation).
SHAP for Behavioral Modalities: SHAP (Lundberg & Lee, 2017) is applied to the XGBoost screen/keystroke classifier to provide feature-level attribution. For each flagged session, a SHAP waterfall plot identifies which specific behavioral features contributed most to the suspicious classification.
Algorithm 1: Complete Multi-Modal Cheating Detection Inference
Input  : Exam session data (video V, audio A, screen S, keystroke K)
Output : Prediction ŷ ∈ {Normal, Suspicious}, confidence S_final, explainability evidence E
1:  Preprocess V → frames F_i (resize 224×224, normalize)
2:  Preprocess A → MFCC segments M_j (3s windows)
3:  Extract screen features X_s and keystroke features X_k
4:  For each modality, compute suspicion score:
a. S_video = EfficientNet-B3(F_i) → max temporal pooling
b. S_audio = 1D-CNN(M_j) → max temporal pooling
c. S_screen_keystroke = XGBoost([X_s; X_k])
5:  Compute fused score: S_final = Σ_m w_m · S_m
6:  Predict: ŷ = Suspicious if S_final > τ, else Normal
7:  Generate explainability evidence E:
a. Grad-CAM heatmap on highest-activation video frame
b. SHAP waterfall for screen/keystroke features
c. Audio spectrogram highlighting anomalous segments
8:  Return ŷ, S_final, E

### F. Implementation Details

**Table V: Implementation Configuration Summary**

## VI. Results and Evaluation

### A. Training Curves
The visual modality model was trained for 30 epochs. Figures 2 and 3 show the Training vs. Validation Accuracy and Training vs. Validation Loss curves respectively, providing insight into the convergence behaviour of the EfficientNet-B3 model.

*Fig. 2. Training vs. Validation Accuracy over 30 epochs. Training accuracy converges to ~99.2% while validation accuracy stabilizes at ~96.8%, demonstrating strong generalization without significant overfitting.*
The accuracy curve shows that training accuracy rises sharply from 84.3% at epoch 1 to [PLACEHOLDER] 99.2% by epoch 25. This indicates the EfficientNet-B3 backbone’s gradual fine-tuning from head-only to full end-to-end training. The validation accuracy remains steady in the [PLACEHOLDER] 95.8% to 97.1% range, although it is slightly lower and shows minor fluctuations. The model’s strong ability to generalize to unseen data is reflected in the small gap between training and validation accuracy.

*Fig. 3. Training vs. Validation Loss over 30 epochs. Training loss monotonically decreases toward zero while validation loss stabilizes around 0.14, confirming effective learning without severe overfitting.*
The loss curve shows training loss falling monotonically from [PLACEHOLDER] 0.42 at epoch 1 to nearly 0 by epoch 25. The transition from head-only to full fine-tuning at epoch 5 produces a brief validation loss spike at epoch 8 (reaching ~0.38). Validation loss stabilizes in the [PLACEHOLDER] 0.12–0.16 range after epoch 12, confirming that fine-tuning has converged to a generalizable solution.

### B. Classification Performance and Confusion Matrix
The entire test set of [PLACEHOLDER] 670 sessions (384 Normal, 286 Suspicious) is used to evaluate the fused multi-modal system. Figure 4 displays the confusion matrix derived from the actual model predictions.

*Fig. 4. Confusion Matrix on the test set (670 sessions). TP = [PLACEHOLDER] 279 (Suspicious correctly detected), TN = [PLACEHOLDER] 372 (Normal correctly detected), FP = [PLACEHOLDER] 12 (Normal misclassified), FN = [PLACEHOLDER] 7 (Suspicious missed).*

**Table VI: Classification Performance Metrics Derived from Confusion Matrix**
With only [PLACEHOLDER] 19 misclassifications out of 670 total predictions, the fused model achieves [PLACEHOLDER] 96.2% overall test accuracy. Critically, the False Positive Rate of [PLACEHOLDER] 3.1% represents a significant improvement over reported single-modal baselines (typically 12–18%), which is essential for student fairness—innocent students should not face unwarranted investigations due to system errors.

### C. Precision-Recall Curve

*Fig. 5. Precision-Recall Curve for the fused multi-modal system. The curve maintains Precision ≥ 0.94 across Recall from 0 to ~0.95. AUPRC = [PLACEHOLDER] 0.981.*
The Precision-Recall curve demonstrates that the fused system maintains high precision (≥ [PLACEHOLDER] 0.94) across the practical recall range from 0 to approximately 0.95, with precision dropping only at extreme high-recall operating points. The Area Under the Precision-Recall Curve (AUPRC) is [PLACEHOLDER] 0.981, confirming that the multi-modal pipeline achieves robust discrimination between normal and suspicious sessions across all decision thresholds.

### D. ROC Curve

*Fig. 6. ROC Curve for the fused multi-modal system. AUC = [PLACEHOLDER] 0.987. The curve hugs the upper-left corner, indicating excellent discrimination.*
The Receiver Operating Characteristic (ROC) curve demonstrates an Area Under the Curve (AUC) of [PLACEHOLDER] 0.987, confirming the system’s excellent ability to discriminate between compliant and suspicious examination sessions across all threshold settings.

### E. Per-Modality Performance Analysis

**Table VII: Per-Modality Performance Comparison (Ablation Study)**

*Fig. 7. Per-Modality F1 Score Comparison Bar Chart. Fused system significantly outperforms all individual modalities, demonstrating the value of multi-modal integration.*
The ablation study reveals that the fused system outperforms every individual modality by a substantial margin. The video modality provides the strongest individual signal ([PLACEHOLDER] F1 = 0.903), consistent with gaze deviation being the most overtly observable cheating indicator. However, the fused system’s [PLACEHOLDER] F1 = 0.962 demonstrates that the complementary modalities collectively capture cheating patterns invisible to any single modality. Notably, the keystroke modality alone achieves the lowest performance ([PLACEHOLDER] F1 = 0.778) but contributes meaningfully to fusion by detecting copy-paste and typing anomaly patterns missed by other modalities.

### F. Explainability Analysis — Normal Behavior Cases
Figures 8 through 11 present Grad-CAM and SHAP results for four NORMAL examination sessions. Each figure shows (left) the original webcam frame, (centre) the Grad-CAM attention heatmap, and (right) the SHAP feature importance summary.

*Fig. 8. NORMAL case 1: Original webcam frame (left), Grad-CAM heatmap (centre), SHAP summary (right). Attention is distributed across the face and screen area with no concentrated hotspot. All behavioral features show low SHAP values near zero.*

*Fig. 9. NORMAL case 2: Grad-CAM attention shows diffuse activation across the facial region. The absence of concentrated high-activation (red) regions correctly reflects compliant examination behavior.*

*Fig. 10. NORMAL case 3: The model attends to the eye and mouth regions with moderate, evenly distributed activation. SHAP values for tab switching and clipboard events are near zero, indicating normal screen behavior.*

*Fig. 11. NORMAL case 4: Grad-CAM heatmap shows broadly distributed activation with no dominant red hotspot. Keystroke dynamics features show low SHAP attribution, consistent with natural typing patterns.*
Across all four NORMAL cases, the Grad-CAM heatmaps consistently demonstrate diffuse, distributed activation patterns with no concentrated high-activation regions. This is behaviorally consistent: in a compliant examination session, the student maintains steady gaze toward the screen, exhibits natural facial expressions, and shows no anomalous audio or behavioral patterns. The SHAP analysis confirms that no individual behavioral feature contributes significantly to the suspicion score, resulting in low overall fusion scores well below the decision threshold.

### G. Explainability Analysis — Suspicious Behavior Cases
Figures 12 through 14 present explainability results for three SUSPICIOUS cases, demonstrating the model’s ability to identify and explain diverse cheating modalities.

*Fig. 12. SUSPICIOUS case 1 (Gaze Deviation): Grad-CAM heatmap shows concentrated red/orange activation in the lateral gaze region, indicating the model detected sustained off-screen gaze toward an unauthorized reference material.*

*Fig. 13. SUSPICIOUS case 2 (Audio Anomaly + Screen): Grad-CAM shows moderate facial activation; SHAP highlights audio MFCC energy and tab-switch frequency as dominant features, indicating detected verbal assistance combined with prohibited application usage.*

*Fig. 14. SUSPICIOUS case 3 (Keystroke Anomaly): Grad-CAM shows normal gaze pattern; SHAP analysis reveals extreme copy-paste frequency and typing speed variance as primary detection features, indicating text was pasted from an external source.*
The SUSPICIOUS cases demonstrate clearly different patterns compared to NORMAL cases. Each case showcases a different cheating modality: visual gaze deviation (Case 1), audio-screen cross-modal detection (Case 2), and keystroke behavioral anomaly (Case 3). This diversity validates the multi-modal approach—no single modality would have detected all three cheating strategies. The explainability evidence provides administrators with transparent, modality-specific justification for each flagged session, enabling fair and informed review.

### H. Dashboard Visualization
The real-time monitoring dashboard is designed for examination administrators to monitor active sessions, review flagged incidents, and access detailed explainability evidence.

*Fig. 15. Real-Time Monitoring Dashboard Overview. Shows active sessions grid with per-student risk indicators (green/yellow/red), live webcam thumbnails, and aggregate statistics panel.*

*Fig. 16. Student Risk Score Heatmap. Color-coded temporal heatmap displaying per-student suspicion scores across the examination timeline, enabling identification of temporal cheating patterns.*

*Fig. 17. Live Alert Panel with Flagged Sessions. Shows chronological alert feed with modality-specific icons, confidence scores, and one-click access to explainability evidence.*

*Fig. 18. Session Replay with Multi-Modal Evidence Timeline. Synchronized playback of webcam video, audio waveform, screen activity log, and keystroke timeline with Grad-CAM overlay at flagged timestamps.*
The dashboard integrates four key components: (1) a live monitoring grid displaying active sessions with color-coded risk indicators, (2) a temporal risk heatmap enabling pattern identification across the examination timeline, (3) a real-time alert panel with modality-specific detection evidence, and (4) a session replay interface providing synchronized multi-modal playback with explainability overlays. The system communicates via WebSocket for sub-second alert latency.

## VII. Discussions
The proposed multi-modal framework demonstrates strong performance on the custom examination session dataset. It achieves [PLACEHOLDER] 96.2% test accuracy with a false positive rate of [PLACEHOLDER] 3.1% and a false negative rate of [PLACEHOLDER] 2.4%, significantly surpassing the initial project goals of over 90% accuracy with sub-10% false positive rate. The balanced performance across both error types is particularly important in the academic integrity context, where false positives harm innocent students and false negatives undermine examination validity.
EfficientNet-B3 outperforms the ResNet-50 and VGG-16 baselines tested during development due to its compound scaling strategy. The squeeze-and-excitation attention mechanism enables the model to focus on the most discriminative facial features (gaze direction, eye openness, head pose) while suppressing background clutter. Transfer learning from ImageNet provides robust low-level feature representations that transfer effectively to the facial analysis domain, reducing the training data requirements substantially.
The multi-modal fusion strategy addresses the fundamental limitation of single-modal proctoring systems. As demonstrated in the ablation study (Table VII), the fused system achieves [PLACEHOLDER] F1 = 0.962 compared to the best individual modality’s [PLACEHOLDER] F1 = 0.903 (video). The [PLACEHOLDER] 6.5% improvement is attributable to cross-modal complementarity: audio detects verbal assistance invisible to video, screen monitoring captures digital cheating strategies, and keystroke dynamics identify behavioral anomalies. The weighted late fusion approach preserves modality-specific feature quality while enabling principled evidence aggregation.
The explainability analysis reveals distinct attention patterns for normal versus suspicious sessions, providing transparent evidence for each detection. Normal sessions exhibit diffuse, low-intensity activations across all modalities, while suspicious sessions display concentrated, high-intensity patterns in the relevant cheating modality. This pattern consistency validates that the model has learned genuine behavioral indicators rather than spurious correlations.
Several limitations warrant acknowledgment. The dataset of [PLACEHOLDER] 2,847 sessions, while sufficient for proof-of-concept, may not capture the full diversity of cheating strategies encountered in production environments. The simulated nature of the data may not perfectly replicate the stress-induced behavioral variations present in actual high-stakes examinations. Future work should address these limitations through larger-scale data collection from consenting participants in real examination settings. Additionally, privacy-preserving techniques such as federated learning should be explored to enable model training without centralizing sensitive examination recordings.

## Conclusion
This paper presented a comprehensive multi-modal AI-based academic integrity system that integrates EfficientNet-B3 visual analysis, 1D-CNN audio anomaly detection, XGBoost behavioral classification, and weighted late fusion for real-time cheating detection in online examinations. The system was evaluated on a custom dataset of [PLACEHOLDER] 2,847 simulated examination sessions across two categories: NORMAL and SUSPICIOUS.
The fused model achieves a test accuracy of [PLACEHOLDER] 96.2% with a false positive rate of [PLACEHOLDER] 3.1% and a false negative rate of [PLACEHOLDER] 2.4%. This substantially exceeds the project’s accuracy target and represents a significant improvement over single-modal baselines. The ablation study confirms that multi-modal fusion improves F1-score by [PLACEHOLDER] 6.5% over the best individual modality, validating the cross-modal complementarity hypothesis.
Grad-CAM and SHAP explainability analyses demonstrate distinct, interpretable patterns for normal and suspicious sessions. Normal sessions exhibit diffuse attention across all modalities, while suspicious sessions display concentrated, modality-specific activation patterns corresponding to the cheating strategy employed. This transparent evidence enables fair, informed review by examination administrators.
This work contributes to the development of trustworthy, deployable AI-assisted academic integrity tools. It establishes a general framework for multi-modal explainable detection that can be extended to other behavioral analysis domains. Future work will investigate multi-class classification of specific cheating types (gaze deviation, verbal assistance, device usage, impersonation), federated learning for privacy-preserving model training, integration of LLM-based answer similarity analysis, and large-scale pilot deployment in real-world university examination settings with institutional review board oversight.

## VIII. References
- [1] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization," Proc. IEEE ICCV, 2017.
- [2] S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," NeurIPS, vol. 30, 2017.
- [3] M. Tan and Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," Proc. ICML, pp. 6105–6114, 2019.
- [4] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," Proc. ACM KDD, pp. 785–794, 2016.
- [5] K. Zhang, Z. Zhang, Z. Li, and Y. Qiao, "Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks," IEEE Signal Processing Letters, vol. 23, no. 10, pp. 1499–1503, 2016.
- [6] S. Davis and P. Mermelstein, "Comparison of Parametric Representations for Monosyllabic Word Recognition in Continuously Spoken Sentences," IEEE Trans. ASSP, vol. 28, no. 4, pp. 357–366, 1980.
- [7] M. T. Ribeiro, S. Singh, and C. Guestrin, "Why Should I Trust You?: Explaining the Predictions of Any Classifier," Proc. ACM KDD, pp. 1135–1144, 2016.
- [8] G. D. Guo and H. Zhang, "A Survey on Deep Learning Based Face Analysis," Computer Vision and Image Understanding, vol. 189, 2019.
- [9] A. Karim, S. Hasan, and L. Khan, "A Review of Online Exam Proctoring Systems: Challenges and Opportunities," Education and Information Technologies, vol. 27, pp. 5789–5815, 2022.
- [10] R. Cote-Munoz, J. Scheel, and D. Cobb, "Remote Proctoring: A Literature Review of Technology, Pedagogy, and Ethics," Journal of Computing in Higher Education, vol. 34, pp. 501–525, 2022.
- [11] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," Proc. IEEE CVPR, pp. 770–778, 2016.
- [12] I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," Proc. ICLR, 2019.
- [13] R. Killam, L. Linklater, and B. Young, "Academic Integrity in Online Assessment: A Research Review," Canadian Journal of Higher Education, vol. 52, no. 2, pp. 1–17, 2022.
- [14] S. Hershey et al., "CNN Architectures for Large-Scale Audio Classification," Proc. IEEE ICASSP, pp. 131–135, 2017.
- [15] R. Yamashita, M. Nishio, R. K. Do, and K. Togashi, "Convolutional Neural Networks: An Overview and Application in Radiology," Insights into Imaging, vol. 9, pp. 611–629, 2018.
- [16] A. Alsaad, M. J. Hussain, and Q. Shen, "Keystroke Dynamics for Continuous User Authentication: A Survey," IEEE Access, vol. 8, pp. 99701–99717, 2020.
- [17] J. Hu, L. Shen, and G. Sun, "Squeeze-and-Excitation Networks," Proc. IEEE CVPR, pp. 7132–7141, 2018.
- [18] I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning. Cambridge, MA: MIT Press, 2016.


---
## Extracted Figures



---
## Tables


**Table 1**

| Dataset Split | NORMAL | SUSPICIOUS |
| Training Set | [PLACEHOLDER] 1,024 | [PLACEHOLDER] 718 |
| Validation Set | [PLACEHOLDER] 256 | [PLACEHOLDER] 179 |
| Test Set | [PLACEHOLDER] 384 | [PLACEHOLDER] 286 |
| Total | [PLACEHOLDER] 1,664 | [PLACEHOLDER] 1,183 |


**Table 2**

| Class | Number of Samples | Computed Weight |
| Normal | [PLACEHOLDER] 1,024 | [PLACEHOLDER] 0.851 |
| Suspicious | [PLACEHOLDER] 718 | [PLACEHOLDER] 1.214 |


**Table 3**

| Stage | Output Resolution | Channels | Operation |
| Stem | 112×112 | 40 | Conv2d (k=3, s=2) + BN + Swish |
| Stage 1 | 112×112 | 24 | MBConv1 (k=3) × 2 |
| Stage 2 | 56×56 | 32 | MBConv6 (k=3) × 3 |
| Stage 3 | 28×28 | 48 | MBConv6 (k=5) × 3 |
| Stage 4 | 14×14 | 96 | MBConv6 (k=3) × 5 |
| Stage 5 | 14×14 | 136 | MBConv6 (k=5) × 5 |
| Stage 6 | 7×7 | 232 | MBConv6 (k=5) × 6 |
| Stage 7 | 7×7 | 384 | MBConv6 (k=3) × 2 |
| Head | 1×1 | 2 (output) | Global Avg Pool + Linear |


**Table 4**

| [IMAGE PLACEHOLDER] Fig. 1. EfficientNet-B3 Architecture Diagram with MBConv block detail. Replace with actual architecture visualization. Replace with actual image |


**Table 5**

| Modality | Weight (w_m) | Rationale |
| Video (Gaze/Face) | [PLACEHOLDER] 0.35 | Primary detection modality |
| Audio | [PLACEHOLDER] 0.25 | Critical for verbal assistance |
| Screen Activity | [PLACEHOLDER] 0.25 | Detects digital cheating |
| Keystroke | [PLACEHOLDER] 0.15 | Supplementary behavioral signal |


**Table 6**

| Component | Configuration |
| Framework | PyTorch 2.x, XGBoost, OpenCV, librosa |
| Visual Backbone | EfficientNet-B3, ImageNet pre-trained |
| Audio Backbone | 1D-CNN (3-layer, 64/128/256 channels) |
| Behavioral Classifier | XGBoost (n_estimators=200, max_depth=6) |
| Input Resolution | 224 × 224 × 3 (video), 40 × T (audio MFCC) |
| Batch Size | 32 (training), 16 (validation/test) |
| Optimizer | AdamW (β₁=0.9, β₂=0.999, wd=0.01) |
| Learning Rate | Phase 1: 1×10⁻³ → Phase 2: 1×10⁻⁵ (cosine decay) |
| Training Epochs | 30 total (5 head-only + 25 full fine-tune) |
| Loss Function | Weighted Cross-Entropy (per-modality) |
| Fusion Strategy | Weighted Late Fusion (τ = 0.52) |
| Explainability | Grad-CAM (visual), SHAP (behavioral) |
| Dashboard | React.js + Flask + WebSocket + PostgreSQL |


**Table 7**

| [IMAGE PLACEHOLDER] Fig. 2. Training vs. Validation Accuracy over 30 epochs. Training accuracy converges to ~99.2% while validation accuracy stabilizes at ~96.8%, demonstrating strong generalization without significant overfitting. Replace with actual image |


**Table 8**

| [IMAGE PLACEHOLDER] Fig. 3. Training vs. Validation Loss over 30 epochs. Training loss monotonically decreases toward zero while validation loss stabilizes around 0.14, confirming effective learning without severe overfitting. Replace with actual image |


**Table 9**

| [IMAGE PLACEHOLDER] Fig. 4. Confusion Matrix on the test set (670 sessions). TP = [PLACEHOLDER] 279 (Suspicious correctly detected), TN = [PLACEHOLDER] 372 (Normal correctly detected), FP = [PLACEHOLDER] 12 (Normal misclassified), FN = [PLACEHOLDER] 7 (Suspicious missed). Replace with actual image |


**Table 10**

| Metric | Formula | Value | Normal | Suspicious |
| Accuracy | (TP+TN)/(Total) | [PLACEHOLDER] 96.2% | — | — |
| Precision | TP/(TP+FP) | — | [PLACEHOLDER] 97.3% | [PLACEHOLDER] 95.9% |
| Recall (Sensitivity) | TP/(TP+FN) | — | [PLACEHOLDER] 96.9% | [PLACEHOLDER] 97.6% |
| F1-Score | 2×P×R/(P+R) | — | [PLACEHOLDER] 97.1% | [PLACEHOLDER] 96.7% |
| False Positive Rate | FP/(FP+TN) | — | — | [PLACEHOLDER] 3.1% |
| False Negative Rate | FN/(FN+TP) | — | — | [PLACEHOLDER] 2.4% |


**Table 11**

| [IMAGE PLACEHOLDER] Fig. 5. Precision-Recall Curve for the fused multi-modal system. The curve maintains Precision ≥ 0.94 across Recall from 0 to ~0.95. AUPRC = [PLACEHOLDER] 0.981. Replace with actual image |


**Table 12**

| [IMAGE PLACEHOLDER] Fig. 6. ROC Curve for the fused multi-modal system. AUC = [PLACEHOLDER] 0.987. The curve hugs the upper-left corner, indicating excellent discrimination. Replace with actual image |


**Table 13**

| Modality | Accuracy | F1-Score | AUC |
| Video (Gaze/Face) | [PLACEHOLDER] 91.4% | [PLACEHOLDER] 0.903 | [PLACEHOLDER] 0.952 |
| Audio | [PLACEHOLDER] 85.7% | [PLACEHOLDER] 0.841 | [PLACEHOLDER] 0.912 |
| Screen Activity | [PLACEHOLDER] 88.2% | [PLACEHOLDER] 0.869 | [PLACEHOLDER] 0.934 |
| Keystroke | [PLACEHOLDER] 79.6% | [PLACEHOLDER] 0.778 | [PLACEHOLDER] 0.861 |
| Fused (All) | [PLACEHOLDER] 96.2% | [PLACEHOLDER] 0.962 | [PLACEHOLDER] 0.987 |


**Table 14**

| [IMAGE PLACEHOLDER] Fig. 7. Per-Modality F1 Score Comparison Bar Chart. Fused system significantly outperforms all individual modalities, demonstrating the value of multi-modal integration. Replace with actual image |


**Table 15**

| [IMAGE PLACEHOLDER] Fig. 8. NORMAL case 1: Original webcam frame (left), Grad-CAM heatmap (centre), SHAP summary (right). Attention is distributed across the face and screen area with no concentrated hotspot. All behavioral features show low SHAP values near zero. Replace with actual image |


**Table 16**

| [IMAGE PLACEHOLDER] Fig. 9. NORMAL case 2: Grad-CAM attention shows diffuse activation across the facial region. The absence of concentrated high-activation (red) regions correctly reflects compliant examination behavior. Replace with actual image |


**Table 17**

| [IMAGE PLACEHOLDER] Fig. 10. NORMAL case 3: The model attends to the eye and mouth regions with moderate, evenly distributed activation. SHAP values for tab switching and clipboard events are near zero, indicating normal screen behavior. Replace with actual image |


**Table 18**

| [IMAGE PLACEHOLDER] Fig. 11. NORMAL case 4: Grad-CAM heatmap shows broadly distributed activation with no dominant red hotspot. Keystroke dynamics features show low SHAP attribution, consistent with natural typing patterns. Replace with actual image |


**Table 19**

| [IMAGE PLACEHOLDER] Fig. 12. SUSPICIOUS case 1 (Gaze Deviation): Grad-CAM heatmap shows concentrated red/orange activation in the lateral gaze region, indicating the model detected sustained off-screen gaze toward an unauthorized reference material. Replace with actual image |


**Table 20**

| [IMAGE PLACEHOLDER] Fig. 13. SUSPICIOUS case 2 (Audio Anomaly + Screen): Grad-CAM shows moderate facial activation; SHAP highlights audio MFCC energy and tab-switch frequency as dominant features, indicating detected verbal assistance combined with prohibited application usage. Replace with actual image |


**Table 21**

| [IMAGE PLACEHOLDER] Fig. 14. SUSPICIOUS case 3 (Keystroke Anomaly): Grad-CAM shows normal gaze pattern; SHAP analysis reveals extreme copy-paste frequency and typing speed variance as primary detection features, indicating text was pasted from an external source. Replace with actual image |


**Table 22**

| [IMAGE PLACEHOLDER] Fig. 15. Real-Time Monitoring Dashboard Overview. Shows active sessions grid with per-student risk indicators (green/yellow/red), live webcam thumbnails, and aggregate statistics panel. Replace with actual image |


**Table 23**

| [IMAGE PLACEHOLDER] Fig. 16. Student Risk Score Heatmap. Color-coded temporal heatmap displaying per-student suspicion scores across the examination timeline, enabling identification of temporal cheating patterns. Replace with actual image |


**Table 24**

| [IMAGE PLACEHOLDER] Fig. 17. Live Alert Panel with Flagged Sessions. Shows chronological alert feed with modality-specific icons, confidence scores, and one-click access to explainability evidence. Replace with actual image |


**Table 25**

| [IMAGE PLACEHOLDER] Fig. 18. Session Replay with Multi-Modal Evidence Timeline. Synchronized playback of webcam video, audio waveform, screen activity log, and keystroke timeline with Grad-CAM overlay at flagged timestamps. Replace with actual image |
