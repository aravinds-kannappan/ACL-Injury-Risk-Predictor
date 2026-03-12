# ACL Injury Risk Predictor

An end-to-end machine learning system that predicts ACL (Anterior Cruciate Ligament) injury risk from athlete movement video using pose estimation and biomechanical feature analysis.

**Author:** Aravind Kannappan — all code in this repository was written solely by me. Third-party tools used: [MediaPipe](https://github.com/google/mediapipe) (Google) for pose estimation, [scikit-learn](https://scikit-learn.org/) for ML models, [Streamlit](https://streamlit.io/) for the web application.

## Clinical Context

Non-contact ACL injuries are among the most common and career-threatening injuries in sports like basketball and football. Research shows that dangerous **knee valgus** (inward knee collapse) during landing and cutting motions is the primary biomechanical risk factor ([Hewett et al., 2005](https://doi.org/10.1177/0363546504269591)). This system quantifies that risk from video alone, without requiring expensive motion capture equipment.

## Architecture

```
Input Video → MediaPipe Pose Detection → Joint Angle Computation → Feature Extraction → ML Model → Risk Score
                    ↓                           ↓                        ↓
              33 body landmarks        Knee flexion, hip flexion,    ~100 statistical
              per frame (3D)           knee valgus, ankle angles      features per sample
```

## Datasets

This project is trained exclusively on **real clinical biomechanics data** (no synthetic data):

1. **COMPWALK-ACL Dataset** ([Nature Scientific Data, 2025](https://www.nature.com/articles/s41597-025-06307-8))
   - 92 participants: 25 healthy adults, 27 healthy adolescents, 40 ACL-injured patients (pre-surgery)
   - Multi-pace IMU gait kinematics with joint angle time series
   - Binary classification: healthy (0) vs. ACL-injured (1)

2. **UCI Multivariate Gait Dataset** ([UCI ML Repository](https://archive.ics.uci.edu/dataset/760/multivariate+gait+data))
   - 10 healthy subjects, 3 conditions (unbraced, knee brace, ankle brace)
   - Supplementary training data for pipeline validation

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Input representation** | Pose keypoints, not raw video | Lightweight, interpretable, clinically meaningful features |
| **Feature engineering** | Statistical summaries over gait cycles | Bridges clinical IMU data and MediaPipe video output into a unified feature space |
| **Primary model** | Random Forest | Handles non-linear relationships, provides feature importance, robust to small datasets |
| **Baseline model** | Logistic Regression (L1/L2) | Interpretable linear model for comparison |
| **Class imbalance** | Balanced class weights | 52 healthy vs 40 injured — weights prevent majority class bias |
| **Validation** | 5-fold stratified CV + held-out test | Stratified to preserve class ratios in each fold |
| **Risk score** | Model probability (0.0–1.0) | Continuous score is more useful than binary prediction for clinical screening |

## Getting Started

### Prerequisites

- Python 3.9+
- Webcam or video files for inference (optional)

### Installation

```bash
git clone https://github.com/yourusername/acl-injury-risk-predictor.git
cd acl-injury-risk-predictor
pip install -r requirements.txt
```

### Quick Start

```bash
# 1. Download training data
python scripts/run_pipeline.py download

# 2. Train models
python scripts/run_pipeline.py train

# 3. Evaluate models
python scripts/run_pipeline.py evaluate

# 4. Predict from video
python scripts/run_pipeline.py predict path/to/video.mp4

# 5. Launch interactive web app
python scripts/run_pipeline.py app
```

### Web Application

The Streamlit web app provides an interactive interface for:
- Uploading movement videos for real-time analysis
- Viewing pose estimation overlays with skeleton visualization
- Exploring joint angle time series plots
- Viewing risk score gauge with contributing factors
- Comparing model performance (Random Forest vs Logistic Regression)
- Interactive feature exploration with distribution and scatter plots

```bash
streamlit run app/streamlit_app.py
```

## Project Structure

```
├── README.md
├── requirements.txt
├── config.py                          # Central configuration
├── app/
│   └── streamlit_app.py               # Interactive web application
├── src/
│   ├── data/
│   │   ├── download.py                # Dataset download automation
│   │   ├── compwalk_loader.py         # COMPWALK-ACL parser (.xlsx)
│   │   ├── uci_loader.py             # UCI Gait data loader
│   │   └── preprocessing.py          # Normalization, outlier removal
│   ├── features/
│   │   ├── joint_angles.py           # 3D angle computation (knee flexion, valgus, etc.)
│   │   ├── gait_features.py          # Gait cycle detection, per-cycle statistics
│   │   └── feature_pipeline.py       # Unified feature extraction (dataset ↔ MediaPipe bridge)
│   ├── pose/
│   │   ├── mediapipe_estimator.py    # MediaPipe wrapper for pose detection
│   │   └── video_processor.py        # Video I/O and frame processing
│   ├── models/
│   │   ├── train.py                  # Training with GridSearchCV
│   │   ├── evaluate.py               # Metrics, ROC curves, confusion matrices
│   │   └── predict.py                # End-to-end inference pipeline
│   └── visualization/
│       ├── plots.py                  # Joint angle and distribution plots
│       ├── dashboard.py              # Risk score gauge and dashboard
│       └── pose_overlay.py           # Skeleton drawing on video frames
├── notebooks/
│   ├── 01_data_exploration.ipynb     # Dataset analysis and visualization
│   ├── 02_feature_engineering.ipynb  # Feature extraction walkthrough
│   └── 03_model_training.ipynb       # Training, evaluation, error analysis
├── tests/
│   ├── test_features.py              # Feature computation tests
│   └── test_models.py                # Model prediction tests
└── scripts/
    └── run_pipeline.py               # CLI entry point
```

## Features Extracted

The pipeline extracts ~100 biomechanical features per sample, organized as:

- **Per-joint, per-side statistics** (knee flexion, hip flexion, ankle dorsiflexion, knee valgus × left/right):
  - Mean, standard deviation, max, min, range over gait cycle
  - Values at gait phases: initial contact (0%), midstance (30%), toe-off (60%), peak swing
  - Angular velocity: maximum and mean rate of angle change

- **Bilateral asymmetry ratios**: `|left - right| / max(|left|, |right|)` for each feature. Asymmetry > 15% is a known injury risk factor.

- **Trunk lean**: Forward lean angle from vertical (mean, std, max, min, range)

## Limitations & Tradeoffs

- **Pose estimation noise**: MediaPipe estimates are less accurate than clinical motion capture, especially for frontal plane angles (knee valgus). Sagittal plane features are more reliable.
- **Small dataset**: 92 participants total. Mitigated by balanced class weights, stratified CV, and regularization.
- **Simplified risk model**: Real ACL injury risk depends on training load, injury history, fatigue, and genetics — this system uses biomechanics only.
- **Single-camera limitation**: Knee valgus requires frontal plane view. Single-camera depth estimates are approximate.

## Future Improvements

- **Temporal models** (LSTM, Transformer) to capture movement sequence dynamics
- **Multi-camera fusion** for more accurate 3D reconstruction
- **Player load integration** (training volume, fatigue metrics)
- **Prospective validation** with actual injury outcome tracking
- **Real-time mobile deployment** for on-field screening

## Running Tests

```bash
pytest tests/ -v
```

## References

1. Hewett, T. E., et al. (2005). Biomechanical measures of neuromuscular control and valgus loading of the knee predict anterior cruciate ligament injury risk in female athletes. *Am J Sports Med*, 33(4), 492-501.
2. COMPWALK-ACL Dataset. (2025). Multi-pace IMU gait kinematics. *Nature Scientific Data*. [DOI](https://www.nature.com/articles/s41597-025-06307-8)
3. Lugaresi, C., et al. (2019). MediaPipe: A framework for building perception pipelines. *arXiv:1906.08172*.

## License

MIT License
