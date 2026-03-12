"""CLI entry point for ACL Injury Risk Predictor.

Usage:
    python scripts/run_pipeline.py download     # Download datasets
    python scripts/run_pipeline.py train        # Train models
    python scripts/run_pipeline.py evaluate     # Evaluate trained models
    python scripts/run_pipeline.py predict VIDEO_PATH  # Predict from video
    python scripts/run_pipeline.py app          # Launch web app
"""

import sys
import logging
from pathlib import Path

import click
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@click.group()
def cli():
    """ACL Injury Risk Predictor — ML pipeline for biomechanical risk assessment."""
    pass


@cli.command()
def download():
    """Download training datasets (COMPWALK-ACL and UCI Gait)."""
    from src.data.download import download_all
    download_all()


@cli.command()
def train():
    """Train ML models on downloaded biomechanics data."""
    from config import PROCESSED_DATA_DIR
    from src.data.compwalk_loader import build_dataset
    from src.data.uci_loader import load_uci_data
    from src.features.feature_pipeline import build_feature_matrix, FEATURE_NAMES
    from src.models.train import train_pipeline
    import pandas as pd

    click.echo("Loading COMPWALK-ACL dataset...")
    compwalk_df = build_dataset()

    click.echo("Loading UCI Gait dataset...")
    uci_df = load_uci_data()

    # Combine datasets
    if not compwalk_df.empty and not uci_df.empty:
        dataset = pd.concat([compwalk_df, uci_df], ignore_index=True)
    elif not compwalk_df.empty:
        dataset = compwalk_df
    elif not uci_df.empty:
        dataset = uci_df
    else:
        click.echo("ERROR: No data available. Run 'download' first.", err=True)
        sys.exit(1)

    click.echo(f"Total records: {len(dataset)}")
    click.echo(f"Participants: {dataset['participant_id'].nunique()}")

    click.echo("Extracting features...")
    X, y, pids = build_feature_matrix(dataset)

    if X.size == 0:
        click.echo("ERROR: Feature extraction produced empty matrix.", err=True)
        sys.exit(1)

    click.echo(f"Feature matrix: {X.shape[0]} samples x {X.shape[1]} features")
    click.echo(f"Class distribution: healthy={sum(y==0)}, injured={sum(y==1)}")

    # Save feature matrix
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        PROCESSED_DATA_DIR / "feature_matrix.npz",
        X=X, y=y,
        participant_ids=np.array(pids),
        feature_names=np.array(FEATURE_NAMES),
    )
    click.echo(f"Feature matrix saved to {PROCESSED_DATA_DIR / 'feature_matrix.npz'}")

    click.echo("Training models...")
    results = train_pipeline(X, y)

    click.echo("\n=== Training Results ===")
    for model_name in ["random_forest", "logistic_regression"]:
        if model_name in results:
            score = results[model_name]["cv_score"]
            click.echo(f"{model_name}: CV ROC AUC = {score:.4f}")

    click.echo("\nModels saved to models/ directory.")


@cli.command()
def evaluate():
    """Evaluate trained models and generate visualizations."""
    from config import PROCESSED_DATA_DIR, MODELS_DIR, OUTPUT_DIR
    from src.models.evaluate import generate_evaluation_report
    from src.models.train import prepare_data

    data = np.load(PROCESSED_DATA_DIR / "feature_matrix.npz", allow_pickle=True)
    X, y = data["X"], data["y"]

    X_train, X_test, y_train, y_test, scaler = prepare_data(X, y)

    # Load models
    import joblib
    results = {"X_test": X_test, "y_test": y_test}

    for model_name in ["random_forest", "logistic_regression"]:
        model_path = MODELS_DIR / f"{model_name}.joblib"
        if model_path.exists():
            artifact = joblib.load(model_path)
            results[model_name] = {"model": artifact["model"]}

    report = generate_evaluation_report(results, OUTPUT_DIR)

    click.echo("\n=== Evaluation Report ===")
    for model_name, metrics in report.items():
        click.echo(f"\n{model_name}:")
        click.echo(f"  Accuracy:  {metrics['accuracy']:.4f}")
        click.echo(f"  Precision: {metrics['precision']:.4f}")
        click.echo(f"  Recall:    {metrics['recall']:.4f}")
        click.echo(f"  F1 Score:  {metrics['f1']:.4f}")
        click.echo(f"  ROC AUC:   {metrics['roc_auc']:.4f}")

    click.echo(f"\nPlots saved to {OUTPUT_DIR}/")


@cli.command()
@click.argument("video_path")
@click.option("--model", default="random_forest", help="Model to use for prediction")
@click.option("--output", default=None, help="Output directory for results")
def predict(video_path, model, output):
    """Predict ACL injury risk from a video file."""
    from src.models.predict import predict_from_video
    from config import OUTPUT_DIR

    output_dir = Path(output) if output else OUTPUT_DIR

    click.echo(f"Analyzing video: {video_path}")
    assessment = predict_from_video(video_path, model_name=model)

    if assessment is None:
        click.echo("ERROR: Could not analyze video.", err=True)
        sys.exit(1)

    click.echo(f"\n=== Risk Assessment ===")
    click.echo(f"Risk Score:  {assessment.risk_score:.1%}")
    click.echo(f"Risk Level:  {assessment.risk_level}")
    click.echo(f"Confidence:  {assessment.confidence:.1%}")

    if assessment.contributing_factors:
        click.echo(f"\nTop Contributing Factors:")
        for f in assessment.contributing_factors:
            click.echo(f"  - {f['feature']}: importance={f['importance']:.4f}")


@cli.command()
def app():
    """Launch the interactive Streamlit web application."""
    import subprocess
    app_path = PROJECT_ROOT / "app" / "streamlit_app.py"
    click.echo("Launching ACL Injury Risk Predictor web app...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])


if __name__ == "__main__":
    cli()
