"""COMPWALK-ACL dataset loader.

Parses the COMPWALK-ACL dataset (Nature Scientific Data, 2025) containing
multi-pace IMU gait kinematics from 92 participants:
- 25 healthy adults
- 27 healthy adolescents
- 40 ACL-injured patients (pre-surgery)

Joint angle time series are extracted from .xlsx files, normalized to
gait cycle percentage (0-100%), and organized into a structured DataFrame.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import RAW_DATA_DIR, COMPWALK_COHORTS, GAIT_CYCLE_POINTS

logger = logging.getLogger(__name__)

# Expected joint angle columns in COMPWALK-ACL .xlsx files
JOINT_COLUMNS = {
    "knee_flexion": ["Knee Flexion/Extension"],
    "hip_flexion": ["Hip Flexion/Extension"],
    "ankle_dorsiflexion": ["Ankle Dorsiflexion/Plantarflexion"],
    "knee_abduction": ["Knee Ab/Adduction"],
    "hip_abduction": ["Hip Ab/Adduction"],
    "ankle_inversion": ["Ankle Inversion/Eversion"],
}

# Alternative column naming patterns
ALT_PATTERNS = {
    "knee_flexion": ["knee_flex", "Knee_FE", "KneeFlexExt"],
    "hip_flexion": ["hip_flex", "Hip_FE", "HipFlexExt"],
    "ankle_dorsiflexion": ["ankle_dorsi", "Ankle_DF", "AnkleDorsiPlantar"],
    "knee_abduction": ["knee_abd", "Knee_AA", "KneeAbdAdd"],
    "hip_abduction": ["hip_abd", "Hip_AA", "HipAbdAdd"],
}


def _find_column(df_columns: list, joint_name: str) -> Optional[str]:
    """Find matching column name using primary and alternative patterns."""
    primary = JOINT_COLUMNS.get(joint_name, [])
    alternatives = ALT_PATTERNS.get(joint_name, [])

    for pattern in primary + alternatives:
        for col in df_columns:
            if pattern.lower() in col.lower():
                return col
    return None


def _normalize_timeseries(ts: np.ndarray, n_points: int = GAIT_CYCLE_POINTS) -> np.ndarray:
    """Resample time series to fixed number of points via linear interpolation."""
    if len(ts) == n_points:
        return ts
    x_old = np.linspace(0, 100, len(ts))
    x_new = np.linspace(0, 100, n_points)
    return np.interp(x_new, x_old, ts)


def load_xlsx_file(filepath: Path) -> Optional[pd.DataFrame]:
    """Load a single .xlsx file containing joint angle data."""
    try:
        df = pd.read_excel(filepath, engine="openpyxl")
        return df
    except Exception as e:
        logger.warning(f"Failed to load {filepath}: {e}")
        return None


def extract_joint_angles(df: pd.DataFrame, side: str = "right") -> dict:
    """Extract joint angle time series from a DataFrame.

    Returns dict mapping joint names to numpy arrays of angle values
    over the gait cycle.
    """
    angles = {}
    columns = list(df.columns)

    for joint_name in JOINT_COLUMNS:
        col = _find_column(columns, joint_name)
        if col is not None:
            values = df[col].dropna().values.astype(float)
            if len(values) >= 10:
                angles[joint_name] = _normalize_timeseries(values)

    return angles


def load_participant(participant_dir: Path) -> list:
    """Load all trials for a single participant.

    Returns list of dicts, each containing:
    - speed: walking speed condition
    - side: left/right
    - angles: dict of joint angle time series
    """
    trials = []

    for speed_dir in participant_dir.iterdir():
        if not speed_dir.is_dir():
            continue

        speed = speed_dir.name.lower()
        xlsx_files = list(speed_dir.glob("*.xlsx"))

        for xlsx_file in xlsx_files:
            df = load_xlsx_file(xlsx_file)
            if df is None:
                continue

            # Determine side from filename
            fname = xlsx_file.stem.lower()
            side = "left" if "left" in fname or "_l_" in fname else "right"

            angles = extract_joint_angles(df, side)
            if angles:
                trials.append({
                    "speed": speed,
                    "side": side,
                    "angles": angles,
                    "source_file": str(xlsx_file),
                })

    return trials


def load_cohort(cohort_name: str, label: int, data_dir: Path = None) -> pd.DataFrame:
    """Load all participants in a cohort.

    Args:
        cohort_name: Directory name of the cohort (e.g., 'healthy_adults')
        label: Binary label (0=healthy, 1=ACL-injured)
        data_dir: Root data directory

    Returns:
        DataFrame with columns: participant_id, cohort, speed, side,
        joint, angle_timeseries, label
    """
    data_dir = data_dir or RAW_DATA_DIR / "compwalk_acl"
    cohort_dir = data_dir / cohort_name

    if not cohort_dir.exists():
        logger.warning(f"Cohort directory not found: {cohort_dir}")
        return pd.DataFrame()

    records = []
    participant_dirs = sorted([d for d in cohort_dir.iterdir() if d.is_dir()])

    for participant_dir in participant_dirs:
        participant_id = participant_dir.name
        trials = load_participant(participant_dir)

        for trial in trials:
            for joint_name, angle_ts in trial["angles"].items():
                records.append({
                    "participant_id": participant_id,
                    "cohort": cohort_name,
                    "speed": trial["speed"],
                    "side": trial["side"],
                    "joint": joint_name,
                    "angle_timeseries": angle_ts,
                    "label": label,
                })

    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} records from {cohort_name} ({len(participant_dirs)} participants)")
    return df


def load_metadata(data_dir: Path = None) -> pd.DataFrame:
    """Load participant metadata from ID.csv."""
    data_dir = data_dir or RAW_DATA_DIR / "compwalk_acl"
    id_file = data_dir / "ID.csv"

    if not id_file.exists():
        logger.warning(f"Metadata file not found: {id_file}")
        return pd.DataFrame()

    return pd.read_csv(id_file)


def build_dataset(data_dir: Path = None) -> pd.DataFrame:
    """Build complete dataset from all cohorts.

    Returns DataFrame with all trials from healthy adults,
    healthy adolescents, and ACLD (pre-surgery) patients.
    ACLR (post-reconstruction) is excluded from training.
    """
    data_dir = data_dir or RAW_DATA_DIR / "compwalk_acl"
    all_data = []

    for cohort_name, label in COMPWALK_COHORTS.items():
        cohort_df = load_cohort(cohort_name, label, data_dir)
        if not cohort_df.empty:
            all_data.append(cohort_df)

    if not all_data:
        logger.error("No data loaded from any cohort.")
        return pd.DataFrame()

    dataset = pd.concat(all_data, ignore_index=True)
    logger.info(
        f"Complete dataset: {len(dataset)} records, "
        f"{dataset['participant_id'].nunique()} participants, "
        f"label distribution: {dataset.groupby('label')['participant_id'].nunique().to_dict()}"
    )
    return dataset
