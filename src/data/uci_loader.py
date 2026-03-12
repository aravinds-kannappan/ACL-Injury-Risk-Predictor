"""UCI Multivariate Gait dataset loader.

Loads and structures the UCI Multivariate Gait dataset (dataset ID 760):
- 10 subjects, 3 conditions (unbraced, knee brace, ankle brace)
- 10 gait cycles per condition per leg
- Joint angles (ankle, knee, hip) over normalized gait cycle (0-100%)

All subjects are healthy, labeled as class 0. The braced conditions
serve as examples of altered gait biomechanics.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import RAW_DATA_DIR, GAIT_CYCLE_POINTS

logger = logging.getLogger(__name__)

# Map UCI joint names to our standard names
UCI_JOINT_MAP = {
    "ankle": "ankle_dorsiflexion",
    "knee": "knee_flexion",
    "hip": "hip_flexion",
}

UCI_CONDITION_MAP = {
    "Barefoot": "unbraced",
    "Knee Brace": "knee_brace",
    "Ankle Brace": "ankle_brace",
}


def load_uci_data(data_dir: Path = None) -> pd.DataFrame:
    """Load UCI Gait dataset from downloaded CSV or raw files.

    Returns DataFrame with columns matching COMPWALK-ACL format:
    participant_id, cohort, speed, side, joint, angle_timeseries, label
    """
    data_dir = data_dir or RAW_DATA_DIR / "uci_gait"

    csv_file = data_dir / "uci_gait_data.csv"
    if csv_file.exists():
        return _load_from_csv(csv_file)

    # Try loading from raw R data files or other formats
    rda_files = list(data_dir.glob("*.rda")) + list(data_dir.glob("*.RData"))
    if rda_files:
        return _load_from_rdata(rda_files[0])

    csv_files = list(data_dir.glob("*.csv"))
    if csv_files:
        return _load_from_csv(csv_files[0])

    logger.error(f"No data files found in {data_dir}")
    return pd.DataFrame()


def _load_from_csv(csv_file: Path) -> pd.DataFrame:
    """Parse UCI gait data from CSV into standardized format."""
    raw = pd.read_csv(csv_file)
    logger.info(f"Loaded UCI CSV with {len(raw)} rows, columns: {list(raw.columns)[:10]}...")

    records = []

    # The UCI dataset structure varies. Handle common formats.
    if "Subject" in raw.columns or "subject" in raw.columns:
        return _parse_long_format(raw)

    # If the data is in wide format (columns are time points)
    return _parse_wide_format(raw)


def _parse_long_format(raw: pd.DataFrame) -> pd.DataFrame:
    """Parse data in long format with subject/condition/joint/time columns."""
    col_map = {c: c.lower().strip() for c in raw.columns}
    raw = raw.rename(columns=col_map)

    records = []
    subject_col = "subject" if "subject" in raw.columns else raw.columns[0]
    condition_col = next((c for c in raw.columns if "condition" in c.lower()), None)
    joint_col = next((c for c in raw.columns if "joint" in c.lower()), None)
    leg_col = next((c for c in raw.columns if "leg" in c.lower() or "side" in c.lower()), None)

    if not all([condition_col, joint_col]):
        # Try alternate parsing
        return _parse_wide_format(raw)

    for (subj, cond, joint, leg), group in raw.groupby(
        [subject_col, condition_col, joint_col, leg_col] if leg_col
        else [subject_col, condition_col, joint_col]
    ):
        angle_cols = [c for c in group.columns if c not in [subject_col, condition_col, joint_col, leg_col, "replication"]]

        # If there are numeric columns representing time points
        angle_values = group[angle_cols].values if len(angle_cols) > 1 else None
        if angle_values is not None and angle_values.shape[1] >= 10:
            # Average across replications
            mean_angles = np.nanmean(angle_values, axis=0)
            if len(mean_angles) != GAIT_CYCLE_POINTS:
                x_old = np.linspace(0, 100, len(mean_angles))
                x_new = np.linspace(0, 100, GAIT_CYCLE_POINTS)
                mean_angles = np.interp(x_new, x_old, mean_angles)

            joint_std = UCI_JOINT_MAP.get(str(joint).lower(), str(joint).lower())
            side = str(leg).lower() if leg_col else "right"

            records.append({
                "participant_id": f"uci_{subj}",
                "cohort": "uci_healthy",
                "speed": UCI_CONDITION_MAP.get(str(cond), str(cond).lower()),
                "side": side if side in ["left", "right"] else "right",
                "joint": joint_std,
                "angle_timeseries": mean_angles,
                "label": 0,
            })

    df = pd.DataFrame(records)
    logger.info(f"Parsed {len(df)} UCI gait records from {df['participant_id'].nunique()} subjects")
    return df


def _parse_wide_format(raw: pd.DataFrame) -> pd.DataFrame:
    """Parse data where columns represent time points or features."""
    records = []
    numeric_cols = raw.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) >= GAIT_CYCLE_POINTS:
        # Each row is likely a gait cycle
        for idx, row in raw.iterrows():
            values = row[numeric_cols].values.astype(float)
            if len(values) >= GAIT_CYCLE_POINTS:
                values = values[:GAIT_CYCLE_POINTS]

            records.append({
                "participant_id": f"uci_{idx // 6}",
                "cohort": "uci_healthy",
                "speed": "normal",
                "side": "right" if (idx % 2) == 0 else "left",
                "joint": "knee_flexion",
                "angle_timeseries": values,
                "label": 0,
            })

    df = pd.DataFrame(records)
    logger.info(f"Parsed {len(df)} UCI gait records (wide format)")
    return df


def _load_from_rdata(rdata_file: Path) -> pd.DataFrame:
    """Load from R data format. Requires rpy2."""
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        pandas2ri.activate()

        ro.r(f'load("{rdata_file}")')
        r_objects = list(ro.r.ls())
        logger.info(f"R objects found: {r_objects}")

        # Try to extract the main data object
        for obj_name in r_objects:
            obj = ro.r[obj_name]
            try:
                df = pandas2ri.rpy2py(obj)
                if isinstance(df, pd.DataFrame):
                    return _parse_long_format(df)
            except Exception:
                continue

    except ImportError:
        logger.warning("rpy2 not installed. Cannot load .rda files directly.")

    return pd.DataFrame()
