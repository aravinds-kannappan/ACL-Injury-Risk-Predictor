"""Dataset download utilities for ACL Injury Risk Predictor."""

import os
import zipfile
import logging
from pathlib import Path

import requests
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import RAW_DATA_DIR, COMPWALK_ACL_ZENODO_URL, UCI_GAIT_DATASET_ID

logger = logging.getLogger(__name__)


def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> Path:
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=dest_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))

    return dest_path


def download_compwalk_acl(dest_dir: Path = None) -> Path:
    """Download COMPWALK-ACL dataset from Zenodo.

    Dataset: Multi-pace IMU gait kinematics from 92 participants including
    25 healthy adults, 27 healthy adolescents, and 40 ACL-injured patients.
    Published in Nature Scientific Data (2025).

    Reference: https://www.nature.com/articles/s41597-025-06307-8
    """
    dest_dir = dest_dir or RAW_DATA_DIR / "compwalk_acl"
    dest_dir.mkdir(parents=True, exist_ok=True)

    marker_file = dest_dir / ".download_complete"
    if marker_file.exists():
        logger.info("COMPWALK-ACL dataset already downloaded.")
        return dest_dir

    logger.info("Fetching COMPWALK-ACL record metadata from Zenodo...")
    try:
        response = requests.get(COMPWALK_ACL_ZENODO_URL)
        response.raise_for_status()
        record = response.json()

        files = record.get("files", [])
        if not files:
            raise ValueError("No files found in Zenodo record.")

        for file_info in files:
            file_url = file_info["links"]["self"]
            file_name = file_info["key"]
            file_path = dest_dir / file_name

            if file_path.exists():
                logger.info(f"Skipping {file_name} (already exists)")
                continue

            logger.info(f"Downloading {file_name}...")
            download_file(file_url, file_path)

            if file_name.endswith(".zip"):
                logger.info(f"Extracting {file_name}...")
                with zipfile.ZipFile(file_path, "r") as zf:
                    zf.extractall(dest_dir)

        marker_file.touch()
        logger.info("COMPWALK-ACL download complete.")

    except Exception as e:
        logger.warning(f"Automated download failed: {e}")
        logger.info(
            "\nManual download instructions:\n"
            "1. Visit https://zenodo.org/records/14618291\n"
            "2. Download all files\n"
            "3. Extract to: %s\n"
            "4. Expected structure:\n"
            "   compwalk_acl/\n"
            "     ACLD/\n"
            "     ACLR/\n"
            "     healthy_adults/\n"
            "     healthy_adolescents/\n"
            "     ID.csv\n",
            dest_dir,
        )
        raise

    return dest_dir


def download_uci_gait(dest_dir: Path = None) -> Path:
    """Download UCI Multivariate Gait dataset.

    Dataset: 10 subjects, 3 conditions (unbraced, knee brace, ankle brace),
    bilateral joint angles (ankle, knee, hip) over normalized gait cycles.

    Reference: UCI ML Repository, dataset ID 760.
    """
    dest_dir = dest_dir or RAW_DATA_DIR / "uci_gait"
    dest_dir.mkdir(parents=True, exist_ok=True)

    output_file = dest_dir / "uci_gait_data.csv"
    if output_file.exists():
        logger.info("UCI Gait dataset already downloaded.")
        return dest_dir

    logger.info("Fetching UCI Multivariate Gait dataset...")
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=UCI_GAIT_DATASET_ID)
        df = dataset.data.features
        if dataset.data.targets is not None:
            import pandas as pd
            df = pd.concat([df, dataset.data.targets], axis=1)
        df.to_csv(output_file, index=False)
        logger.info("UCI Gait dataset saved.")
    except ImportError:
        logger.info("ucimlrepo not installed. Downloading via API...")
        url = f"https://archive.ics.uci.edu/static/public/{UCI_GAIT_DATASET_ID}/multivariate+gait+data.zip"
        zip_path = dest_dir / "uci_gait.zip"
        download_file(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
        logger.info("UCI Gait dataset extracted.")

    return dest_dir


def check_data_exists() -> dict:
    """Check which datasets are available locally."""
    return {
        "compwalk_acl": (RAW_DATA_DIR / "compwalk_acl" / ".download_complete").exists(),
        "uci_gait": (RAW_DATA_DIR / "uci_gait").exists()
        and any((RAW_DATA_DIR / "uci_gait").iterdir()),
    }


def download_all():
    """Download all datasets."""
    logging.basicConfig(level=logging.INFO)
    status = check_data_exists()

    if not status["compwalk_acl"]:
        download_compwalk_acl()
    if not status["uci_gait"]:
        download_uci_gait()

    logger.info("All datasets ready.")


if __name__ == "__main__":
    download_all()
