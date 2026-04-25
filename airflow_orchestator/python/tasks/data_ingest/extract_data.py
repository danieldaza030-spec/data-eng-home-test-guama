"""Domain helpers for downloading and extracting the Iris dataset from Kaggle.

This module is framework-agnostic: it raises only standard Python exceptions
so callers (e.g. Airflow tasks) can translate them into framework-level errors.

Functions:
    download_and_extract_dataset: Download a public Kaggle dataset zip and
        extract one target file to a local directory.
"""

import logging
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from python.tasks.metadata.constants import IrisConstants

logger = logging.getLogger(__name__)

# Expose top-level names kept for backward compatibility with existing imports.
KAGGLE_API_BASE: str = IrisConstants.KAGGLE_API_BASE
IRIS_OWNER: str = IrisConstants.IRIS_OWNER
IRIS_DATASET: str = IrisConstants.IRIS_DATASET
IRIS_FILENAME: str = IrisConstants.IRIS_FILENAME


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def download_and_extract_dataset(
    owner: str,
    dataset: str,
    target_filename: str,
    dest_dir: str | None = None,
) -> str:
    """Download a Kaggle dataset zip and extract a single target file.

    Uses the Kaggle public REST API (``/api/v1/datasets/download``) with HTTP
    Basic Authentication.  The zip archive is written to *dest_dir* and
    removed after the target file has been extracted.

    Args:
        owner: Kaggle dataset owner slug (e.g. ``"uciml"``).
        dataset: Kaggle dataset name slug (e.g. ``"iris"``).
        target_filename: Name of the file to extract from the zip archive
            (e.g. ``"Iris.csv"``).
        dest_dir: Directory where the extracted file will be written.  A
            temporary directory is created automatically when ``None``.

    Returns:
        Absolute path to the extracted file as a string.

    Raises:
        urllib.error.HTTPError: If the Kaggle API returns a non-2xx status
            code (e.g. 404 Not Found).
        urllib.error.URLError: If the host is unreachable or a network timeout
            occurs.
        FileNotFoundError: If *target_filename* is not present in the
            downloaded zip archive.
        zipfile.BadZipFile: If the downloaded file is not a valid zip archive.
    """
    output_dir = Path(dest_dir) if dest_dir else Path(tempfile.mkdtemp(prefix="iris_ingest_"))
    output_dir.mkdir(parents=True, exist_ok=True)

    url = f"{IrisConstants.KAGGLE_API_BASE}/datasets/download/{owner}/{dataset}"
    logger.info("Downloading dataset from: %s", url)

    zip_path = output_dir / f"{dataset}.zip"

    with urllib.request.urlopen(
        url, timeout=IrisConstants.DOWNLOAD_TIMEOUT_S
    ) as response:
        content_length = response.headers.get("Content-Length")
        logger.info(
            "Receiving response — Content-Length: %s byte(s).",
            content_length if content_length else "unknown",
        )
        with open(zip_path, "wb") as zip_file:
            while chunk := response.read(IrisConstants.CHUNK_SIZE):
                zip_file.write(chunk)

    logger.info("Download complete: %s (%d bytes).", zip_path, zip_path.stat().st_size)

    with zipfile.ZipFile(zip_path) as zf:
        available = zf.namelist()
        if target_filename not in available:
            raise FileNotFoundError(
                f"'{target_filename}' was not found in the downloaded archive. "
                f"Available files: {available}"
            )
        zf.extract(target_filename, output_dir)

    zip_path.unlink()
    extracted_path = output_dir / target_filename
    logger.info("Extracted '%s' to: %s", target_filename, extracted_path)

    return str(extracted_path)
