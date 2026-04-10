"""
Shared utility for downloading raw data files from Google Drive.

Both the Jupyter notebooks and the Streamlit app import this module so
that the Drive file IDs only ever need to be updated in one place.

To find a file ID: open the sharing link
    https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing
and copy the FILE_ID portion.
"""

import os
import gdown

# ---------------------------------------------------------------------------
# Update these two values after uploading the files to Google Drive.
# ---------------------------------------------------------------------------
DRIVE_FILE_IDS = {
    "data.csv"  : "1Q1AKjSsG9IhVXmrFbGoX9Yomv9zvzZtl",
    "labels.csv": "1NUihn4bF9fsH3hDgzIlwspqoX1SnLBew",
}


def ensure_data(data_dir: str = "data") -> None:
    """Download raw data files from Google Drive if they are not present locally.

    Parameters
    ----------
    data_dir : str
        Directory where the files should be saved. Defaults to ``"data"``
        (relative to the repository root). Notebooks pass ``"../data"``
        because they run from the ``notebooks/`` subdirectory.
    """
    os.makedirs(data_dir, exist_ok=True)
    for filename, file_id in DRIVE_FILE_IDS.items():
        dest = os.path.join(data_dir, filename)
        if not os.path.exists(dest):
            print(f"Downloading {filename} from Google Drive …")
            gdown.download(id=file_id, output=dest, quiet=False)
        else:
            print(f"Already exists, skipping: {dest}")
