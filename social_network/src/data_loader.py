import os
import gzip
import urllib.request
import numpy as np
import pandas as pd


DATA_URL = "https://snap.stanford.edu/data/CollegeMsg.txt.gz"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
GZ_PATH = os.path.join(DATA_DIR, "CollegeMsg.txt.gz")
TXT_PATH = os.path.join(DATA_DIR, "CollegeMsg.txt")


def download_dataset():
    """Download the real CollegeMsg dataset from Stanford SNAP."""

    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(GZ_PATH):
        print("CollegeMsg dataset already downloaded.")
        return

    print("Downloading CollegeMsg dataset...")
    urllib.request.urlretrieve(DATA_URL, GZ_PATH)
    print("Download complete.")


def extract_dataset():
    """Extract CollegeMsg.txt from the compressed file."""

    if os.path.exists(TXT_PATH):
        print("CollegeMsg.txt already extracted.")
        return

    print("Extracting dataset...")

    with gzip.open(GZ_PATH, "rb") as gz_file:
        with open(TXT_PATH, "wb") as txt_file:
            txt_file.write(gz_file.read())

    print("Extraction complete.")


def load_college_msg():
    """
    Load CollegeMsg temporal interaction data.

    Format:
        source_user destination_user timestamp
    """

    download_dataset()
    extract_dataset()

    print("Loading CollegeMsg...")

    data = pd.read_csv(
        TXT_PATH,
        sep=r"\s+",
        header=None,
        names=["source", "target", "timestamp"],
        comment="#"
    )

    data["source"] = data["source"].astype(int)
    data["target"] = data["target"].astype(int)
    data["timestamp"] = data["timestamp"].astype(int)

    data = data.sort_values("timestamp").reset_index(drop=True)

    print("\nDataset loaded successfully.")
    print(f"Number of interactions: {len(data):,}")
    print(f"Number of users: {len(set(data['source']) | set(data['target'])):,}")

    start_time = data["timestamp"].min()
    end_time = data["timestamp"].max()

    print(f"Start timestamp: {start_time}")
    print(f"End timestamp:   {end_time}")

    return data


def split_temporal_data(data, ratio=0.5):
    """
    Split the real temporal network into two time periods.

    First period  -> Graph G
    Second period -> Graph H
    """

    split_index = int(len(data) * ratio)

    early_data = data.iloc[:split_index].copy()
    later_data = data.iloc[split_index:].copy()

    print("\nTemporal split:")
    print(f"Early interactions: {len(early_data):,}")
    print(f"Later interactions: {len(later_data):,}")

    return early_data, later_data


def get_common_users(early_data, later_data):
    """Find users appearing in both temporal periods."""

    early_users = set(early_data["source"]) | set(early_data["target"])
    later_users = set(later_data["source"]) | set(later_data["target"])

    common_users = sorted(early_users & later_users)

    print(f"\nUsers in early period:  {len(early_users):,}")
    print(f"Users in later period:  {len(later_users):,}")
    print(f"Common users:           {len(common_users):,}")

    return common_users


if __name__ == "__main__":

    data = load_college_msg()

    early_data, later_data = split_temporal_data(data)

    common_users = get_common_users(
        early_data,
        later_data
    )

    print("\nFirst 10 interactions:")
    print(data.head(10))

    print("\nData loader test completed successfully.")