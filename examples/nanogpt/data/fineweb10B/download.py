"""
Download FineWeb10B GPT-2 pretokenized shards from HuggingFace.

Usage:
    python download.py          # downloads 9 train shards + 1 val shard (default)
    python download.py 103      # downloads all 103 train shards + 1 val shard
    python download.py 3        # downloads 3 train shards + 1 val shard

Shards are downloaded to the same directory as this script.
Each shard is ~200MB (~100M tokens).

Dataset: https://huggingface.co/datasets/kjj0/fineweb10B-gpt2
"""

import os
import sys

from huggingface_hub import hf_hub_download

REPO_ID = "kjj0/fineweb10B-gpt2"
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))


def download_shard(fname):
    local_path = os.path.join(LOCAL_DIR, fname)
    if os.path.exists(local_path):
        print(f"  {fname} already exists, skipping")
        return local_path

    print(f"  downloading {fname}...")
    hf_hub_download(
        repo_id=REPO_ID,
        filename=fname,
        repo_type="dataset",
        local_dir=LOCAL_DIR,
    )
    return local_path


def main():
    num_train_shards = 9  # default: small smoke run
    if len(sys.argv) > 1:
        num_train_shards = int(sys.argv[1])

    print(f"Downloading FineWeb10B shards to {LOCAL_DIR}")
    print(f"  - 1 validation shard")
    print(f"  - {num_train_shards} training shards")
    print()

    # validation shard (always shard 0)
    download_shard("fineweb_val_000000.bin")

    # training shards (1-indexed)
    for i in range(1, num_train_shards + 1):
        download_shard(f"fineweb_train_{i:06d}.bin")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
