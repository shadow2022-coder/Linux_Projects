#!/usr/bin/env python3

import argparse
from array import array
import gzip
import shutil
import struct
import urllib.request
from pathlib import Path


MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

MNIST_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist"


def download_file(url: str, destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        print(f"Using cached file: {destination}")
        return

    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def read_idx_images(path: Path):
    with gzip.open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected image magic number in {path}: {magic}")

        raw = f.read()
        expected = count * rows * cols
        if len(raw) != expected:
            raise ValueError(
                f"Image payload size mismatch in {path}: expected {expected}, got {len(raw)}"
            )

    images = array("f", (pixel / 255.0 for pixel in raw))
    return images, (count, rows * cols)


def read_idx_labels(path: Path):
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label magic number in {path}: {magic}")

        raw = f.read()
        if len(raw) != count:
            raise ValueError(
                f"Label payload size mismatch in {path}: expected {count}, got {len(raw)}"
            )

    labels = array("f", (label for label in raw))
    return labels, (count,)


def write_matrix(path: Path, data: array, shape) -> None:
    with path.open("wb") as f:
        data.tofile(f)
    print(f"Wrote {path} with shape {shape}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and prepare MNIST for the C digit recognizer."
    )
    parser.add_argument("--output-dir", default="data", help="Directory for generated .mat files")
    parser.add_argument(
        "--download-dir",
        default="downloads",
        help="Directory for downloaded compressed MNIST source files",
    )
    parser.add_argument("--force", action="store_true", help="Re-download and regenerate all files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    download_dir = Path(args.download_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)

    downloaded = {}
    for key, filename in MNIST_FILES.items():
        destination = download_dir / filename
        download_file(f"{MNIST_BASE_URL}/{filename}", destination, args.force)
        downloaded[key] = destination

    train_images, train_images_shape = read_idx_images(downloaded["train_images"])
    train_labels, train_labels_shape = read_idx_labels(downloaded["train_labels"])
    test_images, test_images_shape = read_idx_images(downloaded["test_images"])
    test_labels, test_labels_shape = read_idx_labels(downloaded["test_labels"])

    write_matrix(output_dir / "train_images.mat", train_images, train_images_shape)
    write_matrix(output_dir / "train_labels.mat", train_labels, train_labels_shape)
    write_matrix(output_dir / "test_images.mat", test_images, test_images_shape)
    write_matrix(output_dir / "test_labels.mat", test_labels, test_labels_shape)


if __name__ == "__main__":
    main()
