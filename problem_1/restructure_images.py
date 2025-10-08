#!/usr/bin/env python3
"""Reorganize the Oxford-IIIT Pet dataset into an ImageFolder-compatible layout."""

import argparse
import logging
from pathlib import Path
import shutil

import polars as pl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group images into split/class sub-directories for torchvision.datasets.ImageFolder."
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "images",
        help="Path to the directory that currently stores the flat collection of images.",
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=Path(__file__).resolve().parent / "oxford_pet_split.csv",
        help="CSV file describing dataset split assignments.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Destination directory that will receive breed-organized copies.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect the planned moves without modifying any files.",
    )
    parser.add_argument(
        "--include-mat",
        action="store_true",
        help="If set, move the paired .mat annotations alongside each image.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def main() -> None:
    configure_logging()
    args = parse_args()

    image_root = args.image_root
    split_csv = args.split_csv
    output_root = args.output_root

    logging.info("Preparing to reorganize images in %s", image_root)
    logging.info("Loading split metadata from %s", split_csv)

    split_frame = pl.read_csv(split_csv)
    sorted_sources = sorted(image_root.glob("*.jpg"))
    sorted_annotation_sources = sorted(image_root.glob("*.mat"))
    logging.info("Located %d images at the dataset root.", len(sorted_sources))
    logging.info("Located %d annotation files at the dataset root.", len(sorted_annotation_sources))
    output_root.mkdir(parents=True, exist_ok=True)

    for row in split_frame.iter_rows(named=True):
        image_name = row["image_name"]
        breed_label = row["label"]
        split_name = row["split"]
        is_cat = image_name[0].isupper()
        animal_group = "cat" if is_cat else "dog"
        logging.debug("Image %s identified as %s class (%s split).", image_name, animal_group, split_name)

        destination_dir = output_root / breed_label
        destination_dir.mkdir(parents=True, exist_ok=True)

        source_image = image_root / image_name
        destination_image = destination_dir / image_name

        if destination_image.exists():
            logging.debug("Destination already contains %s; skipping move.", destination_image)
        elif source_image.exists():
            logging.info("Copying %s -> %s", source_image, destination_image)
            if not args.dry_run:
                destination_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(source_image), str(destination_image))
        else:
            logging.warning("Missing image file: %s", source_image)

        if args.include_mat:
            mat_name = image_name.replace(".jpg", ".mat")
            source_mat = image_root / mat_name
            destination_mat = destination_dir / mat_name
            if destination_mat.exists():
                logging.debug("Destination already contains %s; skipping move.", destination_mat)
            elif source_mat.exists():
                logging.info("Copying %s -> %s", source_mat, destination_mat)
                if not args.dry_run:
                    destination_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(source_mat), str(destination_mat))
            else:
                logging.warning("Missing annotation file: %s", source_mat)


if __name__ == "__main__":
    main()
