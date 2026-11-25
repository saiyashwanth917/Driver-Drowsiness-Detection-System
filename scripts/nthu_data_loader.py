from pathlib import Path
import os
import re
import csv
import random
from collections import Counter

import tensorflow as tf

from config import nthu_baseline as cfg


# ============================================================
# REPRODUCIBILITY
# ============================================================

tf.keras.utils.set_random_seed(cfg.RANDOM_SEED)


# ============================================================
# MANIFEST
# ============================================================


def read_manifest(manifest_path=None):
    manifest_path = Path(manifest_path or cfg.MANIFEST_PATH)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"LOSO manifest not found: {manifest_path}"
        )

    with manifest_path.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        rows = list(csv.DictReader(file))

    required = {
        "fold", "split", "subject_id", "label", "image_path"
    }

    if not rows:
        raise ValueError("Manifest is empty.")

    if not required.issubset(rows[0].keys()):
        raise ValueError(
            f"Manifest columns are invalid. Found: {rows[0].keys()}"
        )

    # Normalize paths according to the operating system.
    for row in rows:
        raw_path = row["image_path"].strip()

        if os.name == "nt":
            # Native Windows: preserve the original Windows path.
            row["image_path"] = str(Path(raw_path))
        else:
            # WSL/Linux: convert Windows drive paths only in WSL.
            normalized = raw_path.replace("\\", "/")

            is_wsl = (
                "WSL_DISTRO_NAME" in os.environ
                or Path("/proc/sys/fs/binfmt_misc/WSLInterop").exists()
            )

            if is_wsl:
                match = re.match(r"^([A-Za-z]):/(.*)$", normalized)
                if match:
                    drive = match.group(1).lower()
                    normalized = f"/mnt/{drive}/{match.group(2)}"

            row["image_path"] = str(Path(normalized))

    return rows



# ============================================================
# FOLD SELECTION
# ============================================================

def get_fold_records(rows, fold_number):
    fold_rows = [
        row for row in rows
        if int(row["fold"]) == int(fold_number)
    ]

    if not fold_rows:
        raise ValueError(f"Fold {fold_number} not found.")

    train_rows = [
        row for row in fold_rows
        if row["split"].strip().lower() == "train"
    ]

    test_rows = [
        row for row in fold_rows
        if row["split"].strip().lower() == "test"
    ]

    train_subjects = sorted(
        set(row["subject_id"] for row in train_rows)
    )

    test_subjects = sorted(
        set(row["subject_id"] for row in test_rows)
    )

    # Strict leakage checks.
    if set(train_subjects) & set(test_subjects):
        raise RuntimeError(
            f"Subject leakage detected in fold {fold_number}"
        )

    if len(test_subjects) != 1:
        raise RuntimeError(
            f"Fold {fold_number} must contain exactly one test subject."
        )

    if len(train_subjects) != 3:
        raise RuntimeError(
            f"Fold {fold_number} must contain exactly three training subjects."
        )

    return train_rows, test_rows, train_subjects, test_subjects


# ============================================================
# SUBJECT-LEVEL VALIDATION SPLIT
# ============================================================

def create_subject_validation_split(
    train_rows,
    fold_number,
    validation_subject_count=1,
):
    subjects = sorted(
        set(row["subject_id"] for row in train_rows)
    )

    if len(subjects) <= validation_subject_count:
        raise ValueError(
            "Not enough training subjects for subject-level validation."
        )

    # Deterministic subject selection.
    rng = random.Random(cfg.RANDOM_SEED + int(fold_number))
    shuffled_subjects = subjects.copy()
    rng.shuffle(shuffled_subjects)

    validation_subjects = set(
        shuffled_subjects[:validation_subject_count]
    )

    actual_train_subjects = set(subjects) - validation_subjects

    train_split = [
        row for row in train_rows
        if row["subject_id"] in actual_train_subjects
    ]

    validation_split = [
        row for row in train_rows
        if row["subject_id"] in validation_subjects
    ]

    # Leakage check.
    train_ids = set(
        row["subject_id"] for row in train_split
    )
    validation_ids = set(
        row["subject_id"] for row in validation_split
    )

    if train_ids & validation_ids:
        raise RuntimeError("Train/validation subject leakage detected.")

    if not train_split or not validation_split:
        raise RuntimeError("Train or validation split is empty.")

    return (
        train_split,
        validation_split,
        sorted(actual_train_subjects),
        sorted(validation_subjects),
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def decode_and_preprocess(image_path, label):
    image_bytes = tf.io.read_file(image_path)

    image = tf.io.decode_image(
        image_bytes,
        channels=cfg.CHANNELS,
        expand_animations=False,
    )

    image.set_shape([None, None, cfg.CHANNELS])

    image = tf.image.resize(
        image,
        cfg.IMAGE_SIZE,
        method="bilinear",
        antialias=True,
    )

    image = tf.cast(image, tf.float32) / 255.0

    label = tf.cast(label, tf.float32)

    return image, label


# ============================================================
# TRAINING AUGMENTATION
# ============================================================

def augment_training_image(image, label):
    # Mild horizontal flip.
    image = tf.image.random_flip_left_right(image)

    # Mild brightness and contrast changes.
    image = tf.image.random_brightness(
        image,
        max_delta=0.08,
    )

    image = tf.image.random_contrast(
        image,
        lower=0.90,
        upper=1.10,
    )

    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


# ============================================================
# TF.DATA PIPELINE
# ============================================================

def build_dataset(
    records,
    batch_size=None,
    training=False,
    augment=False,
):
    batch_size = batch_size or cfg.BATCH_SIZE

    paths = [
        row["image_path"]
        for row in records
    ]

    labels = [
        cfg.CLASS_TO_INDEX[row["label"].strip().lower()]
        for row in records
    ]

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(records),
            seed=cfg.RANDOM_SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        decode_and_preprocess,
        num_parallel_calls=2,
    )

    if training and augment:
        dataset = dataset.map(
            augment_training_image,
            num_parallel_calls=2,
        )

    dataset = dataset.batch(
        batch_size,
        drop_remainder=False,
    )

    dataset = dataset.prefetch(1)

    return dataset


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(train_records):
    labels = [
        cfg.CLASS_TO_INDEX[row["label"].strip().lower()]
        for row in train_records
    ]

    counts = Counter(labels)
    total = len(labels)
    num_classes = len(cfg.CLASS_TO_INDEX)

    weights = {}

    for class_index in range(num_classes):
        count = counts.get(class_index, 0)

        if count == 0:
            raise ValueError(
                f"Training split is missing class {class_index}"
            )

        weights[class_index] = (
            total / (num_classes * count)
        )

    return weights


# ============================================================
# COMPLETE FOLD PREPARATION
# ============================================================

def load_fold_datasets(
    fold_number,
    manifest_path=None,
    validation_subject_count=None,
    batch_size=None,
    augment=True,
):
    rows = read_manifest(manifest_path)

    (
        fold_train_rows,
        test_rows,
        fold_train_subjects,
        test_subjects,
    ) = get_fold_records(rows, fold_number)

    validation_subject_count = (
        validation_subject_count
        if validation_subject_count is not None
        else cfg.VALIDATION_SUBJECT_COUNT
    )

    (
        train_rows,
        validation_rows,
        train_subjects,
        validation_subjects,
    ) = create_subject_validation_split(
        fold_train_rows,
        fold_number,
        validation_subject_count,
    )

    # Final isolation checks.
    if set(train_subjects) & set(validation_subjects):
        raise RuntimeError("Train/validation leakage.")

    if set(train_subjects) & set(test_subjects):
        raise RuntimeError("Train/test leakage.")

    if set(validation_subjects) & set(test_subjects):
        raise RuntimeError("Validation/test leakage.")

    # Check that every image exists before training.
    for split_name, split_records in [
        ("train", train_rows),
        ("validation", validation_rows),
        ("test", test_rows),
    ]:
        missing = [
            row["image_path"]
            for row in split_records
            if not Path(row["image_path"]).is_file()
        ]

        if missing:
            raise FileNotFoundError(
                f"{split_name} contains {len(missing)} missing images. "
                f"First missing: {missing[0]}"
            )

    train_ds = build_dataset(
        train_rows,
        batch_size=batch_size,
        training=True,
        augment=augment,
    )

    validation_ds = build_dataset(
        validation_rows,
        batch_size=batch_size,
        training=False,
        augment=False,
    )

    test_ds = build_dataset(
        test_rows,
        batch_size=batch_size,
        training=False,
        augment=False,
    )

    class_weights = calculate_class_weights(train_rows)

    metadata = {
        "fold": int(fold_number),
        "train_subjects": train_subjects,
        "validation_subjects": validation_subjects,
        "test_subjects": test_subjects,
        "train_images": len(train_rows),
        "validation_images": len(validation_rows),
        "test_images": len(test_rows),
        "class_weights": class_weights,
    }

    return train_ds, validation_ds, test_ds, metadata