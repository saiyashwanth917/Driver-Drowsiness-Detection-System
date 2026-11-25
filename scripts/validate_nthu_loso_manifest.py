from pathlib import Path
import csv
from collections import defaultdict, Counter

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Pillow is missing. Install it using: pip install pillow")

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "dataset_validation"
    / "loso_splits"
    / "nthu_loso_manifest.csv"
)

EXPECTED_SUBJECTS = {"001", "002", "005", "006"}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

FOLDER_LABELS = {
    "notdrowsy": "notdrowsy",
    "sleepyCombination": "drowsy",
    "slowBlinkWithNodding": "drowsy",
    "yawning": "drowsy",
}

# ============================================================
# LOAD MANIFEST
# ============================================================

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

with MANIFEST_PATH.open("r", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

required_columns = {
    "fold",
    "split",
    "subject_id",
    "label",
    "image_path",
}

if not required_columns.issubset(reader.fieldnames or []):
    raise RuntimeError(
        f"Invalid manifest columns. Found: {reader.fieldnames}"
    )

print("=" * 75)
print("NTHU LOSO MANIFEST INTEGRITY AUDIT")
print("=" * 75)
print(f"Manifest rows: {len(rows):,}")

# ============================================================
# VALIDATE RECORDS
# ============================================================

errors = []
warnings = []

fold_records = defaultdict(list)
unique_paths = set()
path_metadata = {}

for row_number, row in enumerate(rows, start=2):

    try:
        fold = int(row["fold"])
    except ValueError:
        errors.append(f"Invalid fold at CSV row {row_number}")
        continue

    split = row["split"].strip().lower()
    subject = row["subject_id"].strip()
    label = row["label"].strip().lower()
    image_path = Path(row["image_path"].strip())

    if split not in {"train", "test"}:
        errors.append(f"Invalid split at row {row_number}: {split}")

    if subject not in EXPECTED_SUBJECTS:
        errors.append(
            f"Unknown subject at row {row_number}: {subject}"
        )

    if label not in {"drowsy", "notdrowsy"}:
        errors.append(
            f"Invalid label at row {row_number}: {label}"
        )

    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        errors.append(
            f"Invalid image extension at row {row_number}: {image_path}"
        )

    if not image_path.is_file():
        errors.append(
            f"Missing image file at row {row_number}: {image_path}"
        )

    # Filename should begin with its subject ID.
    if not image_path.name.startswith(subject + "_"):
        errors.append(
            f"Subject mismatch: {subject} vs {image_path.name}"
        )

    # Verify folder label against manifest label.
    expected_label = FOLDER_LABELS.get(image_path.parent.name)

    if expected_label is None:
        warnings.append(
            f"Unrecognized image folder: {image_path.parent.name}"
        )
    elif expected_label != label:
        errors.append(
            f"Folder/label mismatch at row {row_number}: "
            f"{expected_label} vs {label}"
        )

    fold_records[fold].append(row)

    normalized_path = str(image_path.resolve()).lower()
    unique_paths.add(normalized_path)

    path_metadata[normalized_path] = {
        "path": image_path,
        "subject": subject,
        "label": label,
    }

# ============================================================
# FOLD VALIDATION
# ============================================================

print("\nFOLD VALIDATION")

for fold in sorted(fold_records):

    fold_rows = fold_records[fold]

    train_rows = [r for r in fold_rows if r["split"] == "train"]
    test_rows = [r for r in fold_rows if r["split"] == "test"]

    train_subjects = set(r["subject_id"] for r in train_rows)
    test_subjects = set(r["subject_id"] for r in test_rows)

    train_paths = set(r["image_path"] for r in train_rows)
    test_paths = set(r["image_path"] for r in test_rows)

    train_counts = Counter(r["label"] for r in train_rows)
    test_counts = Counter(r["label"] for r in test_rows)

    print(f"\nFold {fold}")
    print("-" * 50)
    print(f"Train rows      : {len(train_rows):,}")
    print(f"Test rows      : {len(test_rows):,}")
    print(f"Train subjects  : {sorted(train_subjects)}")
    print(f"Test subjects   : {sorted(test_subjects)}")
    print(f"Train classes   : {dict(train_counts)}")
    print(f"Test classes   : {dict(test_counts)}")

    if train_subjects & test_subjects:
        errors.append(f"Subject leakage detected in fold {fold}")

    if len(test_subjects) != 1:
        errors.append(f"Fold {fold} must have exactly one test subject")

    if len(train_subjects) != 3:
        errors.append(f"Fold {fold} must have exactly three train subjects")

    if train_paths & test_paths:
        errors.append(f"Image leakage detected in fold {fold}")

    if not test_counts["drowsy"] or not test_counts["notdrowsy"]:
        errors.append(f"Missing class in fold {fold} test set")

    if len(fold_rows) != 16595:
        errors.append(
            f"Fold {fold} total is not 16,595: {len(fold_rows)}"
        )

# ============================================================
# SUBJECT-TO-FOLD VALIDATION
# ============================================================

print("\nTEST SUBJECT VALIDATION")

for fold in sorted(fold_records):

    test_subjects = {
        r["subject_id"]
        for r in fold_records[fold]
        if r["split"] == "test"
    }

    expected_subject = {
        "001": 1,
        "002": 2,
        "005": 3,
        "006": 4,
    }

    if len(test_subjects) == 1:

        subject = next(iter(test_subjects))

        if expected_subject.get(subject) != fold:
            errors.append(
                f"Incorrect subject/fold assignment: "
                f"subject {subject}, fold {fold}"
            )

        print(f"Fold {fold}: test subject {subject}")

# ============================================================
# DUPLICATE PATH CHECK
# ============================================================

print("\nUNIQUE IMAGE CHECK")
print("-" * 50)

print(f"Unique image paths: {len(unique_paths):,}")

if len(unique_paths) != 16595:
    errors.append(
        f"Expected 16,595 unique images, found {len(unique_paths)}"
    )

# ============================================================
# IMAGE READABILITY CHECK
# ============================================================

print("\nIMAGE READABILITY CHECK")
print("-" * 50)

missing_files = 0
unreadable_files = 0
checked_images = 0

for i, metadata in enumerate(path_metadata.values(), start=1):

    image_path = metadata["path"]

    if not image_path.is_file():
        missing_files += 1
        continue

    try:
        with Image.open(image_path) as img:
            img.verify()

        checked_images += 1

    except Exception as exc:
        unreadable_files += 1
        errors.append(
            f"Unreadable image: {image_path} ({exc})"
        )

    if i % 1000 == 0:
        print(f"Checked {i:,}/{len(path_metadata):,} unique images")

print(f"Readable images : {checked_images:,}")
print(f"Missing files  : {missing_files:,}")
print(f"Unreadable     : {unreadable_files:,}")

# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 75)
print("FINAL AUDIT RESULT")
print("=" * 75)

if warnings:
    print(f"\nWarnings: {len(warnings)}")
    for warning in warnings[:20]:
        print("WARNING:", warning)

if errors:
    print(f"\nFAILED: {len(errors)} error(s)")
    for error in errors[:50]:
        print("ERROR:", error)

    raise SystemExit("\nManifest integrity validation FAILED.")

print("Manifest integrity: PASSED")
print("Fold assignments  : PASSED")
print("Subject isolation : PASSED")
print("Image existence   : PASSED")
print("Image readability : PASSED")
print("Labels           : PASSED")
print("\nALL CHECKS PASSED")