from pathlib import Path
import csv
import json
import re

# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path(
    r"C:\Users\sowmi\Downloads\NTHU-DDD Dataset\train"
)

OUTPUT_DIR = Path("outputs/dataset_validation/loso_splits")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

SUBJECT_PATTERN = re.compile(r"^(\d{3})_")

FOLDER_LABELS = {
    "notdrowsy": "notdrowsy",
    "sleepyCombination": "drowsy",
    "slowBlinkWithNodding": "drowsy",
    "yawning": "drowsy",
}

# ============================================================
# DISCOVER DATASET
# ============================================================

records = []

for image_path in DATASET_ROOT.rglob("*"):
    if not image_path.is_file():
        continue

    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        continue

    match = SUBJECT_PATTERN.match(image_path.name)

    if not match:
        print(f"WARNING: Could not parse subject: {image_path.name}")
        continue

    subject_id = match.group(1)

    folder_name = image_path.parent.name

    if folder_name not in FOLDER_LABELS:
        print(f"WARNING: Unknown folder: {folder_name}")
        continue

    label = FOLDER_LABELS[folder_name]

    records.append({
        "subject_id": subject_id,
        "label": label,
        "path": str(image_path.resolve()),
    })

if not records:
    raise RuntimeError("No dataset images found. Check DATASET_ROOT.")

subjects = sorted(set(record["subject_id"] for record in records))

print("=" * 70)
print("NTHU-DDD LEAVE-ONE-SUBJECT-OUT SPLIT GENERATOR")
print("=" * 70)

print(f"Total images: {len(records)}")
print(f"Subjects: {subjects}")

# ============================================================
# GENERATE LOSO FOLDS
# ============================================================

all_manifest_rows = []
fold_summaries = []

for fold_number, test_subject in enumerate(subjects, start=1):

    train_subjects = [
        subject for subject in subjects
        if subject != test_subject
    ]

    train_records = [
        record for record in records
        if record["subject_id"] in train_subjects
    ]

    test_records = [
        record for record in records
        if record["subject_id"] == test_subject
    ]

    # Safety checks
    train_ids = set(r["subject_id"] for r in train_records)
    test_ids = set(r["subject_id"] for r in test_records)

    assert train_ids.isdisjoint(test_ids), (
        f"Subject leakage detected in fold {fold_number}"
    )

    assert test_ids == {test_subject}

    # --------------------------------------------------------
    # Create rows
    # --------------------------------------------------------

    for record in train_records:
        all_manifest_rows.append({
            "fold": fold_number,
            "split": "train",
            "subject_id": record["subject_id"],
            "label": record["label"],
            "image_path": record["path"],
        })

    for record in test_records:
        all_manifest_rows.append({
            "fold": fold_number,
            "split": "test",
            "subject_id": record["subject_id"],
            "label": record["label"],
            "image_path": record["path"],
        })

    # --------------------------------------------------------
    # Counts
    # --------------------------------------------------------

    train_drowsy = sum(
        r["label"] == "drowsy" for r in train_records
    )

    train_notdrowsy = sum(
        r["label"] == "notdrowsy" for r in train_records
    )

    test_drowsy = sum(
        r["label"] == "drowsy" for r in test_records
    )

    test_notdrowsy = sum(
        r["label"] == "notdrowsy" for r in test_records
    )

    summary = {
        "fold": fold_number,
        "test_subject": test_subject,
        "train_subjects": train_subjects,
        "train_total": len(train_records),
        "train_drowsy": train_drowsy,
        "train_notdrowsy": train_notdrowsy,
        "test_total": len(test_records),
        "test_drowsy": test_drowsy,
        "test_notdrowsy": test_notdrowsy,
    }

    fold_summaries.append(summary)

    print(f"\nFold {fold_number}")
    print(f"  Test subject : {test_subject}")
    print(f"  Train subjects: {train_subjects}")
    print(f"  Train images : {len(train_records)}")
    print(f"    Drowsy    : {train_drowsy}")
    print(f"    Not drowsy: {train_notdrowsy}")
    print(f"  Test images  : {len(test_records)}")
    print(f"    Drowsy    : {test_drowsy}")
    print(f"    Not drowsy: {test_notdrowsy}")

# ============================================================
# SAVE MANIFEST
# ============================================================

manifest_path = OUTPUT_DIR / "nthu_loso_manifest.csv"

with open(manifest_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "fold",
            "split",
            "subject_id",
            "label",
            "image_path",
        ],
    )

    writer.writeheader()
    writer.writerows(all_manifest_rows)

# ============================================================
# SAVE SUMMARY
# ============================================================

summary_path = OUTPUT_DIR / "nthu_loso_summary.json"

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(fold_summaries, f, indent=4)

# ============================================================
# FINAL VALIDATION
# ============================================================

expected_fold_count = len(subjects)

assert len(fold_summaries) == expected_fold_count

print("\n" + "=" * 70)
print("SPLIT GENERATION COMPLETE")
print("=" * 70)

print(f"Manifest: {manifest_path}")
print(f"Summary : {summary_path}")
print(f"Folds  : {len(fold_summaries)}")

print("\nSubject leakage check: PASSED")
print("Each fold tests on exactly one unseen subject.")