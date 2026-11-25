from pathlib import Path
from collections import Counter
import csv
import json
import re

DATASET_ROOT = (
    Path.home() / "Downloads" /
    "NTHU-DDD Dataset full" / "train"
)

OUTPUT_DIR = Path(
    "outputs/dataset_validation/loso_splits_full"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUBJECT_PATTERN = re.compile(r"^(\d{3})_")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

FOLDER_LABELS = {
    "notdrowsy": "notdrowsy",
    "sleepyCombination": "drowsy",
    "slowBlinkWithNodding": "drowsy",
    "yawning": "drowsy",
}

EXPECTED_SUBJECTS = {"001", "002", "005", "006"}

if not DATASET_ROOT.is_dir():
    raise FileNotFoundError(DATASET_ROOT)

records = []

for path in DATASET_ROOT.rglob("*"):
    if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
        continue

    match = SUBJECT_PATTERN.match(path.name)

    if not match:
        raise ValueError(f"Unparsed subject filename: {path}")

    subject = match.group(1)
    folder = path.parent.name

    if folder not in FOLDER_LABELS:
        raise ValueError(f"Unknown class folder: {path}")

    records.append({
        "subject_id": subject,
        "label": FOLDER_LABELS[folder],
        "path": str(path.resolve()),
    })

subjects = {r["subject_id"] for r in records}

if subjects != EXPECTED_SUBJECTS:
    raise ValueError(f"Unexpected subjects: {sorted(subjects)}")

if len(records) != 66381:
    raise ValueError(
        f"Expected 66381 images, found {len(records)}"
    )

manifest_rows = []
summaries = []

for fold, test_subject in enumerate(sorted(subjects), 1):

    train_subjects = subjects - {test_subject}

    train_records = [
        r for r in records
        if r["subject_id"] in train_subjects
    ]

    test_records = [
        r for r in records
        if r["subject_id"] == test_subject
    ]

    assert not (
        {r["subject_id"] for r in train_records}
        & {r["subject_id"] for r in test_records}
    )

    for split, split_records in [
        ("train", train_records),
        ("test", test_records),
    ]:
        for r in split_records:
            manifest_rows.append({
                "fold": fold,
                "split": split,
                "subject_id": r["subject_id"],
                "label": r["label"],
                "image_path": r["path"],
            })

    train_counts = Counter(r["label"] for r in train_records)
    test_counts = Counter(r["label"] for r in test_records)

    summaries.append({
        "fold": fold,
        "test_subject": test_subject,
        "train_subjects": sorted(train_subjects),
        "train_total": len(train_records),
        "train_drowsy": train_counts["drowsy"],
        "train_notdrowsy": train_counts["notdrowsy"],
        "test_total": len(test_records),
        "test_drowsy": test_counts["drowsy"],
        "test_notdrowsy": test_counts["notdrowsy"],
    })

    print(f"\nFOLD {fold}")
    print("Test subject:", test_subject)
    print("Training subjects:", sorted(train_subjects))
    print("Training images:", len(train_records))
    print("Testing images:", len(test_records))

manifest_path = OUTPUT_DIR / "nthu_loso_manifest.csv"
summary_path = OUTPUT_DIR / "nthu_loso_summary.json"

with manifest_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "fold", "split", "subject_id",
            "label", "image_path"
        ],
    )
    writer.writeheader()
    writer.writerows(manifest_rows)

with summary_path.open("w", encoding="utf-8") as f:
    json.dump(summaries, f, indent=4)

print("\n" + "=" * 65)
print("FULL DATASET LOSO GENERATION COMPLETE")
print("=" * 65)
print("Total images:", len(records))
print("Manifest:", manifest_path)
print("Summary:", summary_path)
print("Folds:", len(summaries))
print("Subject leakage: PASSED")
