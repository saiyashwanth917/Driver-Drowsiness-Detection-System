from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import csv
import json
import re

# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path(
    r"C:\Users\sowmi\Downloads\NTHU-DDD Dataset\train"
)

OUTPUT_DIR = Path("outputs/dataset_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

SUBJECT_PATTERN = re.compile(r"^(\d{3})_")
CONDITION_PATTERN = re.compile(r"^\d{3}_([^_]+)_")

# Folder name to class mapping
CLASS_MAPPING = {
    "notdrowsy": "notdrowsy",
    "sleepyCombination": "drowsy",
    "slowBlinkWithNodding": "drowsy",
    "yawning": "drowsy",
}


# ============================================================
# HELPERS
# ============================================================

def get_class(path: Path) -> str:

    folder = path.parent.name

    if folder == "notdrowsy":
        return "notdrowsy"

    if folder in CLASS_MAPPING:
        return "drowsy"

    return "unknown"


def get_behavior(path: Path) -> str:

    folder = path.parent.name

    if folder == "notdrowsy":
        return "nonsleepyCombination"

    return folder


def get_subject(filename: str) -> str:

    match = SUBJECT_PATTERN.match(filename)

    if match:
        return match.group(1)

    return "unknown"


def get_condition(filename: str) -> str:

    match = CONDITION_PATTERN.match(filename)

    if match:
        return match.group(1)

    return "unknown"


def sha256_file(path: Path) -> str:

    digest = hashlib.sha256()

    with open(path, "rb") as file:

        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def write_csv(path: Path, rows: list, fields: list):

    with open(path, "w", newline="", encoding="utf-8-sig") as file:

        writer = csv.DictWriter(file, fieldnames=fields)

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("NTHU SUBJECT DISTRIBUTION & DUPLICATE ANALYSIS")
    print("=" * 70)

    images = sorted(
        p for p in DATASET_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    print(f"\nTotal images: {len(images)}")

    subject_class = Counter()
    subject_condition = Counter()
    subject_behavior = Counter()

    hash_groups = defaultdict(list)

    # --------------------------------------------------------
    # Analyze metadata and hashes
    # --------------------------------------------------------

    for i, path in enumerate(images, start=1):

        subject = get_subject(path.name)
        condition = get_condition(path.name)
        class_label = get_class(path)
        behavior = get_behavior(path)

        subject_class[(subject, class_label)] += 1
        subject_condition[(subject, condition)] += 1
        subject_behavior[(subject, behavior)] += 1

        image_hash = sha256_file(path)
        hash_groups[image_hash].append(
            str(path.relative_to(DATASET_ROOT))
        )

        if i % 2000 == 0:
            print(f"Processed {i}/{len(images)} images")


    # --------------------------------------------------------
    # Subject-class report
    # --------------------------------------------------------

    subjects = sorted({
        subject for subject, _ in subject_class
    })

    classes = sorted({
        class_label for _, class_label in subject_class
    })

    subject_class_rows = []

    for subject in subjects:

        row = {"subject_id": subject}

        for class_label in classes:
            row[class_label] = subject_class[
                (subject, class_label)
            ]

        row["total"] = sum(
            subject_class[(subject, class_label)]
            for class_label in classes
        )

        subject_class_rows.append(row)

    write_csv(
        OUTPUT_DIR / "subject_class_distribution.csv",
        subject_class_rows,
        ["subject_id"] + classes + ["total"]
    )


    # --------------------------------------------------------
    # Subject-condition report
    # --------------------------------------------------------

    conditions = sorted({
        condition for _, condition in subject_condition
    })

    subject_condition_rows = []

    for subject in subjects:

        row = {"subject_id": subject}

        for condition in conditions:

            row[condition] = subject_condition[
                (subject, condition)
            ]

        subject_condition_rows.append(row)

    write_csv(
        OUTPUT_DIR / "subject_condition_distribution.csv",
        subject_condition_rows,
        ["subject_id"] + conditions
    )


    # --------------------------------------------------------
    # Exact duplicate report
    # --------------------------------------------------------

    duplicate_rows = []

    for image_hash, paths in hash_groups.items():

        if len(paths) > 1:

            duplicate_rows.append({
                "sha256": image_hash,
                "duplicate_count": len(paths),
                "files": " | ".join(paths),
            })

    write_csv(
        OUTPUT_DIR / "exact_duplicates.csv",
        duplicate_rows,
        ["sha256", "duplicate_count", "files"]
    )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {

        "dataset": "NTHU-DDD",
        "total_images": len(images),
        "unique_subjects": subjects,
        "subject_count": len(subjects),
        "classes": classes,
        "exact_duplicate_groups": len(duplicate_rows),
        "images_in_duplicate_groups": sum(
            row["duplicate_count"]
            for row in duplicate_rows
        ),
        "subject_class_distribution": {
            subject: {
                class_label: subject_class[
                    (subject, class_label)
                ]
                for class_label in classes
            }
            for subject in subjects
        },
        "subject_condition_distribution": {
            subject: {
                condition: subject_condition[
                    (subject, condition)
                ]
                for condition in conditions
            }
            for subject in subjects
        }
    }

    summary_path = (
        OUTPUT_DIR / "dataset_integrity_summary.json"
    )

    with open(summary_path, "w", encoding="utf-8") as file:

        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False
        )


    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SUBJECT CLASS DISTRIBUTION")
    print("=" * 70)

    for subject in subjects:

        print(f"\nSubject {subject}")

        for class_label in classes:

            count = subject_class[
                (subject, class_label)
            ]

            print(f"  {class_label:<15}: {count}")

    print("\n" + "=" * 70)
    print("CONDITION DISTRIBUTION")
    print("=" * 70)

    for subject in subjects:

        print(f"\nSubject {subject}")

        for condition in conditions:

            count = subject_condition[
                (subject, condition)
            ]

            print(f"  {condition:<15}: {count}")

    print("\n" + "=" * 70)
    print("DUPLICATE ANALYSIS")
    print("=" * 70)

    print(f"Exact duplicate groups: {len(duplicate_rows)}")

    print(
        "Images involved in duplicate groups:",

        sum(row["duplicate_count"] for row in duplicate_rows)
    )

    print("\nReports generated:")

    print(OUTPUT_DIR / "subject_class_distribution.csv")
    print(OUTPUT_DIR / "subject_condition_distribution.csv")
    print(OUTPUT_DIR / "exact_duplicates.csv")
    print(summary_path)

    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    main()