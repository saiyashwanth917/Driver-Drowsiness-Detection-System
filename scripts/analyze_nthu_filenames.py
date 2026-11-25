from pathlib import Path
import csv
import json
import re
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    r"C:\Users\sowmi\Downloads\NTHU-DDD Dataset\train"
)

OUTPUT_DIR = Path(
    "outputs/dataset_validation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Filename pattern:
# 001_glasses_sleepyCombination_1003_drowsy.jpg
#
# Behavior is greedy so that behavior names containing underscores
# can also be captured if present.

FILENAME_PATTERN = re.compile(
    r"^(?P<subject>\d+)_"
    r"(?P<condition>[^_]+)_"
    r"(?P<behavior>.+)_"
    r"(?P<frame>\d+)_"
    r"(?P<label>notdrowsy|drowsy)$",
    re.IGNORECASE
)

# Folder label mapping
FOLDER_LABELS = {
    "notdrowsy": "notdrowsy",
    "sleepyCombination": "drowsy",
    "slowBlinkWithNodding": "drowsy",
    "yawning": "drowsy",
}


# ============================================================
# HELPERS
# ============================================================

def get_folder_class(image_path: Path) -> str:
    """
    Determine the class based on the image's parent directory.
    """

    parent_name = image_path.parent.name

    if parent_name == "notdrowsy":
        return "notdrowsy"

    if parent_name in FOLDER_LABELS:
        return FOLDER_LABELS[parent_name]

    return "unknown"


def get_behavior_from_folder(image_path: Path) -> str:
    """
    Return the behavior category represented by the folder.
    """

    parent_name = image_path.parent.name

    if parent_name == "notdrowsy":
        return "nonsleepyCombination"

    return parent_name


def parse_filename(filename: str):
    """
    Parse NTHU-DDD filename metadata.
    """

    stem = Path(filename).stem

    match = FILENAME_PATTERN.match(stem)

    if not match:
        return None

    metadata = match.groupdict()

    metadata["subject"] = metadata["subject"].zfill(3)
    metadata["frame"] = int(metadata["frame"])
    metadata["label"] = metadata["label"].lower()

    return metadata


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print("=" * 70)
    print("NTHU-DDD FILENAME ANALYSIS")
    print("=" * 70)

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_ROOT}"
        )

    image_extensions = {
        ".jpg", ".jpeg", ".png", ".bmp"
    }

    images = [
        p for p in DATASET_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in image_extensions
    ]

    print(f"\nTotal images discovered: {len(images)}")

    records = []

    malformed = 0
    label_mismatches = 0

    class_counts = Counter()
    behavior_counts = Counter()
    subject_counts = Counter()
    condition_counts = Counter()

    malformed_files = []
    mismatch_files = []

    for index, image_path in enumerate(images, start=1):

        folder_class = get_folder_class(image_path)
        folder_behavior = get_behavior_from_folder(image_path)

        metadata = parse_filename(image_path.name)

        class_counts[folder_behavior] += 1

        record = {
            "filename": image_path.name,
            "relative_path": str(
                image_path.relative_to(DATASET_ROOT)
            ),
            "folder_behavior": folder_behavior,
            "folder_label": folder_class,
            "parsed": False,
            "subject_id": "",
            "condition": "",
            "filename_behavior": "",
            "frame_number": "",
            "filename_label": "",
            "label_match": "",
            "status": "",
        }

        if metadata is None:

            malformed += 1
            malformed_files.append(image_path.name)

            record["status"] = "malformed_filename"

        else:

            record["parsed"] = True
            record["subject_id"] = metadata["subject"]
            record["condition"] = metadata["condition"]
            record["filename_behavior"] = metadata["behavior"]
            record["frame_number"] = metadata["frame"]
            record["filename_label"] = metadata["label"]

            subject_counts[metadata["subject"]] += 1
            condition_counts[metadata["condition"]] += 1
            behavior_counts[metadata["behavior"]] += 1

            expected_folder_label = folder_class
            actual_filename_label = metadata["label"]

            label_match = (
                expected_folder_label == actual_filename_label
            )

            record["label_match"] = label_match

            if not label_match:

                label_mismatches += 1
                mismatch_files.append({
                    "filename": image_path.name,
                    "folder_label": expected_folder_label,
                    "filename_label": actual_filename_label,
                })

                record["status"] = "label_mismatch"

            else:

                record["status"] = "valid"

        records.append(record)

        if index % 2000 == 0:
            print(f"Processed {index}/{len(images)} images")


    # ========================================================
    # CSV REPORT
    # ========================================================

    csv_path = OUTPUT_DIR / "nthu_filename_analysis.csv"

    fieldnames = list(records[0].keys()) if records else []

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(records)


    # ========================================================
    # JSON SUMMARY
    # ========================================================

    summary = {

        "dataset": "NTHU-DDD",

        "dataset_root": str(DATASET_ROOT),

        "total_images": len(images),

        "parsed_filenames": len(images) - malformed,

        "malformed_filenames": malformed,

        "label_mismatches": label_mismatches,

        "folder_class_counts": dict(class_counts),

        "behavior_counts": dict(behavior_counts),

        "subject_counts": dict(subject_counts),

        "condition_counts": dict(condition_counts),

        "malformed_files": malformed_files,

        "label_mismatch_files": mismatch_files,

    }

    json_path = OUTPUT_DIR / "nthu_filename_analysis.json"

    with open(json_path, "w", encoding="utf-8") as f:

        json.dump(
            summary,
            f,
            indent=4,
            ensure_ascii=False
        )


    # ========================================================
    # CONSOLE REPORT
    # ========================================================

    print("\n" + "=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)

    print(f"Total images:              {len(images)}")
    print(f"Parsed filenames:          {len(images) - malformed}")
    print(f"Malformed filenames:      {malformed}")
    print(f"Label mismatches:         {label_mismatches}")

    print("\nFolder behavior counts:")

    for behavior, count in class_counts.items():
        print(f"  {behavior:<35} {count:>6}")

    print("\nParsed filename behaviors:")

    for behavior, count in behavior_counts.items():
        print(f"  {behavior:<35} {count:>6}")

    print("\nSubject IDs discovered:")

    for subject, count in sorted(subject_counts.items()):
        print(f"  Subject {subject:<10} {count:>6}")

    print("\nCondition distribution:")

    for condition, count in condition_counts.items():
        print(f"  {condition:<20} {count:>6}")

    print("\nReports generated:")

    print(f"  CSV : {csv_path}")
    print(f"  JSON: {json_path}")

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()