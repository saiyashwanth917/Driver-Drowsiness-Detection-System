from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image, ImageOps, ImageDraw
import csv
import random

# ==========================================================
# CONFIGURATION
# ==========================================================

MRL_ROOT = Path(r"C:\Users\sowmi\Downloads\MRL-Eye-Reduced")
NTHU_ROOT = Path(r"C:\Users\sowmi\Downloads\NTHU-DDD Dataset")

PROJECT_ROOT = Path(r"C:\Users\sowmi\Downloads\Hybrid-Driver-Drowsiness")
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "dataset_audit"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SAMPLES_PER_CLASS = 25
GRID_COLUMNS = 5
IMAGE_SIZE = (180, 140)

random.seed(42)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ==========================================================
# HELPERS
# ==========================================================

def collect_images(folder):
    return [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def inspect_image(path):
    try:
        with Image.open(path) as img:
            img.verify()

        with Image.open(path) as img:
            width, height = img.size
            mode = img.mode

        return {
            "valid": True,
            "width": width,
            "height": height,
            "mode": mode,
            "error": ""
        }

    except Exception as exc:
        return {
            "valid": False,
            "width": 0,
            "height": 0,
            "mode": "",
            "error": str(exc)
        }


def create_contact_sheet(image_paths, output_path, title):
    if not image_paths:
        print(f"No images for preview: {title}")
        return

    samples = list(image_paths)

    if len(samples) > SAMPLES_PER_CLASS:
        samples = random.sample(samples, SAMPLES_PER_CLASS)

    cell_width = IMAGE_SIZE[0]
    cell_height = IMAGE_SIZE[1] + 30

    rows = (len(samples) + GRID_COLUMNS - 1) // GRID_COLUMNS

    sheet = Image.new(
        "RGB",
        (GRID_COLUMNS * cell_width, rows * cell_height),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    for index, path in enumerate(samples):
        try:
            with Image.open(path) as img:
                img = ImageOps.exif_transpose(img)
                img = img.convert("RGB")
                img.thumbnail(IMAGE_SIZE)

            x = (index % GRID_COLUMNS) * cell_width
            y = (index // GRID_COLUMNS) * cell_height

            sheet.paste(img, (x, y))

            filename = path.name[:23]
            draw.text(
                (x + 2, y + IMAGE_SIZE[1] + 3),
                filename,
                fill="black"
            )

        except Exception as exc:
            print(f"Preview failed: {path} - {exc}")

    sheet.save(output_path)
    print(f"Preview saved: {output_path}")


# ==========================================================
# DATASET INSPECTION
# ==========================================================

all_records = []
class_images = defaultdict(list)
invalid_images = []
dimension_counts = defaultdict(Counter)
class_counts = Counter()


def inspect_dataset(dataset_name, root):
    print("\n" + "=" * 70)
    print(f"INSPECTING: {dataset_name}")
    print(f"ROOT: {root}")
    print("=" * 70)

    if not root.exists():
        print(f"ERROR: Dataset not found: {root}")
        return

    image_paths = collect_images(root)

    print(f"Total image files: {len(image_paths)}")

    for index, path in enumerate(image_paths, start=1):

        # Relative folder becomes the label
        label = str(path.parent.relative_to(root))

        result = inspect_image(path)

        record = {
            "dataset": dataset_name,
            "label": label,
            "filepath": str(path),
            "filename": path.name,
            "valid": result["valid"],
            "width": result["width"],
            "height": result["height"],
            "mode": result["mode"],
            "error": result["error"]
        }

        all_records.append(record)
        class_counts[(dataset_name, label)] += 1

        if result["valid"]:
            class_images[(dataset_name, label)].append(path)

            dimension_counts[(dataset_name, label)][
                f"{result['width']}x{result['height']}"
            ] += 1

        else:
            invalid_images.append(record)

        if index % 1000 == 0:
            print(f"Processed {index}/{len(image_paths)} images")

    print(f"Completed: {dataset_name}")


# ==========================================================
# RUN INSPECTION
# ==========================================================

inspect_dataset("MRL-Eye", MRL_ROOT)
inspect_dataset("NTHU-DDD", NTHU_ROOT)


# ==========================================================
# SAVE MASTER CSV
# ==========================================================

master_csv = OUTPUT_ROOT / "image_inspection_report.csv"

with open(master_csv, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "dataset",
            "label",
            "filepath",
            "filename",
            "valid",
            "width",
            "height",
            "mode",
            "error"
        ]
    )

    writer.writeheader()
    writer.writerows(all_records)

print(f"\nMaster CSV saved: {master_csv}")


# ==========================================================
# SAVE INVALID IMAGES
# ==========================================================

invalid_csv = OUTPUT_ROOT / "invalid_images.csv"

with open(invalid_csv, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "dataset",
            "label",
            "filepath",
            "filename",
            "valid",
            "width",
            "height",
            "mode",
            "error"
        ]
    )

    writer.writeheader()
    writer.writerows(invalid_images)

print(f"Invalid image report saved: {invalid_csv}")


# ==========================================================
# SUMMARY
# ==========================================================

summary_csv = OUTPUT_ROOT / "dataset_summary.csv"

with open(summary_csv, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Dataset",
        "Class",
        "Total Images",
        "Valid Images",
        "Invalid Images",
        "Dimensions"
    ])

    for (dataset, label), total in sorted(class_counts.items()):

        valid_count = len(class_images[(dataset, label)])
        invalid_count = total - valid_count

        dimensions = dimension_counts[(dataset, label)]

        dimension_string = "; ".join(
            f"{dimension}: {count}"
            for dimension, count in dimensions.most_common()
        )

        writer.writerow([
            dataset,
            label,
            total,
            valid_count,
            invalid_count,
            dimension_string
        ])

print(f"Summary saved: {summary_csv}")


# ==========================================================
# CONTACT SHEETS
# ==========================================================

preview_root = OUTPUT_ROOT / "previews"
preview_root.mkdir(parents=True, exist_ok=True)

for (dataset, label), paths in class_images.items():

    safe_label = label.replace("\\", "_").replace("/", "_")

    output_folder = preview_root / dataset
    output_folder.mkdir(parents=True, exist_ok=True)

    output_file = output_folder / f"{safe_label}.jpg"

    create_contact_sheet(
        paths,
        output_file,
        f"{dataset} - {label}"
    )


# ==========================================================
# FINAL REPORT
# ==========================================================

print("\n" + "=" * 70)
print("DATASET INSPECTION COMPLETE")
print("=" * 70)

print(f"Total images scanned: {len(all_records)}")
print(f"Valid images: {len(all_records) - len(invalid_images)}")
print(f"Invalid images: {len(invalid_images)}")

print(f"\nReports: {OUTPUT_ROOT}")
print(f"Preview images: {preview_root}")