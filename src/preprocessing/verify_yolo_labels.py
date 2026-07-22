from pathlib import Path

ROOT = Path("data/processed/heritage_yolo")

splits = ["train", "valid", "test"]

missing = 0
empty = 0
invalid = 0
total = 0

print("=" * 60)
print("VERIFYING YOLO LABELS")
print("=" * 60)

for split in splits:

    image_dir = ROOT / split / "images"
    label_dir = ROOT / split / "labels"

    print(f"\n{split.upper()}")

    for img in image_dir.iterdir():

        if img.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        total += 1

        label = label_dir / (img.stem + ".txt")

        if not label.exists():
            missing += 1
            continue

        lines = label.read_text().strip().splitlines()

        if len(lines) == 0:
            empty += 1
            continue

        for line in lines:

            parts = line.split()

            if len(parts) != 5:
                invalid += 1
                continue

            try:
                cls, x, y, w, h = map(float, parts)

                if not (
                    0 <= x <= 1 and
                    0 <= y <= 1 and
                    0 <= w <= 1 and
                    0 <= h <= 1
                ):
                    invalid += 1

            except:
                invalid += 1

print("\n" + "=" * 60)
print(f"Total Images   : {total}")
print(f"Missing Labels : {missing}")
print(f"Empty Labels   : {empty}")
print(f"Invalid Labels : {invalid}")
print("=" * 60)