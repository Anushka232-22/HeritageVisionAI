from pathlib import Path
import shutil

ROOT = Path("data/heritage/various-materials-from-historic-buildings/MAIN DATASET CRACK DETECTION/MAIN DATASET")

OUTPUT = Path("data/processed/heritage_yolo")

datasets = [
    ("brick","Crack Detection brick.v17i.yolov5pytorch"),
    ("clay","crack detection clay.v12i.yolov5pytorch"),
    ("concrete","crack detection concrete.v10i.yolov5pytorch"),
    ("stone","Crack Detection stone.v12i.yolov5pytorch"),
]

splits = ["train","valid","test"]

for prefix, folder in datasets:

    dataset_root = ROOT / folder

    print(f"\nProcessing {prefix}")

    for split in splits:

        image_dir = dataset_root / split / "images"
        label_dir = dataset_root / split / "labels"

        out_images = OUTPUT / split / "images"
        out_labels = OUTPUT / split / "labels"

        out_images.mkdir(parents=True,exist_ok=True)
        out_labels.mkdir(parents=True,exist_ok=True)

        for image in image_dir.iterdir():

            new_name = f"{prefix}_{image.name}"

            shutil.copy2(
                image,
                out_images / new_name
            )

            label = label_dir / (image.stem + ".txt")

            if label.exists():

                shutil.copy2(
                    label,
                    out_labels / f"{prefix}_{label.name}"
                )

print("\nDONE!")