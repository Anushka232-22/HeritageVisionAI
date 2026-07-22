from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

SOURCE = Path(
    "data/defects/concrete-structural-defect-imaging-dataset/Crack_Dataset"
)

DEST = Path(
    "data/processed/concrete_defects"
)

train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

for class_dir in SOURCE.iterdir():

    if not class_dir.is_dir():
        continue

    images = list(class_dir.glob("*.jpg"))

    train_imgs, temp_imgs = train_test_split(
        images,
        test_size=(1 - train_ratio),
        random_state=42
    )

    val_imgs, test_imgs = train_test_split(
        temp_imgs,
        test_size=0.5,
        random_state=42
    )

    for split_name, split_imgs in {
        "train": train_imgs,
        "val": val_imgs,
        "test": test_imgs
    }.items():

        target = DEST / split_name / class_dir.name
        target.mkdir(parents=True, exist_ok=True)

        for img in split_imgs:
            shutil.copy(img, target / img.name)

print("Dataset split complete!")