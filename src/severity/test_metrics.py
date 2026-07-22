from pathlib import Path
from crack_metrics import extract_crack_metrics

images = list(
    Path(
        "data/processed/heritage_yolo/test/images"
    ).glob("*.jpg")
)

for img in images[:5]:

    print("\n" + "=" * 50)
    print(img.name)

    result = extract_crack_metrics(str(img))

    for k, v in result.items():
        print(f"{k}: {v}")