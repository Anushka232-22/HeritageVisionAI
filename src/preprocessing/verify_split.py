from pathlib import Path

DATASET = Path("data/processed/concrete_defects")

splits = ["train", "val", "test"]

total = 0

print("="*60)
print("VERIFYING DATASET")
print("="*60)

for split in splits:

    print(f"\n{split.upper()}")

    split_total = 0

    split_path = DATASET / split

    for cls in sorted(split_path.iterdir()):

        if cls.is_dir():

            count = len(list(cls.glob("*.*")))

            split_total += count

            print(f"{cls.name:<15}: {count}")

    total += split_total

    print("-"*40)
    print(f"{split} Total : {split_total}")

print("\n"+"="*60)
print(f"TOTAL IMAGES : {total}")
print("="*60)