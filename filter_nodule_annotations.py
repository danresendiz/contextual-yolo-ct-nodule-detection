import argparse
import json
from pathlib import Path

IMG_EXTENSIONS = [".jpg", ".png", ".jpeg", ".tif"]


def clean_nodule_dataset(dataset_dir: Path, target_label: str = "nodulo"):
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Directory not found: {dataset_dir}")

    removed_files = 0
    cleaned_files = 0

    for file_path in dataset_dir.iterdir():
        if file_path.suffix.lower() != ".json":
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            continue

        if "shapes" not in data:
            continue

        nodule_shapes = [
            shape for shape in data["shapes"]
            if shape.get("label", "").strip().lower() == target_label.lower()
        ]

        if len(nodule_shapes) == 0:
            file_path.unlink()
            for ext in IMG_EXTENSIONS:
                img_path = dataset_dir / f"{file_path.stem}{ext}"
                if img_path.exists():
                    img_path.unlink()
                    break
            removed_files += 1
        else:
            data["shapes"] = nodule_shapes
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            cleaned_files += 1

    print(f"Empty slices removed: {removed_files} | Nodule annotations cleaned: {cleaned_files}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--target_label", type=str, default="nodulo")
    args = parser.parse_args()
    clean_nodule_dataset(args.dataset_dir, args.target_label)