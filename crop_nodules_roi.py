import argparse
import json
import numpy as np
from PIL import Image
from pathlib import Path


def process_labels(base_dir: Path, output_dir: Path, padding: int = 15):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_files = list(base_dir.glob("*.json"))
    print(f"Processing {len(json_files)} files...")

    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            img_filename = data.get("imagePath", "")
            img_path = base_dir / img_filename

            if not img_path.exists():
                for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG"]:
                    temp_path = base_dir / f"{json_path.stem}{ext}"
                    if temp_path.exists():
                        img_path = temp_path
                        break
                if not img_path.exists():
                    continue

            with Image.open(img_path) as img:
                w, h = img.size
                all_pts = [pt for shape in data.get("shapes", []) for pt in shape.get("points", [])]
                if not all_pts:
                    continue

                pts_array = np.array(all_pts)
                x_min, y_min = np.min(pts_array, axis=0)
                x_max, y_max = np.max(pts_array, axis=0)

                left = max(0, int(x_min - padding))
                top = max(0, int(y_min - padding))
                right = min(w, int(x_max + padding))
                bottom = min(h, int(y_max + padding))

                cropped_img = img.crop((left, top, right, bottom))
                new_img_name = f"{json_path.stem}{img_path.suffix}"
                cropped_img.save(output_dir / new_img_name)

                for shape in data["shapes"]:
                    shape["points"] = [[float(px - left), float(py - top)] for px, py in shape["points"]]

                data["imageHeight"] = cropped_img.height
                data["imageWidth"] = cropped_img.width
                data["imagePath"] = new_img_name
                data["imageData"] = None

            with open(output_dir / json_path.name, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)

        except Exception as e:
            print(f"Error processing {json_path.name}: {e}")

    print(f"Finished. Output saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--padding", type=int, default=15)
    args = parser.parse_args()
    process_labels(args.input_dir, args.output_dir, args.padding)