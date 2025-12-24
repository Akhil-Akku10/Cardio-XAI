import os
import random
import shutil

SOURCE_DIR = r"C:\Users\akhil\Downloads\archive (8)\chest_xray"
DEST_DIR = r"data\images"

SPLITS = ["train", "val", "test"]
CLASSES = ["NORMAL", "PNEUMONIA"]

SAMPLES_PER_CLASS = {
    "train": 150,
    "val": 30,
    "test": 30
}

random.seed(42)

for split in SPLITS:
    for cls in CLASSES:
        src_path = os.path.join(SOURCE_DIR, split, cls)
        dst_path = os.path.join(DEST_DIR, split, cls.lower())

        os.makedirs(dst_path, exist_ok=True)

        images = [
            f for f in os.listdir(src_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        available = len(images)
        requested = SAMPLES_PER_CLASS[split]
        take = min(available, requested)

        if take == 0:
            print(f"⚠️ No images found in {src_path}")
            continue

        selected = random.sample(images, take)

        for img in selected:
            shutil.copy(
                os.path.join(src_path, img),
                os.path.join(dst_path, img)
            )

        print(f"Copied {take} images from {split}/{cls}")

print("✅ Dataset subset created successfully!")
