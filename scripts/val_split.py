import os
import shutil
import random

train_pneu = r"C:\Users\akhil\OneDrive\Desktop\Heart disease\data\Images\val"
val_pneu = r"C:\Users\akhil\OneDrive\Desktop\Heart disease\data\Images\val"

os.makedirs(val_pneu, exist_ok=True)

images = os.listdir(train_pneu)
sample = random.sample(images, min(25,len(images)))  # move 25 images

for img in sample:
    shutil.move(
        os.path.join(train_pneu, img),
        os.path.join(val_pneu, img)
    )

print("Success")