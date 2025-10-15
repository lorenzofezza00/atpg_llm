import os
import subprocess

# Crea cartella
os.makedirs("data/coco2014", exist_ok=True)
os.chdir("data/coco2014")

# Link ufficiali COCO 2014
images_train_url = "http://images.cocodataset.org/zips/train2014.zip"
images_val_url = "http://images.cocodataset.org/zips/val2014.zip"
annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"

# Scarica tutto con wget
subprocess.run(["wget", images_train_url])
subprocess.run(["wget", images_val_url])
subprocess.run(["wget", annotations_url])

print("✅ Download completato! Ora estrai gli archivi ZIP.")
