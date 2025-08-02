# use with Python 3.10 and transformers

import os
import json
import subprocess
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

TRANSCRIPT_FOLDER = "D:/project/archiver/prog/automontage/transcripts"
IMAGE_VECTOR_TAG = "has_image_vectors"
IMAGE_VECTOR_KEY = "image_vector"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def extract_frame(video_path, timestamp, output_path):
    cmd = [
        "ffmpeg", "-y", "-ss", str(timestamp),
        "-i", video_path, "-vframes", "1",
        "-q:v", "2", "-loglevel", "error", output_path
    ]
    subprocess.run(cmd, check=True)

def image_to_vector(img_path):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        vec = model.get_image_features(**inputs)
    vec = vec / vec.norm(p=2, dim=-1, keepdim=True)
    return vec.squeeze().cpu().numpy().tolist()

for fname in tqdm(os.listdir(TRANSCRIPT_FOLDER), desc="🖼️ Processing image vectors"):
    if not fname.endswith(".json") or fname.startswith("000_"):
        continue

    json_path = os.path.join(TRANSCRIPT_FOLDER, fname)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Skip already processed or failed
    tag = data.get(IMAGE_VECTOR_TAG)
    if tag is True or tag == "failed":
        continue

    source = data.get("source_path")
    if not source or not os.path.exists(source):
        continue

    segments = data.get("segments", [])
    temp_img = "temp_frame.jpg"
    updated = False
    total_attempts = 0
    total_success = 0

    for seg in segments:
        if IMAGE_VECTOR_KEY in seg:
            continue
        start, end = seg.get("start"), seg.get("end")
        if start is None or end is None or end <= start:
            continue

        total_attempts += 1
        middle = (start + end) / 2.0
        try:
            extract_frame(source, middle, temp_img)
            vec = image_to_vector(temp_img)
            seg[IMAGE_VECTOR_KEY] = [round(float(x), 6) for x in vec]
            updated = True
            total_success += 1
        except Exception as e:
            print(f"⚠️ Failed to process frame at {middle:.2f}s in {fname}: {e}")

    if total_success > 0:
        data[IMAGE_VECTOR_TAG] = True
        print(f"✅ {fname}: {total_success}/{total_attempts} image vectors extracted.")
    elif total_attempts > 0:
        data[IMAGE_VECTOR_TAG] = "failed"
        print(f"⚠️ {fname}: All image vector attempts failed.")

    if total_attempts > 0:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

print("✅ All files processed.")
