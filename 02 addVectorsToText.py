
# use (venv_chain) version 3.10 for this script

import os
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

TRANSCRIPT_FOLDER = "D:/project/archiver/prog/automontage/transcripts"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_KEY = "vector"
VECTOR_TAG = "has_vectors"

model = SentenceTransformer(MODEL_NAME)

def all_segments_have_vector(segments):
    return all(VECTOR_KEY in seg for seg in segments)

def update_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get(VECTOR_TAG) is True:
        return False  # Skip, already tagged

    segments = data.get("segments", [])
    if not segments:
        return False  # No segments

    if all_segments_have_vector(segments):
        # Just mark the file
        data[VECTOR_TAG] = True
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True

    # Compute and add vectors
    texts = [seg["text"] for seg in segments]
    vectors = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    for seg, vec in zip(segments, vectors):
        seg[VECTOR_KEY] = [round(float(x), 6) for x in vec]

    data[VECTOR_TAG] = True  # Add tag

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return True

def main():
    files = [
        f for f in os.listdir(TRANSCRIPT_FOLDER)
        if f.endswith(".json") and not f.startswith("000_")
    ]
    print(f"🔍 Found {len(files)} transcript files.")

    updated = 0
    for fname in tqdm(files, desc="🧠 Adding vectors"):
        full_path = os.path.join(TRANSCRIPT_FOLDER, fname)
        if update_file(full_path):
            updated += 1

    print(f"\n✅ Done. {updated} file(s) updated.")

if __name__ == "__main__":
    main()
