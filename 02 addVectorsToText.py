
# use (venv_chain) version 3.10 for this script

import os
import json
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

TRANSCRIPT_FOLDER = "D:/project/archiver/prog/automontage/transcripts"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_KEY = "vector"
VECTOR_TAG = "has_vectors"

model = None

def get_model():
    """Load the embedding model only when required."""
    global model
    if model is None:
        model = SentenceTransformer(MODEL_NAME)
    return model

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
    model = get_model()
    vectors = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    for seg, vec in zip(segments, vectors):
        seg[VECTOR_KEY] = [round(float(x), 6) for x in vec]

    data[VECTOR_TAG] = True  # Add tag

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return True

def collect_files(args):
    """Collect list of transcript files based on CLI arguments."""
    if args.files:
        files = args.files
    else:
        files = [
            f for f in os.listdir(TRANSCRIPT_FOLDER)
            if f.endswith(".json") and not f.startswith("000_")
        ]

    # Convert to absolute paths
    files = [
        f if os.path.isabs(f) else os.path.join(TRANSCRIPT_FOLDER, f)
        for f in files
    ]
    if args.only_missing:
        missing = []
        for path in files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            if not data.get(VECTOR_TAG):
                missing.append(path)
        files = missing

    return files


def main():
    parser = argparse.ArgumentParser(description="Add embedding vectors to transcript files")
    parser.add_argument("--files", nargs="*", help="Specific transcript files to process")
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Process only files missing the 'has_vectors' tag",
    )
    args = parser.parse_args()

    files = collect_files(args)
    if not files:
        print("🔍 No transcript files to process.")
        return

    print(f"🔍 Found {len(files)} transcript files to inspect.")

    updated = 0
    for path in tqdm(files, desc="🧠 Adding vectors"):
        if update_file(path):
            updated += 1

    print(f"\n✅ Done. {updated} file(s) updated.")


if __name__ == "__main__":
    main()
