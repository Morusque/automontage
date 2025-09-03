
# use version 3.10

import os
import json
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

TRANSCRIPT_FOLDER = "D:/project/archiver/prog/automontage/transcripts"
INDEX_FILE = os.path.join(TRANSCRIPT_FOLDER, "000_index_all_files.json")
OUTPUT_FOLDER = "D:/project/archiver/prog/automontage/semantic_chains"

AVOID_SAME_MOVIE = True

CHAIN_LENGTH = 50
TEXT_KEY = "vector"
IMAGE_KEY = "image_vector"
USE_RANDOM_VECTORS = False

USE_TEXT = True
USE_IMAGE = True

START_PHRASE = "C'est la terrible maladie." # None <- to skip this
END_PHRASE = "Je suis malade."   # None <- to skip this
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(MODEL_NAME) if (START_PHRASE or END_PHRASE) else None

ALLOWED_MODELS = {"small", "medium", "large"}
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def normalize_path(p):
    return os.path.normpath(p).replace("\\", "/")

def load_index():
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        return {k: normalize_path(v) for k, v in json.load(f).items()}

def load_segments(index):
    all_segments = []
    for fname in os.listdir(TRANSCRIPT_FOLDER):
        if not fname.endswith(".json") or fname.startswith("000_"):
            continue

        path = os.path.join(TRANSCRIPT_FOLDER, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("whisper_model") not in ALLOWED_MODELS:
            continue

        source = normalize_path(data.get("source_path", index.get(fname)))
        if not source:
            continue

        for i, seg in enumerate(data.get("segments", [])):
            text_vec = np.array(seg[TEXT_KEY], dtype=np.float32) if TEXT_KEY in seg else None
            img_vec = np.array(seg[IMAGE_KEY], dtype=np.float32) if IMAGE_KEY in seg else None
            if (USE_TEXT and text_vec is not None) or (USE_IMAGE and img_vec is not None):
                all_segments.append({
                    "video": fname,
                    "index": i,
                    "text": seg["text"],
                    "start": seg["start"],
                    "stop": seg["end"],
                    "url": source,
                    "text_vector": text_vec,
                    "image_vector": img_vec,
                    "audio_level": seg.get("audio_level"),
                    "whisper_model": data.get("whisper_model")
                })
    return all_segments

def interpolate_vectors(v1, v2, steps):
    return [(1 - t) * v1 + t * v2 for t in np.linspace(0, 1, steps)]

def compute_similarities(vec_text_target, vec_img_target, segments):
    sim_text = np.zeros(len(segments))
    sim_image = np.zeros(len(segments))

    if USE_TEXT and vec_text_target is not None:
        valid_text = [s["text_vector"] for s in segments if s["text_vector"] is not None]
        if valid_text:
            text_vectors = np.stack(valid_text)
            sim_text_full = cosine_similarity(vec_text_target.reshape(1, -1), text_vectors)[0]
            j = 0
            for i, s in enumerate(segments):
                if s["text_vector"] is not None:
                    sim_text[i] = sim_text_full[j]
                    j += 1

    if USE_IMAGE and vec_img_target is not None:
        valid_img = [s["image_vector"] for s in segments if s["image_vector"] is not None]
        if valid_img:
            # Check shape consistency
            shapes = set(tuple(v.shape) for v in valid_img)
            if len(shapes) > 1:
                print(f"❌ Inconsistent image vector shapes: {shapes}")
            else:
                image_vectors = np.stack(valid_img)
                # print(f"🔍 Target image vector shape: {vec_img_target.shape}")
                # print(f"🔍 First image in pool shape: {image_vectors[0].shape}")                
                sim_img_full = cosine_similarity(vec_img_target.reshape(1, -1), image_vectors)[0]
                j = 0
                for i, s in enumerate(segments):
                    if s["image_vector"] is not None:
                        sim_image[i] = sim_img_full[j]
                        j += 1

    # Combine per segment
    sims = []
    for i, s in enumerate(segments):
        has_text = s["text_vector"] is not None
        has_img = s["image_vector"] is not None
        if has_text and has_img:
            sims.append(sim_text[i]/2.0 + sim_image[i]/2.0)
        elif has_text:
            sims.append(sim_text[i]/2.0)
        elif has_img:
            sims.append(sim_image[i]/2.0)
        else:
            sims.append(0.0)

    return np.array(sims), sim_text, sim_image

def find_closest_segment_by_phrase(segments, phrase):
    if model is None or not phrase:
        return None
    valid = [s for s in segments if s["text_vector"] is not None]
    if not valid:
        return None
    vec = model.encode([phrase], normalize_embeddings=True)[0]
    vectors = np.stack([s["text_vector"] for s in valid])
    sims = cosine_similarity(vec.reshape(1, -1), vectors)[0]
    best_idx = int(np.argmax(sims))
    return valid[best_idx]

def generate_chain(all_segments, length, phrase_start=None, phrase_end=None):
    if len(all_segments) < length + 2:
        print("❌ Not enough valid segments.")
        return []

    if USE_RANDOM_VECTORS:
        seg_start = find_closest_segment_by_phrase(all_segments, phrase_start) or random.choice(all_segments)

        # Random end vector must match dimensions
        vec_start_text = seg_start["text_vector"] if USE_TEXT and seg_start["text_vector"] is not None else None
        vec_end_text = np.random.normal(0, 0.05, vec_start_text.shape).clip(-1, 1) if vec_start_text is not None else None

        vec_start_img = seg_start["image_vector"] if USE_IMAGE and seg_start["image_vector"] is not None else None
        vec_end_img = np.random.normal(0, 0.05, vec_start_img.shape).clip(-1, 1) if vec_start_img is not None else None

        seg_end = {"text": "[Random End]", "start": 0.0, "stop": 0.0, "url": "[None]"}
    else:
        seg_start = find_closest_segment_by_phrase(all_segments, phrase_start) or random.choice(all_segments)
        seg_end = find_closest_segment_by_phrase(all_segments, phrase_end) or random.choice(all_segments)
        if seg_end is seg_start:
            candidates = [s for s in all_segments if s is not seg_start]
            seg_end = random.choice(candidates)
        vec_start_text = seg_start["text_vector"] if USE_TEXT else None
        vec_end_text   = seg_end["text_vector"]   if USE_TEXT else None
        vec_start_img  = seg_start["image_vector"] if USE_IMAGE else None
        vec_end_img    = seg_end["image_vector"]   if USE_IMAGE else None

    # Interpolate
    targets_text = interpolate_vectors(vec_start_text, vec_end_text, length) if vec_start_text is not None and vec_end_text is not None else [None] * length
    targets_img  = interpolate_vectors(vec_start_img, vec_end_img, length) if vec_start_img is not None and vec_end_img is not None else [None] * length

    print(f"🔗 Start: {seg_start['text'][:50]}... ({seg_start['start']}s)")
    print(f"🔗 End: {seg_end['text'][:50]}... ({seg_end['start']}s)")
    print(f"🔗 Interpolating {length} steps...")

    used = set()
    used_movies = set() if AVOID_SAME_MOVIE else None
    chain = []

    for t in tqdm(range(length), desc="🔗 Building chain"):
        vec_t_text = targets_text[t]
        vec_t_img  = targets_img[t]
        sims, sim_text, sim_image = compute_similarities(vec_t_text, vec_t_img, all_segments)

        best_idx = np.argsort(sims)[::-1]

        for idx in best_idx:
            seg = all_segments[idx]
            seg_id = (seg["video"], seg["index"])
            if seg_id in used:
                continue
            if AVOID_SAME_MOVIE and seg["video"] in used_movies:
                continue
            used.add(seg_id)
            if AVOID_SAME_MOVIE:
                used_movies.add(seg["video"])
            chain.append({
                "url": seg["url"],
                "start": seg["start"],
                "stop": seg["stop"],
                "text": seg["text"].strip(),
                "audio_level": seg.get("audio_level"),
                "whisper_model": seg.get("whisper_model"),
                "distance_text": round(1.0 - sim_text[idx], 6),
                "distance_image": round(1.0 - sim_image[idx], 6),
                "distance": round(1.0 - sims[idx], 6)
            })
            if not USE_RANDOM_VECTORS and seg_id == (seg_end["video"], seg_end["index"]):
                print("🏁 Reached target segment.")
                return chain
            break

    return chain

def save_chain(chain):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_FOLDER, f"chain_combined_{timestamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chain, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved chain: {out_path}")

def main():
    print("📥 Loading index...")
    index = load_index()

    print("📥 Loading segments...")
    segments = load_segments(index)

    print(f"🔍 {len(segments)} segments available.")
    if len(segments) < CHAIN_LENGTH + 2:
        print("❌ Not enough segments to build a chain.")
        return

    chain = generate_chain(segments, CHAIN_LENGTH, START_PHRASE, END_PHRASE)
    if chain:
        save_chain(chain)
    else:
        print("❌ Chain generation failed or returned empty.")

main()