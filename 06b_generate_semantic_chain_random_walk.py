
# use Python 3.10
"""
Random-walk next-segment search — CUMULATIVE per-index mutations (no boosts / no max-tries)
Your intended behavior:
- Start at segment A.
- Build synthetic target R by mutating K indices (text/image as configured).
- Compute baseline_d = dist(R, A). Find nearest valid B to R. Accept if dist(R,B) < dist(R,A).
- If not accepted, DO NOT reset R — keep mutating R cumulatively by adding K new indices each attempt
  (until all indices have been mutated; then continue by choosing random indices).
- Repeat until a B is accepted, then continue from B for the next link.
- Never reuse a segment; never reuse a movie (if AVOID_SAME_MOVIE).
"""

import os
import json
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ---------------------------- Config ---------------------------- #
TRANSCRIPT_FOLDER = "D:/project/archiver/prog/automontage/transcripts"
INDEX_FILE = os.path.join(TRANSCRIPT_FOLDER, "000_index_all_files.json")
OUTPUT_FOLDER = "D:/project/archiver/prog/automontage/semantic_chains"

CHAIN_LENGTH = 50
AVOID_SAME_MOVIE = True

# Modalities & keys
USE_TEXT = True
USE_IMAGE = True
TEXT_KEY = "vector"
IMAGE_KEY = "image_vector"

# Start
START_PHRASE = "Un homme à la mer!"  # set to None to start from a random usable segment
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(MODEL_NAME) if START_PHRASE else None

# Pool filter
ALLOWED_MODELS = {"small", "medium", "large"}

# --- Proposal (per-index mutation) parameters --- #
SIGMA = 0.15                    # fixed std for replacement values (stronger default so R moves away from A)
RNG = np.random.default_rng(12345)

# Mutation strategy
MUTATE_BOTH_MODALITIES = True   # if True and both exist, mutate both cumulatively each attempt; else mutate ONE picked modality
TEXT_MUTATION_K = 50            # indices to replace in text vector per attempt (≈15% if 300-dim; tune as needed)
IMAGE_MUTATION_K = 100           # indices to replace in image vector per attempt
MODALITY_PICK_PROBS = {"text": 0.5, "image": 0.5}  # used when not mutating both

# Replacement distribution: "gaussian" or "uniform"
REPLACEMENT_DIST = "gaussian"   # gaussian mean=0, std=SIGMA; uniform has matched variance
CLIP_MIN, CLIP_MAX = -1.0, 1.0

# Diagnostics
LOG_EVERY_ATTEMPTS = 10         # print attempt logs more often for cumulative mode
PRINT_TEXT_SNIPPET = True

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------------------- Utils ---------------------------- #
def normalize_path(p: str) -> str:
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
            text_vec = np.array(seg.get(TEXT_KEY), dtype=np.float32) if seg.get(TEXT_KEY) is not None else None
            img_vec = np.array(seg.get(IMAGE_KEY), dtype=np.float32) if seg.get(IMAGE_KEY) is not None else None

            if (USE_TEXT and text_vec is not None) or (USE_IMAGE and img_vec is not None):
                all_segments.append({
                    "video": fname,
                    "index": i,
                    "text": seg.get("text", ""),
                    "start": seg.get("start"),
                    "stop": seg.get("end"),
                    "url": source,
                    "text_vector": text_vec,
                    "image_vector": img_vec,
                    "audio_level": seg.get("audio_level"),
                    "whisper_model": data.get("whisper_model"),
                })
    return all_segments

def vector_distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Return 1 - cosine_similarity for two 1D vectors, or 1.0 if either is None."""
    if a is None or b is None:
        return 1.0
    a = a.reshape(1, -1)
    b = b.reshape(1, -1)
    sim = float(cosine_similarity(a, b)[0, 0])
    return 1.0 - sim

def combined_distance_to_target(text_target, img_target, seg):
    """Average distance across available modalities of SEG to the TARGET vectors."""
    parts = []
    if USE_TEXT and seg["text_vector"] is not None and text_target is not None:
        parts.append(vector_distance(text_target, seg["text_vector"]))
    if USE_IMAGE and seg["image_vector"] is not None and img_target is not None:
        parts.append(vector_distance(img_target, seg["image_vector"]))
    if not parts:
        return 1.0
    return sum(parts) / len(parts)

def find_closest_segment(segments, text_target, img_target, forbidden_movies, used_ids):
    """Return (best_seg, best_distance_to_target)."""
    best = None
    best_d = None
    for seg in segments:
        seg_id = (seg["video"], seg["index"])
        if seg_id in used_ids:
            continue
        if AVOID_SAME_MOVIE and seg["video"] in forbidden_movies:
            continue
        d = combined_distance_to_target(text_target, img_target, seg)
        if best_d is None or d < best_d:
            best = seg
            best_d = d
    return best, best_d

def find_start_segment(segments, phrase: str | None):
    if phrase and model:
        candidates = [s for s in segments if s["text_vector"] is not None]
        if candidates:
            vec = model.encode([phrase], normalize_embeddings=True)[0].astype(np.float32)
            best = None
            best_d = None
            for s in candidates:
                d = vector_distance(vec, s["text_vector"])
                if best_d is None or d < best_d:
                    best = s
                    best_d = d
            if best is not None:
                return best
    pool = [s for s in segments if (USE_TEXT and s["text_vector"] is not None) or (USE_IMAGE and s["image_vector"] is not None)]
    return random.choice(pool) if pool else None

# --------------------- Mutation helpers (cumulative) --------------------- #
def draw_replacement_values(size: int) -> np.ndarray:
    if REPLACEMENT_DIST == "gaussian":
        vals = RNG.normal(0.0, SIGMA, size=size)
    else:
        half = SIGMA * np.sqrt(3.0)  # roughly match variance
        vals = RNG.uniform(-half, half, size=size)
    return np.clip(vals, CLIP_MIN, CLIP_MAX).astype(np.float32)

def mutate_cumulative(current_vec: np.ndarray, remaining_indices: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Mutate up to k NEW indices on current_vec, chosen from remaining_indices.
       Returns (updated_vec, updated_remaining, mutated_count_this_round)."""
    if current_vec is None:
        return None, remaining_indices, 0
    if remaining_indices.size == 0:
        # all indices already mutated once; pick k random indices to re-randomize
        n = current_vec.shape[0]
        idx = RNG.choice(n, size=max(1, min(k, n)), replace=False)
        current_vec = current_vec.copy()
        current_vec[idx] = draw_replacement_values(idx.size)
        return current_vec, remaining_indices, int(idx.size)

    m = min(k, remaining_indices.size)
    pick_idx = RNG.choice(remaining_indices.size, size=m, replace=False)
    idx = remaining_indices[pick_idx]
    # remove these from remaining
    mask = np.ones(remaining_indices.shape[0], dtype=bool)
    mask[pick_idx] = False
    remaining_indices = remaining_indices[mask]

    current_vec = current_vec.copy()
    current_vec[idx] = draw_replacement_values(idx.size)
    return current_vec, remaining_indices, int(idx.size)

def pick_modalities_to_mutate(has_text: bool, has_image: bool) -> list[str]:
    """Return which modalities to mutate this attempt."""
    if has_text and has_image:
        if MUTATE_BOTH_MODALITIES:
            return ["text", "image"]
        choices, probs = ["text", "image"], [MODALITY_PICK_PROBS.get("text", 0.5), MODALITY_PICK_PROBS.get("image", 0.5)]
        probs = np.array(probs, dtype=float); s = probs.sum()
        probs = (probs / s) if s > 0 else np.ones_like(probs) / 2
        pick = int(RNG.choice(2, p=probs))
        return [choices[pick]]
    elif has_text:
        return ["text"]
    elif has_image:
        return ["image"]
    return []

def step_next_segment(current_seg, segments, used_movies, used_ids, step_idx: int):
    """CUMULATIVE per-index mutations of R until acceptance. No boosts, no max tries."""
    base_text = current_seg["text_vector"] if USE_TEXT else None
    base_img  = current_seg["image_vector"] if USE_IMAGE else None

    if base_text is None and base_img is None:
        print(f"[step {step_idx:02d}] No modalities available in current segment — cannot proceed.")
        return None, {"reason": "no_modalities"}

    forbidden_movies = used_movies if AVOID_SAME_MOVIE else set()

    # Initialize R with a COPY of A
    R_text = base_text.copy() if base_text is not None else None
    R_img  = base_img.copy()  if base_img  is not None else None

    # Track remaining indices to mutate at least once
    remain_text = np.arange(R_text.shape[0]) if R_text is not None else np.array([], dtype=int)
    remain_img  = np.arange(R_img.shape[0])  if R_img  is not None else np.array([], dtype=int)

    attempt = 0
    while True:
        attempt += 1
        to_mutate = pick_modalities_to_mutate(base_text is not None, base_img is not None)

        # Mutate K NEW indices on R (cumulatively)
        mut_counts = {}
        if "text" in to_mutate and R_text is not None:
            R_text, remain_text, m = mutate_cumulative(R_text, remain_text, TEXT_MUTATION_K)
            mut_counts["text"] = m
        if "image" in to_mutate and R_img is not None:
            R_img, remain_img, m = mutate_cumulative(R_img, remain_img, IMAGE_MUTATION_K)
            mut_counts["image"] = m

        # Distances relative to R
        baseline_d = combined_distance_to_target(R_text, R_img, current_seg)  # dist(R, A)
        cand, cand_d = find_closest_segment(segments, R_text, R_img, forbidden_movies, used_ids)  # best B wrt R

        # Logs
        if attempt % LOG_EVERY_ATTEMPTS == 0:
            tx_dim = None if R_text is None else R_text.shape[0]
            im_dim = None if R_img  is None else R_img.shape[0]
            tx_left = None if R_text is None else remain_text.size
            im_left = None if R_img  is None else remain_img.size
            print(f"[step {step_idx:02d}] Attempt {attempt:04d}: mutated={to_mutate} counts={mut_counts}, "
                  f"baseline_d={baseline_d:.6f}, best_cand_d={(cand_d if cand_d is not None else float('nan')):.6f}, "
                  f"left(text={tx_left}/{tx_dim}, image={im_left}/{im_dim})")
        # Accept?
        if cand is not None and (cand["video"], cand["index"]) != (current_seg["video"], current_seg["index"]):
            if cand_d < baseline_d:
                seg_txt = (cand.get("text") or "").replace("\n", " ").strip()
                print(f"[step {step_idx:02d}]  ACCEPT on attempt {attempt:04d}: mutated={to_mutate}, "
                    f"cand_d={cand_d:.6f} < baseline_d={baseline_d:.6f} "
                    f"-> next=[{cand['video']} #{cand['index']}]  \"{seg_txt}\"")
                return cand, {
                    "mutated": to_mutate,
                    "baseline_d": baseline_d,
                    "candidate_d": cand_d,
                    "attempts": attempt,
                }
        # else: keep looping, R will continue accumulating mutations

def save_chain(chain):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_FOLDER, f"chain_randomwalk_cumul_{timestamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chain, f, ensure_ascii=False, indent=2)
    return out_path

# ---------------------------- Main ---------------------------- #
def main():
    print("📥 Loading index...")
    index = load_index()

    print("📥 Loading segments...")
    segments = load_segments(index)
    print(f"🔍 {len(segments)} usable segments.")

    if len(segments) < CHAIN_LENGTH + 1:
        print("❌ Not enough segments to build a chain.")
        return

    start_seg = find_start_segment(segments, START_PHRASE)
    if start_seg is None:
        print("❌ Could not find a suitable start segment.")
        return

    print(f"🚀 Start from: {start_seg['text'][:80].strip()}  [{start_seg['video']} #{start_seg['index']}]")

    used_ids = {(start_seg["video"], start_seg["index"])}
    used_movies = {start_seg["video"]} if AVOID_SAME_MOVIE else set()

    chain = [{
        "url": start_seg["url"],
        "start": start_seg["start"],
        "stop": start_seg["stop"],
        "text": start_seg["text"].strip(),
        "audio_level": start_seg.get("audio_level"),
        "whisper_model": start_seg.get("whisper_model"),
        "video": start_seg["video"],
        "index": start_seg["index"],
        "distance_text": 0.0,
        "distance_image": 0.0,
        "distance": 0.0
    }]

    current = start_seg

    for step in tqdm(range(1, CHAIN_LENGTH), desc="🔗 Building chain"):
        nxt, meta = step_next_segment(current, segments, used_movies, used_ids, step_idx=step)

        seg_id = (nxt["video"], nxt["index"])
        used_ids.add(seg_id)
        if AVOID_SAME_MOVIE:
            used_movies.add(nxt["video"])

        d_text = vector_distance(current["text_vector"], nxt["text_vector"]) if USE_TEXT else 1.0
        d_img  = vector_distance(current["image_vector"], nxt["image_vector"]) if USE_IMAGE else 1.0
        parts = [d for d in [d_text, d_img] if d is not None]
        d_avg = sum(parts)/len(parts) if parts else 1.0

        chain.append({
            "url": nxt["url"],
            "start": nxt["start"],
            "stop": nxt["stop"],
            "text": nxt["text"].strip(),
            "audio_level": nxt.get("audio_level"),
            "whisper_model": nxt.get("whisper_model"),
            "video": nxt["video"],
            "index": nxt["index"],
            "distance_text": round(d_text, 6),
            "distance_image": round(d_img, 6),
            "distance": round(d_avg, 6)
        })
        current = nxt

    out = save_chain(chain)
    print(f"✅ Saved chain to: {out}")


if __name__ == "__main__":
    main()

