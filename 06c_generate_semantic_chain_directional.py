# use Python 3.10
"""
Directional semantic chain (text + image) — extrapolation with momentum.

Overview
- Start at A (closest to start phrase).
- Hop to nearest valid B.
- Compute modality-wise direction vectors d = normalize(v_B - v_A).
- Extrapolate a synthetic target R = normalize(v_B + alpha * d).
- Pick next C among allowed segments by a score that combines:
    * closeness to R (text & image)
    * projection along the momentum (dot(v_C, d))
  (never reuse a segment or a movie)
- Update momentum with a blend: d <- normalize(beta * d + (1-beta) * normalize(v_C - v_B)).
- Repeat until CHAIN_LENGTH.

Notes
- We keep this fast with optional random subsampling of candidates per step.
- We don't modify any transcript files; we only read them and output a chain JSON.
"""

import os
import json
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ---------------------------- Config ---------------------------- #
TRANSCRIPT_FOLDER = "D:/project/archiver/prog/automontage/transcripts"
INDEX_FILE = os.path.join(TRANSCRIPT_FOLDER, "000_index_all_files.json")
OUTPUT_FOLDER = "D:/project/archiver/prog/automontage/semantic_chains"

# Chain + constraints
CHAIN_LENGTH = 50
AVOID_SAME_MOVIE = True

# Modalities & keys
USE_TEXT = True
USE_IMAGE = True
TEXT_KEY = "vector"
IMAGE_KEY = "image_vector"

# Start
START_PHRASE = "C'est à moi que tu parles?"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # only for start phrase retrieval
PHRASE_EMBED_NORMALIZE = True

# Directional walk knobs
ALPHA_INIT = 1.0        # step scale along the direction ray
BETA_MOMENTUM = 0.7     # momentum blend for direction update
RESAMPLE_SUBSAMPLE_EVERY = 25  # resample candidate subsample every N steps (set 0 to disable)

# Scoring weights
W_TEXT_R      = 0.55    # weight for text closeness to extrapolated target R
W_IMAGE_R     = 0.55    # weight for image closeness to extrapolated target R
W_TEXT_ANGLE  = 0.25    # weight for text projection along direction d
W_IMAGE_ANGLE = 0.25    # weight for image projection along direction d

# If a modality missing for a candidate, its terms are skipped; remaining weights are re-normalized.
SMALL_EPS = 1e-12

# Speed knobs
CANDIDATE_SUBSAMPLE = 20000        # 0 to disable subsampling (may be slow on 195k)
TOPK_LOG = 8                       # set 0 to disable top-K debug
PRINT_TEXT_SNIPPET = True          # include a small text snippet in logs

# Output
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------------------- Data structures ---------------------------- #
@dataclass
class Segment:
    video: str
    index: int
    text: str
    start: float
    stop: float
    url: str
    text_vec: Optional[np.ndarray]  # L2-normalized or None
    img_vec: Optional[np.ndarray]   # L2-normalized or None
    audio_level: Optional[float]
    whisper_model: Optional[str]

# ---------------------------- Utils ---------------------------- #
def normalize_path(p: str) -> str:
    return os.path.normpath(p).replace("\\", "/")

def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / (n + SMALL_EPS)

def load_index():
    if not os.path.exists(INDEX_FILE):
        return {}
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            # Loose read: take first root object
            s = f.read()
            obj, _ = json.JSONDecoder().raw_decode(s)
            data = obj
    # normalize values as paths
    return {k: normalize_path(v) for k, v in data.items()}

def load_segments(index_map) -> List[Segment]:
    segs: List[Segment] = []
    for fname in os.listdir(TRANSCRIPT_FOLDER):
        if not fname.endswith(".json") or fname.startswith("000_"):
            continue

        path = os.path.join(TRANSCRIPT_FOLDER, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    s = f.read()
                obj, _ = json.JSONDecoder().raw_decode(s)
                data = obj
                print(f"⚠️  Loose JSON in {path} — parsed first object.")
            except Exception as e2:
                print(f"❌  Skip malformed JSON: {path} ({e2})")
                continue

        if data.get("whisper_model") not in {"small", "medium", "large"}:
            continue

        source = normalize_path(data.get("source_path", index_map.get(fname)))
        if not source:
            continue

        for i, seg in enumerate(data.get("segments", [])):
            text_vec = None
            img_vec = None
            if USE_TEXT and seg.get(TEXT_KEY) is not None:
                tv = np.asarray(seg.get(TEXT_KEY), dtype=np.float32)
                text_vec = l2_normalize(tv)
            if USE_IMAGE and seg.get(IMAGE_KEY) is not None:
                iv = np.asarray(seg.get(IMAGE_KEY), dtype=np.float32)
                img_vec = l2_normalize(iv)

            if (text_vec is None and img_vec is None):
                continue

            segs.append(Segment(
                video=fname,
                index=i,
                text=(seg.get("text") or ""),
                start=float(seg.get("start", 0.0) or 0.0),
                stop=float(seg.get("end", 0.0) or 0.0),
                url=source,
                text_vec=text_vec,
                img_vec=img_vec,
                audio_level=seg.get("audio_level"),
                whisper_model=data.get("whisper_model"),
            ))
    return segs

def encode_phrase(phrase: str) -> Optional[np.ndarray]:
    if not phrase:
        return None
    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode([phrase], normalize_embeddings=PHRASE_EMBED_NORMALIZE)[0].astype(np.float32)
    if not PHRASE_EMBED_NORMALIZE:
        emb = l2_normalize(emb)
    return emb

def cosine_sim(u: Optional[np.ndarray], v: Optional[np.ndarray]) -> Optional[float]:
    if u is None or v is None:
        return None
    return float(np.dot(u, v))

def pick_start_segment(segs: List[Segment], phrase: Optional[str]) -> Segment:
    if phrase:
        q = encode_phrase(phrase)
        # search only among segments with text vectors
        candidates = [s for s in segs if s.text_vec is not None]
        if q is not None and candidates:
            best = None
            best_sim = -1.0
            for s in candidates:
                sim = cosine_sim(q, s.text_vec)
                if sim is not None and sim > best_sim:
                    best = s
                    best_sim = sim
            if best:
                return best
    # fallback random
    return random.choice(segs)

def allowed_indices(segs: List[Segment], used_ids: set, used_movies: set) -> np.ndarray:
    idxs = []
    for i, s in enumerate(segs):
        if (s.video, s.index) in used_ids:
            continue
        if AVOID_SAME_MOVIE and s.video in used_movies:
            continue
        idxs.append(i)
    return np.asarray(idxs, dtype=np.int32)

def nearest_to_segment(base: Segment, segs: List[Segment], allowed: np.ndarray) -> int:
    """Return index into segs for nearest neighbor to base (avg of modalities)."""
    best_i = -1
    best_score = -1e9
    for i in allowed:
        if segs[i].video == base.video and segs[i].index == base.index:
            continue
        score_parts = []
        # similarity, not distance
        if base.text_vec is not None and segs[i].text_vec is not None:
            score_parts.append(np.dot(base.text_vec, segs[i].text_vec))
        if base.img_vec is not None and segs[i].img_vec is not None:
            score_parts.append(np.dot(base.img_vec, segs[i].img_vec))
        if not score_parts:
            continue
        score = float(np.mean(score_parts))
        if score > best_score:
            best_score = score
            best_i = int(i)
    return best_i

def l2n_opt(v: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if v is None:
        return None
    return l2_normalize(v)

def make_direction(a: Segment, b: Segment) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    dt = None
    di = None
    if a.text_vec is not None and b.text_vec is not None:
        dt = l2_normalize(b.text_vec - a.text_vec)
    if a.img_vec is not None and b.img_vec is not None:
        di = l2_normalize(b.img_vec - a.img_vec)
    return dt, di

def extrapolate_target(b: Segment, d_text: Optional[np.ndarray], d_img: Optional[np.ndarray], alpha: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    Rt = None
    Ri = None
    if d_text is not None and b.text_vec is not None:
        Rt = l2_normalize(b.text_vec + alpha * d_text)
    if d_img is not None and b.img_vec is not None:
        Ri = l2_normalize(b.img_vec + alpha * d_img)
    return Rt, Ri

def renormalize_weights(pairs):
    vals = [w if flag else 0.0 for (w, flag) in pairs]
    s = sum(vals)
    if s <= 0:
        return [0.0 for _ in vals]
    return [v / s for v in vals]

def score_candidates(
    segs: List[Segment],
    candidates: np.ndarray,
    Rt: Optional[np.ndarray],
    Ri: Optional[np.ndarray],
    d_text: Optional[np.ndarray],
    d_img: Optional[np.ndarray],
    w_text_R: float,
    w_img_R: float,
    w_text_angle: float,
    w_img_angle: float,
    b: Segment
) -> Tuple[int, float]:
    """Return best candidate index (absolute index into segs) and score."""
    wtR, wiR = w_text_R, w_img_R
    wtA, wiA = w_text_angle, w_img_angle
    pairs = [
        (wtR, Rt is not None),
        (wiR, Ri is not None),
        (wtA, d_text is not None),
        (wiA, d_img is not None),
    ]
    wtR, wiR, wtA, wiA = renormalize_weights(pairs)

    proj_B_text = float(np.dot(b.text_vec, d_text)) if (b.text_vec is not None and d_text is not None) else None
    proj_B_img  = float(np.dot(b.img_vec,  d_img))  if (b.img_vec  is not None and d_img  is not None) else None

    best_idx = -1
    best_score = -1e9

    for i in candidates:
        s = segs[int(i)]
        if s.video == b.video and s.index == b.index:
            continue

        score = 0.0
        if Rt is not None and s.text_vec is not None:
            score += wtR * float(np.dot(Rt, s.text_vec))
        if Ri is not None and s.img_vec is not None:
            score += wiR * float(np.dot(Ri, s.img_vec))

        if d_text is not None and s.text_vec is not None:
            proj = float(np.dot(s.text_vec, d_text))
            score += wtA * proj
        if d_img is not None and s.img_vec is not None:
            proj = float(np.dot(s.img_vec, d_img))
            score += wiA * proj

        if score > best_score:
            best_score = score
            best_idx = int(i)

    return best_idx, best_score

def snippet(txt: str, n: int = 80) -> str:
    s = (txt or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[:n-3] + "..."

def log_topk(segs: List[Segment], cand_idxs: np.ndarray, Rt, Ri, d_text, d_img, k: int):
    rows = []
    for i in cand_idxs:
        s = segs[int(i)]
        score = 0.0
        if Rt is not None and s.text_vec is not None:
            score += float(np.dot(Rt, s.text_vec))
        if Ri is not None and s.img_vec is not None:
            score += float(np.dot(Ri, s.img_vec))
        if d_text is not None and s.text_vec is not None:
            score += 0.5 * float(np.dot(s.text_vec, d_text))
        if d_img is not None and s.img_vec is not None:
            score += 0.5 * float(np.dot(s.img_vec, d_img))
        rows.append((score, s))
    rows.sort(key=lambda x: x[0], reverse=True)
    k = min(k, len(rows))
    print(f"Top-{k} candidates (rough score):")
    for r in rows[:k]:
        s = r[1]
        print(f"  score={r[0]:.4f}  [{s.video} #{s.index}]  \"{snippet(s.text)}\"")

# ---------------------------- Main ---------------------------- #
def main():
    print("📥 Loading index...")
    index_map = load_index()

    print("📥 Loading segments...")
    segs = load_segments(index_map)
    print(f"🔍 {len(segs)} usable segments.")

    if len(segs) < CHAIN_LENGTH + 1:
        print("❌ Not enough segments to build a chain.")
        return

    # Start
    A = pick_start_segment(segs, START_PHRASE)
    print(f"🚀 Start from: {snippet(A.text)}  [{A.video} #{A.index}]")

    used_ids = {(A.video, A.index)}
    used_movies = {A.video} if AVOID_SAME_MOVIE else set()

    # Allowed set and optional subsample
    allow = allowed_indices(segs, used_ids, used_movies)
    if allow.size == 0:
        print("❌ No allowed candidates from start.")
        return

    # First hop: nearest valid neighbor B
    if CANDIDATE_SUBSAMPLE and allow.size > CANDIDATE_SUBSAMPLE:
        allow0 = np.random.default_rng(123).choice(allow, size=CANDIDATE_SUBSAMPLE, replace=False)
    else:
        allow0 = allow
    idxB = nearest_to_segment(A, segs, allow0)
    if idxB < 0:
        print("❌ Could not find a first neighbor.")
        return
    B = segs[idxB]
    used_ids.add((B.video, B.index))
    if AVOID_SAME_MOVIE:
        used_movies.add(B.video)

    # Compute initial directions
    d_text, d_img = make_direction(A, B)
    alpha = ALPHA_INIT

    chain = [{
        "url": A.url, "start": A.start, "stop": A.stop, "text": A.text.strip(),
        "audio_level": A.audio_level, "whisper_model": A.whisper_model,
        "video": A.video, "index": A.index, "distance_text": None, "distance_image": None, "distance": None
    }, {
        "url": B.url, "start": B.start, "stop": B.stop, "text": B.text.strip(),
        "audio_level": B.audio_level, "whisper_model": B.whisper_model,
        "video": B.video, "index": B.index, "distance_text": None, "distance_image": None, "distance": None
    }]

    current_prev = A
    current = B

    rng = np.random.default_rng(42)
    for step in tqdm(range(2, CHAIN_LENGTH), desc="🔗 Building chain"):
        # Allowed set
        allow = allowed_indices(segs, used_ids, used_movies)
        if allow.size == 0:
            print("⛔ Stuck: no more allowed candidates.")
            break

        # Optional subsample
        if CANDIDATE_SUBSAMPLE and allow.size > CANDIDATE_SUBSAMPLE:
            # Resample per step for exploration
            allow = rng.choice(allow, size=CANDIDATE_SUBSAMPLE, replace=False)

        # Extrapolated target from current along momentum
        Rt, Ri = extrapolate_target(current, d_text, d_img, alpha)

        # Score candidates
        idxC, best_score = score_candidates(
            segs, allow, Rt, Ri, d_text, d_img,
            W_TEXT_R, W_IMAGE_R, W_TEXT_ANGLE, W_IMAGE_ANGLE,
            current
        )
        if idxC < 0:
            # Try smaller alpha and retry next loop
            alpha = max(0.25, 0.5 * alpha)
            print(f"[step {step:02d}] No candidate scored; shrinking alpha -> {alpha:.3f} and retrying.")
            continue

        C = segs[idxC]
        print(f"[step {step:02d}] pick -> [{C.video} #{C.index}]  \"{snippet(C.text)}\"")

        # Update used sets
        used_ids.add((C.video, C.index))
        if AVOID_SAME_MOVIE:
            used_movies.add(C.video)

        # Distances from previous (for info)
        dt = None
        di = None
        if USE_TEXT and current.text_vec is not None and C.text_vec is not None:
            dt = float(1.0 - np.dot(current.text_vec, C.text_vec))
        if USE_IMAGE and current.img_vec is not None and C.img_vec is not None:
            di = float(1.0 - np.dot(current.img_vec, C.img_vec))
        dparts = [x for x in [dt, di] if x is not None]
        davg = float(np.mean(dparts)) if dparts else None

        chain.append({
            "url": C.url,
            "start": C.start,
            "stop": C.stop,
            "text": C.text.strip(),
            "audio_level": C.audio_level,
            "whisper_model": C.whisper_model,
            "video": C.video,
            "index": C.index,
            "distance_text": (None if dt is None else round(dt, 6)),
            "distance_image": (None if di is None else round(di, 6)),
            "distance": (None if davg is None else round(davg, 6))
        })

        # Update momentum directions using blend with (C - B)
        if current.text_vec is not None and C.text_vec is not None:
            step_dir_t = l2_normalize(C.text_vec - current.text_vec)
            if d_text is None:
                d_text = step_dir_t
            else:
                d_text = l2_normalize(BETA_MOMENTUM * d_text + (1.0 - BETA_MOMENTUM) * step_dir_t)

        if current.img_vec is not None and C.img_vec is not None:
            step_dir_i = l2_normalize(C.img_vec - current.img_vec)
            if d_img is None:
                d_img = step_dir_i
            else:
                d_img = l2_normalize(BETA_MOMENTUM * d_img + (1.0 - BETA_MOMENTUM) * step_dir_i)

        # Gentle alpha auto-tune
        if davg is not None:
            if davg < 0.2:
                alpha = min(2.0, alpha * 1.1)
            elif davg > 0.6:
                alpha = max(0.25, alpha * 0.9)

        # Advance
        current_prev, current = current, C

        # Optional debug: show a rough top-K around Rt/Ri
        if TOPK_LOG and (step % 10 == 0):
            log_topk(segs, allow[:min(len(allow), 4000)], Rt, Ri, d_text, d_img, k=TOPK_LOG)

    # Save chain
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_FOLDER, f"chain_directional_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chain, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved chain to: {out_path}")


if __name__ == "__main__":
    main()
