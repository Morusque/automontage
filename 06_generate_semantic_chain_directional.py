# use Python 3.10
"""
Directional semantic chain (text + image)
- Extrapolation with momentum
- Dynamic per-candidate modality gating (evidence-driven)
- Margin-vs-generic (Option B) anti-generic nudge
- Modality completeness term (prefer both text+image; penalize missing image)
- Tiny image guidance along direction
- First hop uses the same scorer (so influences/gating apply immediately)
- Long-term EMA "theme" (text + image) with optional angle-aware pull
- Saves per-step gates into the output JSON
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
START_PHRASE = "On est sur la bonne voie, continue."  # pick any short-but-meaningful line
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
PHRASE_EMBED_NORMALIZE = True

# Directional walk knobs
ALPHA_INIT = 1.2        # slightly larger push along the ray
BETA_MOMENTUM = 0.7     # blend of previous direction vs new step

# Base scoring weights (R-closeness vs angle)
W_TEXT_R      = 0.55
W_IMAGE_R     = 0.55
W_TEXT_ANGLE  = 0.25
W_IMAGE_ANGLE = 0.25

# Prior modality balance (base tilt before dynamic gating)
TEXT_INFLUENCE_BASE  = 0.40
IMAGE_INFLUENCE_BASE = 0.60

# Dynamic gating knobs (evidence -> gate)
E_P = 2.0     # exponent on closeness
E_Q = 1.5     # exponent on novelty (1 - centroid sim)  (was 1.0)
E_R = 1.0     # exponent on margin vs generic
GAMMA = 3.5   # gating sharpness (softmax-like)
GATE_FLOOR = 0.05

# Margin vs generic (Option B)
NEG_POOL_MAX = 2000         # draw negatives from allowed
NEG_COUNT    = 256          # count to average for "generic" centroid
MARGIN_WEIGHT = 0.40        # push away from generic centroid a bit more (was 0.35)

# Modality completeness preferences (always-on)
BOTH_MODALITIES_BONUS   = 0.12
MISSING_TEXT_PENALTY    = 0.04
MISSING_IMAGE_PENALTY   = 0.15   # stronger push away from image-missing
TINY_IMAGE_GUIDE        = 0.03   # tiny floor along image direction when available

# --- Theme (long-term EMA) ---
THEME_LAMBDA = 0.10        # EMA update speed (0.05–0.15 good)
THEME_WEIGHT_TEXT  = 0.10  # bonus toward text theme
THEME_WEIGHT_IMAGE = 0.10  # bonus toward image theme
THEME_WARMUP_STEPS = 4     # don’t use theme until a few picks in
ANGLE_AWARE_THEME  = True  # scale theme pull by agreement with current direction

# Speed knobs
CANDIDATE_SUBSAMPLE = 20000     # 0 to disable subsampling (slower on ~195k)
TOPK_LOG = 0                    # set >0 to print rough top-K debug
PRINT_TEXT_SNIPPET = True

# Misc
SMALL_EPS = 1e-12

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
            s = f.read()
            obj, _ = json.JSONDecoder().raw_decode(s)
            data = obj
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

def snippet(txt: str, n: int = 80) -> str:
    s = (txt or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[:n-3] + "..."

# Theme EMA helper
def ema_update(theme: Optional[np.ndarray], v: Optional[np.ndarray], lam: float) -> Optional[np.ndarray]:
    if v is None:
        return theme
    if theme is None:
        return v.copy()
    u = (1.0 - lam) * theme + lam * v
    return u / (np.linalg.norm(u) + SMALL_EPS)

# ----------------------- Scoring / Gating helpers ----------------------- #
def build_neg_centroids(segs: List[Segment], allow: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Per-step generic 'mean negative' centroids (very cheap)."""
    if allow.size == 0:
        return None, None
    rng = np.random.default_rng()
    base = allow[:min(len(allow), NEG_POOL_MAX)]
    count = min(NEG_COUNT, len(base))
    pick = rng.choice(base, size=count, replace=False) if count > 0 else np.array([], dtype=int)

    neg_T_mean = None
    neg_I_mean = None
    if USE_TEXT and len(pick) > 0:
        rows = [segs[j].text_vec for j in pick if segs[j].text_vec is not None]
        if rows:
            m = np.mean(np.stack(rows), axis=0); neg_T_mean = m / (np.linalg.norm(m)+SMALL_EPS)
    if USE_IMAGE and len(pick) > 0:
        rows = [segs[j].img_vec for j in pick if segs[j].img_vec is not None]
        if rows:
            m = np.mean(np.stack(rows), axis=0); neg_I_mean = m / (np.linalg.norm(m)+SMALL_EPS)
    return neg_T_mean, neg_I_mean

def build_centroids(segs: List[Segment]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Global modality centroids (for novelty proxy)."""
    T_all = np.stack([s.text_vec for s in segs if s.text_vec is not None]) if USE_TEXT else None
    I_all = np.stack([s.img_vec  for s in segs if s.img_vec  is not None]) if USE_IMAGE else None
    text_centroid = None
    image_centroid = None
    if T_all is not None and len(T_all):
        c = T_all.mean(axis=0); text_centroid = c / (np.linalg.norm(c)+SMALL_EPS)
    if I_all is not None and len(I_all):
        c = I_all.mean(axis=0); image_centroid = c / (np.linalg.norm(c)+SMALL_EPS)
    return text_centroid, image_centroid

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
    b: Segment,
    neg_T_mean: Optional[np.ndarray],
    neg_I_mean: Optional[np.ndarray],
    text_centroid: Optional[np.ndarray],
    image_centroid: Optional[np.ndarray],
    T_theme: Optional[np.ndarray],
    I_theme: Optional[np.ndarray],
    theme_steps: int,
    first_hop: bool = False
) -> Tuple[int, float, Optional[Tuple[float, float]]]:
    """Return (best index, best score, gates_for_best)."""

    # Base weights then base-prior tilt
    wtR, wiR = w_text_R, w_img_R
    wtA, wiA = w_text_angle, w_img_angle
    wtR *= TEXT_INFLUENCE_BASE
    wtA *= TEXT_INFLUENCE_BASE
    wiR *= IMAGE_INFLUENCE_BASE
    wiA *= IMAGE_INFLUENCE_BASE

    best_idx = -1
    best_score = -1e9
    best_gates = None

    for i in candidates:
        s = segs[int(i)]
        if not first_hop and (s.video == b.video and s.index == b.index):
            continue

        # --- Evidence (per-candidate) ---
        def _pos(x): return x if x > 0.0 else 0.0

        # Text evidence
        Et = 0.0
        if Rt is not None and s.text_vec is not None:
            clos = _pos(float(np.dot(Rt, s.text_vec))) ** E_P
            nov = 0.0
            if text_centroid is not None:
                nov = max(0.0, 1.0 - float(np.dot(text_centroid, s.text_vec))) ** E_Q
            marg = 0.0
            if neg_T_mean is not None:
                marg = max(0.0, float(np.dot(Rt, s.text_vec)) - float(np.dot(neg_T_mean, s.text_vec))) ** E_R
            Et = clos * (1.0 + marg) * (1.0 + nov)

        # Image evidence
        Ei = 0.0
        if Ri is not None and s.img_vec is not None:
            clos = _pos(float(np.dot(Ri, s.img_vec))) ** E_P
            nov = 0.0
            if image_centroid is not None:
                nov = max(0.0, 1.0 - float(np.dot(image_centroid, s.img_vec))) ** E_Q
            marg = 0.0
            if neg_I_mean is not None:
                marg = max(0.0, float(np.dot(Ri, s.img_vec)) - float(np.dot(neg_I_mean, s.img_vec))) ** E_R
            Ei = clos * (1.0 + marg) * (1.0 + nov)

        # Dynamic gates (softmax-ish) with floors
        Etg = Et ** GAMMA
        Eig = Ei ** GAMMA
        den = (Etg + Eig) if (Etg + Eig) > SMALL_EPS else 1.0
        gate_t = max(GATE_FLOOR, Etg / den)
        gate_i = max(GATE_FLOOR, Eig / den)
        norm = gate_t + gate_i
        gate_t /= norm
        gate_i /= norm

        # Candidate-specific weights after gating
        twR = wtR * gate_t
        iwR = wiR * gate_i
        twA = wtA * gate_t
        iwA = wiA * gate_i

        # --- Score accumulation ---
        score = 0.0
        # R-closeness
        if Rt is not None and s.text_vec is not None:
            score += twR * float(np.dot(Rt, s.text_vec))
        if Ri is not None and s.img_vec is not None:
            score += iwR * float(np.dot(Ri, s.img_vec))

        # Angle/projection
        if d_text is not None and s.text_vec is not None:
            score += twA * float(np.dot(s.text_vec, d_text))
        if d_img is not None and s.img_vec is not None:
            score += iwA * float(np.dot(s.img_vec, d_img))

        # Margin-vs-generic (extra small direct term)
        if Rt is not None and s.text_vec is not None and neg_T_mean is not None:
            score += MARGIN_WEIGHT * (float(np.dot(Rt, s.text_vec)) - float(np.dot(neg_T_mean, s.text_vec)))
        if Ri is not None and s.img_vec is not None and neg_I_mean is not None:
            score += MARGIN_WEIGHT * (float(np.dot(Ri, s.img_vec)) - float(np.dot(neg_I_mean, s.img_vec)))

        # Modality completeness term (always-on)
        has_text  = (s.text_vec is not None)
        has_image = (s.img_vec  is not None)
        if has_text and has_image:
            score += BOTH_MODALITIES_BONUS
        else:
            if not has_text:
                score -= MISSING_TEXT_PENALTY
            if not has_image:
                score -= MISSING_IMAGE_PENALTY

        # Theme attraction (kick in after warmup)
        if theme_steps >= THEME_WARMUP_STEPS:
            if T_theme is not None and s.text_vec is not None:
                t_scale = 1.0
                if ANGLE_AWARE_THEME and d_text is not None:
                    t_scale = max(0.0, float(np.dot(d_text, T_theme)))
                score += t_scale * THEME_WEIGHT_TEXT * float(np.dot(T_theme, s.text_vec))

            if I_theme is not None and s.img_vec is not None:
                i_scale = 1.0
                if ANGLE_AWARE_THEME and d_img is not None:
                    i_scale = max(0.0, float(np.dot(d_img, I_theme)))
                score += i_scale * THEME_WEIGHT_IMAGE * float(np.dot(I_theme, s.img_vec))

        # Tiny image guide along direction (if available)
        if d_img is not None and s.img_vec is not None:
            score += TINY_IMAGE_GUIDE * float(np.dot(s.img_vec, d_img))

        if score > best_score:
            best_score = score
            best_idx = int(i)
            best_gates = (float(gate_t), float(gate_i))

    return best_idx, best_score, best_gates

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

    # Global centroids (novelty proxy)
    text_centroid, image_centroid = build_centroids(segs)

    # Start
    A = pick_start_segment(segs, START_PHRASE)
    print(f"🚀 Start from: {snippet(A.text)}  [{A.video} #{A.index}]")

    used_ids = {(A.video, A.index)}
    used_movies = {A.video} if AVOID_SAME_MOVIE else set()

    # Allowed set
    allow = allowed_indices(segs, used_ids, used_movies)
    if allow.size == 0:
        print("❌ No allowed candidates from start.")
        return

    # First hop uses the SAME scorer (so influences/gating apply immediately)
    Rt0 = A.text_vec
    Ri0 = A.img_vec
    d_text0 = None
    d_img0  = None

    # Optional subsample for speed
    if CANDIDATE_SUBSAMPLE and allow.size > CANDIDATE_SUBSAMPLE:
        rng = np.random.default_rng()  # fresh seed => variability
        allow_1st = rng.choice(allow, size=CANDIDATE_SUBSAMPLE, replace=False)
    else:
        allow_1st = allow

    neg_T_mean, neg_I_mean = build_neg_centroids(segs, allow_1st)

    # No theme yet (we'll init after B); pass None and steps=0
    idxB, _, gatesB = score_candidates(
        segs, allow_1st, Rt0, Ri0, d_text0, d_img0,
        W_TEXT_R, W_IMAGE_R, W_TEXT_ANGLE, W_IMAGE_ANGLE,
        A, neg_T_mean, neg_I_mean, text_centroid, image_centroid,
        T_theme=None, I_theme=None, theme_steps=0,
        first_hop=True
    )
    if idxB < 0:
        print("❌ Could not find a first neighbor.")
        return
    B = segs[idxB]
    used_ids.add((B.video, B.index))
    if AVOID_SAME_MOVIE:
        used_movies.add(B.video)

    # Distances A -> B
    dist_text = None
    dist_image = None
    if USE_TEXT and A.text_vec is not None and B.text_vec is not None:
        dist_text = float(1.0 - np.dot(A.text_vec, B.text_vec))
    if USE_IMAGE and A.img_vec is not None and B.img_vec is not None:
        dist_image = float(1.0 - np.dot(A.img_vec, B.img_vec))
    partsAB = [x for x in [dist_text, dist_image] if x is not None]
    dist_avg = float(np.mean(partsAB)) if partsAB else None

    # Compute initial directions A->B
    d_text, d_img = make_direction(A, B)
    alpha = ALPHA_INIT

    gtB, giB = (gatesB or (None, None))
    if gtB is not None:
        print(f"[first] gates text={gtB:.2f} image={giB:.2f}")
    else:
        print("[first] gates n/a")

    # Initialize themes with B (first accepted after A)
    T_theme = B.text_vec.copy() if B.text_vec is not None else None
    I_theme = B.img_vec.copy()  if B.img_vec  is not None else None
    theme_steps = 1  # we have B

    chain = [{
        "url": A.url, "start": A.start, "stop": A.stop, "text": A.text.strip(),
        "audio_level": A.audio_level, "whisper_model": A.whisper_model,
        "video": A.video, "index": A.index,
        "distance_text": None, "distance_image": None, "distance": None,
        "gate_text": None, "gate_image": None
    }, {
        "url": B.url, "start": B.start, "stop": B.stop, "text": B.text.strip(),
        "audio_level": B.audio_level, "whisper_model": B.whisper_model,
        "video": B.video, "index": B.index,
        "distance_text": (None if dist_text is None else round(dist_text, 6)),
        "distance_image": (None if dist_image is None else round(dist_image, 6)),
        "distance": (None if dist_avg is None else round(dist_avg, 6)),
        "gate_text": (None if gtB is None else round(gtB, 4)),
        "gate_image": (None if giB is None else round(giB, 4))
    }]

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
            allow = rng.choice(allow, size=CANDIDATE_SUBSAMPLE, replace=False)

        # Per-step generic centroids (very cheap)
        neg_T_mean, neg_I_mean = build_neg_centroids(segs, allow)

        # Extrapolated target from current along momentum
        Rt, Ri = extrapolate_target(current, d_text, d_img, alpha)

        # Score candidates
        idxC, best_score, gatesC = score_candidates(
            segs, allow, Rt, Ri, d_text, d_img,
            W_TEXT_R, W_IMAGE_R, W_TEXT_ANGLE, W_IMAGE_ANGLE,
            current, neg_T_mean, neg_I_mean, text_centroid, image_centroid,
            T_theme, I_theme, theme_steps,
            first_hop=False
        )
        if idxC < 0:
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

        gt, gi = (gatesC or (None, None))
        if gt is not None:
            print(f"[step {step:02d}] gates text={gt:.2f} image={gi:.2f}")

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
            "distance": (None if davg is None else round(davg, 6)),
            "gate_text": (None if gt is None else round(gt, 4)),
            "gate_image": (None if gi is None else round(gi, 4))
        })

        # Update momentum directions using blend with (C - current)
        if current.text_vec is not None and C.text_vec is not None:
            step_dir_t = l2_normalize(C.text_vec - current.text_vec)
            d_text = step_dir_t if d_text is None else l2_normalize(BETA_MOMENTUM * d_text + (1.0 - BETA_MOMENTUM) * step_dir_t)

        if current.img_vec is not None and C.img_vec is not None:
            step_dir_i = l2_normalize(C.img_vec - current.img_vec)
            d_img = step_dir_i if d_img is None else l2_normalize(BETA_MOMENTUM * d_img + (1.0 - BETA_MOMENTUM) * step_dir_i)

        # Gentle alpha auto-tune
        if davg is not None:
            if davg < 0.2:
                alpha = min(2.0, alpha * 1.1)
            elif davg > 0.6:
                alpha = max(0.25, alpha * 0.9)

        # Advance
        current = C

        # Update themes (EMA)
        T_theme = ema_update(T_theme, C.text_vec, THEME_LAMBDA)
        I_theme = ema_update(I_theme, C.img_vec,  THEME_LAMBDA)
        theme_steps += 1

        # Optional debug
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
