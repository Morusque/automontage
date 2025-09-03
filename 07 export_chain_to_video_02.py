
# use version 3.10 for this script

import os
import json
import numpy as np
from moviepy.editor import VideoFileClip, CompositeVideoClip, ColorClip, concatenate_videoclips
from moviepy.audio.fx.audio_normalize import audio_normalize
from moviepy.audio.fx.volumex import volumex
from tqdm import tqdm

# === Configuration ===
FILE_NAME = "chain_directional_20250903_202214"
INPUT_JSON = f"D:/project/archiver/prog/automontage/semantic_chains/{FILE_NAME}.json"
OUTPUT_VIDEO = f"D:/project/archiver/prog/automontage/exports/{FILE_NAME}_semantic_montage.mp4"

TARGET_RES = (1920, 1080)
FPS = 30
homogenize_audio = 0.7  # 0 = no change, 1 = full normalization, 0.5 = in between
TARGET_DBFS = -24.0     # Target loudness level in dBFS (can adjust based on your audio metadata)

# === Utility Functions ===

def db_to_volume_ratio(db):
    return 10 ** (db / 20)

def normalize_audio_fast(audio_clip, segment_level, target_level, strength=1.0):
    if audio_clip is None or segment_level is None:
        return audio_clip
    gain_db = target_level - segment_level
    gain_ratio = db_to_volume_ratio(gain_db * strength)
    return audio_clip.fx(volumex, gain_ratio)

def normalize_audio_fallback(audio_clip, strength=1.0):
    try:
        normed = audio_normalize(audio_clip)
        if strength < 1.0:
            return audio_clip.fx(volumex, (1 - strength) + strength * normed.max_volume())
        return normed
    except Exception as e:
        print(f"⚠️ Audio normalization failed: {e}")
        return audio_clip

def check_files_exist(segments):
    return [seg["url"] for seg in segments if not os.path.exists(seg["url"])]

def load_segments(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

# === Main Processing ===

def extract_clip(segment):
    try:
        path = segment["url"]
        start = segment["start"]
        end = segment["stop"]

        clip = VideoFileClip(path).subclip(start - 0.3, end + 1.0)

        # Normalize audio using precomputed level or fallback method
        if clip.audio:
            if "audio_level" in segment and segment["audio_level"] is not None:
                clip = clip.set_audio(
                    normalize_audio_fast(clip.audio, segment["audio_level"], TARGET_DBFS, strength=homogenize_audio)
                )
            elif homogenize_audio > 0:
                clip = clip.set_audio(normalize_audio_fallback(clip.audio, strength=homogenize_audio))

        # Resize while preserving aspect ratio
        clip = clip.resize(height=TARGET_RES[1])
        if clip.w > TARGET_RES[0]:
            clip = clip.resize(width=TARGET_RES[0])

        # Pad to match target resolution
        if clip.size != TARGET_RES:
            bg = ColorClip(size=TARGET_RES, color=(0, 0, 0), duration=clip.duration)
            clip = CompositeVideoClip([bg, clip.set_position("center")]).set_fps(FPS)
        else:
            clip = clip.set_fps(FPS)

        return clip

    except Exception as e:
        print(f"⚠️ Error with clip: {segment['url']} ({segment['start']}s) → {e}")
        return None

def main():
    print(f"📂 Loading: {INPUT_JSON}")
    segments = load_segments(INPUT_JSON)
    missing = check_files_exist(segments)

    if missing:
        print("❌ Missing files:")
        for m in missing:
            print("  -", m)
        print("⚠️ Aborting export. Please reconnect drives.")
        return

    print("🎞️ Extracting clips...")
    clips = []
    for seg in tqdm(segments):
        clip = extract_clip(seg)
        if clip:
            clips.append(clip)

    if not clips:
        print("❌ No valid clips found.")
        return

    print("🧵 Concatenating...")
    final = concatenate_videoclips(clips, method="compose")

    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
    print(f"💾 Exporting to: {OUTPUT_VIDEO}")
    final.write_videofile(OUTPUT_VIDEO, codec="libx264", audio_codec="aac")

if __name__ == "__main__":
    main()
