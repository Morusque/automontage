
# use 3.10.0

import os
import json
import subprocess
import numpy as np
import soundfile as sf
from tqdm import tqdm

TRANSCRIPT_FOLDER = "D:/project/archiver/prog/automontage/transcripts"
AUDIO_TAG = "has_audio_levels"

def extract_audio_to_wav(video_path, temp_wav_path):
    command = [
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000",
        "-f", "wav", temp_wav_path
    ]

    subprocess.run(command, check=True)

def load_audio_array(wav_path):
    data, samplerate = sf.read(wav_path)
    return data, samplerate

def compute_db_from_slice(audio, samplerate, start, end):
    start_idx = int(start * samplerate)
    end_idx = int(end * samplerate)
    if end_idx <= start_idx or end_idx > len(audio):
        return None
    segment = audio[start_idx:end_idx]
    if len(segment) == 0:
        return None
    rms = np.sqrt(np.mean(segment.astype(np.float32)**2))
    return round(float(20 * np.log10(rms)), 2) if rms > 0 else -float("inf")

transcript_files = [
    f for f in os.listdir(TRANSCRIPT_FOLDER)
    if f.endswith(".json") and not f.startswith("000_")
]

for fname in tqdm(transcript_files, desc="📁 Processing transcripts"):
    full_path = os.path.join(TRANSCRIPT_FOLDER, fname)
    with open(full_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get(AUDIO_TAG) is True:
        continue  # Already tagged

    source = data.get("source_path")
    if not source or not os.path.exists(source):
        continue

    segments = data.get("segments", [])
    if not segments:
        continue
    
    data[AUDIO_TAG] = True
    
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    continue

    temp_wav = "temp_audio.wav"

    try:
        extract_audio_to_wav(source, temp_wav)
        audio, samplerate = load_audio_array(temp_wav)
    except Exception as e:
        print(f"⚠️ Failed to extract audio from {source}: {e}")
        continue

    updated = False
    for i, seg in enumerate(tqdm(segments, leave=False, desc=f"🎧 Segments in {fname}")):
        if "audio_level" in seg:
            continue
        start = seg.get("start")
        end = seg.get("end")
        if start is None or end is None or end <= start:
            print(f"⚠️ Invalid segment {i}: {start} → {end}")
            continue
        level = compute_db_from_slice(audio, samplerate, start, end)
        if level is not None:
            seg["audio_level"] = float(level)
            updated = True
            print(f"📊 [{fname}] Segment {i}: {start:.2f} → {end:.2f} = {level:.2f} dB")
