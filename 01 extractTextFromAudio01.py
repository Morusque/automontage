# Use Python 3.12.3

import os
import json
import whisper
import datetime
import random
from tqdm import tqdm
import subprocess
import re

def normalize_path(p):
    return os.path.normpath(p).replace("\\", "/")

def load_priority_urls(filepath="D:/project/archiver/prog/automontage/priority_movies.txt"):
    if not os.path.isfile(filepath):
        return set()
    with open(filepath, "r", encoding="utf-8") as f:
        return set(normalize_path(line.strip()) for line in f if line.strip())

SOURCE_FOLDERS = ["D:/recup/current/films", "E:/", "F:/"]
TRANSCRIPT_FOLDER = "D:/project/archiver/prog/automontage/transcripts/"
INDEX_FILE = os.path.join(TRANSCRIPT_FOLDER, "000_index_all_files.json")
os.makedirs(TRANSCRIPT_FOLDER, exist_ok=True)

# Load or init index
if os.path.exists(INDEX_FILE):
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        index = json.load(f)
else:
    index = {}

WHISPER_MODEL = "medium"
model = whisper.load_model(WHISPER_MODEL)

video_exts = ('.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv',
              '.webm', '.mpg', '.mpeg', '.3gp', '.m4v')

# Build full list of available videos
all_videos = []
for folder in SOURCE_FOLDERS:
    for root, _, files in os.walk(folder):
        for fn in files:
            if fn.lower().endswith(video_exts):
                all_videos.append(os.path.join(root, fn))

# Normalize paths
all_videos = [normalize_path(p) for p in all_videos]
indexed_paths = set(normalize_path(p) for p in index.values())

# Split priority and others
priority_paths = load_priority_urls()
priority_videos = [p for p in all_videos if p in priority_paths and p not in indexed_paths]
other_videos = [p for p in all_videos if p not in priority_paths and p not in indexed_paths]
random.shuffle(other_videos)

def parse_srt(srt_text):
    segments = []
    blocks = srt_text.strip().split('\n\n')
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) >= 2:
            timestamp_line = lines[1] if '-->' in lines[1] else lines[0]
            match = re.match(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s+-->\s+(\d{2}):(\d{2}):(\d{2}),(\d{3})', timestamp_line)
            if not match:
                continue
            h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, match.groups())
            start = h1*3600 + m1*60 + s1 + ms1/1000
            end = h2*3600 + m2*60 + s2 + ms2/1000
            text = ' '.join(lines[2:]) if '-->' in lines[1] else ' '.join(lines[1:])
            segments.append({
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text
            })
    return segments

def extract_subtitles(path):
    try:
        result = subprocess.run([
            "ffmpeg", "-y", "-i", path,
            "-map", "0:s:0", "-f", "srt", "-"
        ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=30)

        srt = result.stdout.decode("utf-8", errors="ignore")
        if len(srt.strip()) < 10:
            return None
        return parse_srt(srt)
    except Exception as e:
        print(f"⚠️ Failed to extract subtitles: {e}")
        return None

def transcribe_video(path):
    try:
        basename = os.path.splitext(os.path.basename(path))[0]
        safe_name = basename.replace(" ", "_").replace(":", "") \
                             .replace("/", "_").replace("\\", "_")
        
        base_name = safe_name
        suffix = 1
        json_filename = f"{base_name}.json"
        out_path = os.path.join(TRANSCRIPT_FOLDER, json_filename)

        while os.path.exists(out_path):
            suffix += 1
            json_filename = f"{base_name}_{suffix}.json"
            out_path = os.path.join(TRANSCRIPT_FOLDER, json_filename)

        print(f"🔍 Processing: {basename}", flush=True)

        segments = extract_subtitles(path)
        if segments:
            result = {
                "source_path": path,
                "segments": segments,
                "whisper_model": "subtitles"
            }
            result["transcribed_at"] = datetime.datetime.now().isoformat()
            print(f"✅ Used subtitles for {basename}")
        else:
            print(f"🧠 Whispering {basename}...")
            result = whisper.transcribe(model, path, language=None)
            result["source_path"] = path
            result["whisper_model"] = WHISPER_MODEL
            result["transcribed_at"] = datetime.datetime.now().isoformat()

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        index[json_filename] = path
        with open(INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] ✅ Saved: {safe_name}.json")

    except Exception as e:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] ❌ Error processing {path}: {e}")

# Process in priority order
print(f"🎯 Found {len(priority_videos)} prioritized videos to process.")
for path in tqdm(priority_videos, desc="🎙 Processing prioritized"):
    transcribe_video(path)

# Then process the rest
for path in tqdm(other_videos, desc="🎙 Processing remaining"):
    transcribe_video(path)
