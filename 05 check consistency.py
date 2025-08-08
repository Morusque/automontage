# check_index_files.py

import os
import json

TRANSCRIPT_FOLDER = "D:/project/archiver/prog/automontage/transcripts"
INDEX_FILE = os.path.join(TRANSCRIPT_FOLDER, "000_index_all_files.json")

def normalize_path(p):
    return os.path.normpath(p).replace("\\", "/")

def main():
    if not os.path.exists(INDEX_FILE):
        print(f"❌ Index file not found at {INDEX_FILE}")
        return

    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        index = json.load(f)

    print(f"🔍 Checking {len(index)} entries in index...")

    missing_files = 0
    for json_name, file_path in index.items():
        full_path = normalize_path(file_path)
        if not os.path.exists(full_path):
            print(f"⚠️ Missing file for {json_name}: {full_path}")
            missing_files += 1

    if missing_files == 0:
        print("✅ All referenced files exist.")
    else:
        print(f"⚠️ {missing_files} file(s) referenced in index are missing.")

if __name__ == "__main__":
    main()

input("Press Enter to exit...")