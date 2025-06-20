import os

def list_all_clips():
    if not os.path.isdir(RAW_VIDEO_DIR := os.path.join(os.path.dirname(__file__), "..", "data", "raw_videos")):
        raise FileNotFoundError(f"Raw videos directory not found: {RAW_VIDEO_DIR}")
    clips = []
    for root, _, files in os.walk(RAW_VIDEO_DIR):
        for f in files:
            if f.lower().endswith((".mp4", ".mpg")):
                clips.append(os.path.join(root, f))
    clips.sort()
    print(f"[dataset] Found {len(clips)} clips under {RAW_VIDEO_DIR}")
    return clips