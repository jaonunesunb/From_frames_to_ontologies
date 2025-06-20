import os
import pickle
import cv2
import torch
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort
from config import CACHE_DIR, YOLO_WEIGHTS, YOLO_CONF

# Patch deprecated torch.cuda.amp.autocast signature
if hasattr(torch.cuda, 'amp'):
    from torch.amp import autocast as new_autocast
    def patched_autocast(enabled=True):
        return new_autocast(device_type='cuda', enabled=enabled)
    torch.cuda.amp.autocast = patched_autocast
from config import CACHE_DIR, YOLO_WEIGHTS, YOLO_CONF

# Constants
FRAME_SKIP = 5    # process 1 out of every 5 frames
MAX_FRAMES = 3500 # max frames to process per clip

os.makedirs(CACHE_DIR, exist_ok=True)
model = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLO_WEIGHTS, trust_repo=True)
model.conf = YOLO_CONF
tracker = DeepSort(max_age=30)

def process_and_cache(clip_path):
    vid_id = os.path.splitext(os.path.basename(clip_path))[0]
    cache_file = os.path.join(CACHE_DIR, f"{vid_id}.pkl")
    if os.path.exists(cache_file):
        return pickle.load(open(cache_file, 'rb'))
    cap = cv2.VideoCapture(clip_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    trajectories = defaultdict(list)
    frame_idx = 0
    while True:
        if frame_idx >= MAX_FRAMES:
            break
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % FRAME_SKIP != 0:
            frame_idx += 1
            continue
        preds = model(frame[..., ::-1]).xyxy[0].cpu().numpy()
        dets = [([int(x1),int(y1),int(x2),int(y2)], float(conf), int(cls))
                for *xyxy, conf, cls in preds
                for x1,y1,x2,y2 in [xyxy]]
        tracks = tracker.update_tracks(dets, frame=frame)
        for t in tracks:
            if not t.is_confirmed():
                continue
            tid = t.track_id
            x1,y1,x2,y2 = map(int, t.to_ltrb())
            cx, cy = (x1+x2)//2, (y1+y2)//2
            trajectories[tid].append((frame_idx, cx, cy))
        frame_idx += 1
    cap.release()
    pickle.dump((trajectories, fps), open(cache_file, 'wb'))
    print(f"[video_processing] Cached {vid_id}: {len(trajectories)} tracks at {fps} fps.")
    return trajectories, fps