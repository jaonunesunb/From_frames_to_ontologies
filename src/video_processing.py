#!/usr/bin/env python3
# arquivo: video_processing.py
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import cv2
import numpy as np
from collections import defaultdict
from multiprocessing import Pool, cpu_count

# ----------- Tracker para detecção de objetos (geoms) -----------
class CentroidTracker:
    def __init__(self, max_disappeared=50, max_distance=50):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        if not rects:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects
        input_centroids = np.array([[
            (x1 + x2) // 2,
            (y1 + y2) // 2
        ] for x1, y1, x2, y2 in rects])
        if not self.objects:
            for cen in input_centroids:
                self.register(cen)
        else:
            oids = list(self.objects.keys())
            ocent = np.array([self.objects[oid] for oid in oids])
            D = np.linalg.norm(ocent[:, None] - input_centroids[None, :], axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            used_r, used_c = set(), set()
            for r, c in zip(rows, cols):
                if r in used_r or c in used_c or D[r, c] > self.max_distance:
                    continue
                oid = oids[r]
                self.objects[oid] = input_centroids[c]
                self.disappeared[oid] = 0
                used_r.add(r)
                used_c.add(c)
            for r in set(range(D.shape[0])) - used_r:
                oid = oids[r]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            for c in set(range(D.shape[1])) - used_c:
                self.register(input_centroids[c])
        return self.objects

# ----------- Detecção em vídeo (optimizado) -----------
def detect_geoms(video_path, resize_scale, frame_skip):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mog = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    tracker = CentroidTracker()
    geoms = defaultdict(list)
    pbar = tqdm(total=total, desc=f"Geoms {Path(video_path).stem}", leave=False)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pbar.update(1)
        if frame_idx % frame_skip:
            frame_idx += 1
            continue
        frame = cv2.resize(frame, None, fx=resize_scale, fy=resize_scale)
        fg = mog.apply(frame)
        _, th = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = []
        for c in cnts:
            if cv2.contourArea(c) < 500:
                continue
            x, y, w, h = cv2.boundingRect(c)
            x1 = int(x / resize_scale)
            y1 = int(y / resize_scale)
            x2 = int((x + w) / resize_scale)
            y2 = int((y + h) / resize_scale)
            rects.append((x1, y1, x2, y2))
        objs = tracker.update(rects)
        for oid, cen in objs.items():
            for r in rects:
                cx = (r[0] + r[2]) // 2
                cy = (r[1] + r[3]) // 2
                if abs(cx - cen[0]) < 5 and abs(cy - cen[1]) < 5:
                    geoms[oid].append({'frame': frame_idx, 'bbox': r})
                    break
        frame_idx += 1
    cap.release()
    pbar.close()
    return geoms

def detect_regions(video_path, resize_scale, frame_skip):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mog = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    regions = defaultdict(list)
    pbar = tqdm(total=total, desc=f"Regs {Path(video_path).stem}", leave=False)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pbar.update(1)
        if frame_idx % frame_skip:
            frame_idx += 1
            continue
        frame = cv2.resize(frame, None, fx=resize_scale, fy=resize_scale)
        fg = mog.apply(frame)
        _, th = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) < 500:
                continue
            eps = 0.01 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, eps, True)
            poly = (approx.reshape(-1, 2) / resize_scale).tolist()
            regions[frame_idx].append(poly)
        frame_idx += 1
    cap.release()
    pbar.close()
    return regions

def detected_activities(video_path, resize_scale, frame_skip, min_event_frames):
    geoms = detect_geoms(video_path, resize_scale, frame_skip)
    events = []
    for oid, track in geoms.items():
        frames = sorted(r['frame'] for r in track)
        seg_start = prev = frames[0]
        for f in frames[1:]:
            if f - prev > 1:
                if prev - seg_start + 1 >= min_event_frames:
                    events.append({'start': seg_start, 'end': prev, 'actors': [oid], 'type': 'Unknown'})
                seg_start = f
            prev = f
        if prev - seg_start + 1 >= min_event_frames:
            events.append({'start': seg_start, 'end': prev, 'actors': [oid], 'type': 'Unknown'})
    return events

# ----------- Extração uniformizada em paralelo -----------
from feature_extraction import extract_event_features

def process_video(params):
    vid, resize, skip, min_ev = params
    stem = vid.stem
    acts = detected_activities(str(vid), resize, skip, min_ev)
    geom = detect_geoms(str(vid), resize, skip)
    regs = detect_regions(str(vid), resize, skip)
    rec_raw, rec_feat, rec_geom, rec_reg = [], [], [], []
    for e in acts:
        rec_raw.append({
            'video': stem,
            'start_frame': e['start'],
            'end_frame': e['end'],
            'type': e['type'],
            'actors': ','.join(map(str, e['actors']))
        })
    for f in extract_event_features(acts, geom, regs):
        f['video'] = stem
        rec_feat.append(f)
    for oid, trk in geom.items():
        for r in trk:
            rec_geom.append({
                'video': stem,
                'obj_id': oid,
                'frame': r['frame'],
                'x1': r['bbox'][0],
                'y1': r['bbox'][1],
                'x2': r['bbox'][2],
                'y2': r['bbox'][3]
            })
    for fr, polys in regs.items():
        for p in polys:
            rec_reg.append({
                'video': stem,
                'frame': fr,
                'polygon': p
            })
    return rec_raw, rec_feat, rec_geom, rec_reg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processa vídeos em paralelo')
    parser.add_argument('--videos_txt', required=True, help='TXT com stems')
    parser.add_argument('--videos_root', required=True, help='Pasta base com raw_videos_1..4')
    parser.add_argument('--output_dir', required=True, help='Diretório de saída')
    parser.add_argument('--resize_scale', type=float, default=0.5, help='Fator de escala')
    parser.add_argument('--frame_skip', type=int, default=2, help='Skip de quadros')
    parser.add_argument('--min_event_frames', type=int, default=30, help='Min frames por evento')
    parser.add_argument('--num_workers', type=int, default=cpu_count(), help='Número de processos')
    parser.add_argument('--limit', type=int, default=100, help='Limita primeiros N vídeos')
    args = parser.parse_args()

    # leitura dos stems e limitação
    stems = [l.strip() for l in open(args.videos_txt, encoding='utf-8') if l.strip()]
    stems = set(stems[:args.limit])

    # coleta os paths
    base = Path(args.videos_root)
    vids = []
    for i in range(1, 5):
        dir_i = base / f"raw_videos_{i}"
        if dir_i.is_dir():
            vids.extend(list(dir_i.rglob("*.mp4")))
    vids = [v for v in vids if v.stem in stems]

    # parâmetros por vídeo
    params = [(v, args.resize_scale, args.frame_skip, args.min_event_frames) for v in vids]

    # paralelização com progress bar
    with Pool(args.num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_video, params), total=len(params), desc="Vídeos totais"))

    # agregação
    all_raw, all_feat, all_geom, all_reg = [], [], [], []
    for r, f, g, rg in results:
        all_raw.extend(r)
        all_feat.extend(f)
        all_geom.extend(g)
        all_reg.extend(rg)

    # salvamento
    out = Path(args.output_dir)
    out.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(all_raw).to_csv(out / 'activities_unlabeled_raw.csv', index=False)
    pd.DataFrame(all_feat).to_csv(out / 'activities_unlabeled_feats.csv', index=False)
    pd.DataFrame(all_geom).to_csv(out / 'geoms_unlabeled.csv', index=False)
    pd.DataFrame(all_reg).to_csv(out / 'regions_unlabeled.csv', index=False)
    print("✅ Extração paralela completa em", out)
