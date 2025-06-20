import numpy as np
import pandas as pd

def extract_features(trajectories, fps):
    rows = []
    for tid, pts in trajectories.items():
        if len(pts) < 2:
            continue
        times = np.array([f for f,_,_ in pts]) / fps
        coords = np.array([[x,y] for _,x,y in pts])
        deltas = coords[1:] - coords[:-1]
        dt     = times[1:] - times[:-1]
        speeds = np.linalg.norm(deltas, axis=1) / dt
        avg_speed = float(speeds.mean())
        mean_dir  = deltas.mean(axis=0)
        dir_norm  = mean_dir / (np.linalg.norm(mean_dir)+1e-6)
        duration  = float(times[-1] - times[0])
        rows.append({
            'track_id': tid,
            'avg_speed': avg_speed,
            'dir_x': float(dir_norm[0]),
            'dir_y': float(dir_norm[1]),
            'duration': duration
        })
    return pd.DataFrame(rows)
