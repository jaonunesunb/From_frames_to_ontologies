import os

# Absolute paths for data directories
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_VIDEO_DIR = os.path.join(PROJECT_ROOT, "data", "raw_videos")
CACHE_DIR     = os.path.join(PROJECT_ROOT, "data", "cache", "trajectories")
FEATURES_CSV  = os.path.join(PROJECT_ROOT, "data", "features.csv")
MINING_CSV    = os.path.join(PROJECT_ROOT, "data", "mining_results.csv")

# YOLOv5 settings
YOLO_WEIGHTS = os.path.join(PROJECT_ROOT, "yolov5s.pt")
YOLO_CONF    = 0.4

# Clustering parameters
DBSCAN_EPS         = 0.5
DBSCAN_MIN_SAMPLES = 5
