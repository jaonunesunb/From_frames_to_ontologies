import csv
import yaml
from pathlib import Path
from tqdm import tqdm

# onde estão suas anotações
ANN_DIRS = [Path("viratannotations/train"), Path("viratannotations/validate")]

# onde salvará os CSVs
OUT = Path("data")
OUT.mkdir(exist_ok=True)

def iter_files(ext):
    for d in ANN_DIRS:
        yield from d.glob(f"*{ext}.yml")

def dump_geom_csv():
    paths = list(iter_files(".geom"))
    with open(OUT/"geom.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video","track_id","frame","x1","y1","x2","y2"])
        for path in tqdm(paths, desc="Gerando geom.csv"):
            video = path.stem.replace(".geom","")
            data = yaml.safe_load(path.read_text())
            for item in data:
                if "geom" in item:
                    g = item["geom"]
                    x1,y1,x2,y2 = map(int, g["g0"].split())
                    writer.writerow([video, g["id1"], g["ts0"], x1,y1,x2,y2])

def dump_activity_csv():
    paths = list(iter_files(".activities"))
    with open(OUT/"activities.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video","activity_id","type","start_frame","end_frame"])
        for path in tqdm(paths, desc="Gerando activities.csv"):
            video = path.stem.replace(".activities","")
            data = yaml.safe_load(path.read_text())
            for item in data:
                if "act" in item:
                    a = item["act"]
                    typ = next(iter(a["act2"].keys()))
                    start,end = a["timespan"][0]["tsr0"]
                    writer.writerow([video, a["id2"], typ, start, end])

def dump_region_csv():
    paths = list(iter_files(".regions"))
    with open(OUT/"regions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video","region_id","frame","polygon"])
        for path in tqdm(paths, desc="Gerando regions.csv"):
            video = path.stem.replace(".regions","")
            data = yaml.safe_load(path.read_text())
            for item in data:
                if "regions" in item:
                    r = item["regions"]
                    writer.writerow([video, r["id1"], r["ts0"], r["poly0"]])

if __name__ == "__main__":
    dump_geom_csv()
    dump_activity_csv()
    dump_region_csv()
    print("✅ CSVs gerados em data/: geom.csv, activities.csv, regions.csv")
