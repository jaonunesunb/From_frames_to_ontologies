import pandas as pd
from pathlib import Path
from tqdm import tqdm
import yaml
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Funções utilitárias
def load_yml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def parse_geom_data(geom_list):
    objects = defaultdict(list)
    for item in geom_list:
        if 'geom' in item:
            g = item['geom']
            obj_id = g['id1']
            bbox = list(map(int, g['g0'].split()))
            ts = g['ts0']
            objects[obj_id].append({'frame': ts, 'bbox': bbox})
    return objects

def parse_activity_data(activity_list):
    events = []
    for item in activity_list:
        if 'act' in item:
            a = item['act']
            event_type = list(a['act2'].keys())[0]
            ts_start, ts_end = a['timespan'][0]['tsr0']
            actors = [actor['id1'] for actor in a.get('actors', [])]
            events.append({
                'type': event_type,
                'start': ts_start,
                'end': ts_end,
                'duration': ts_end - ts_start,
                'actors': actors
            })
    return events

def parse_region_data(region_list):
    regions = defaultdict(list)
    for item in region_list:
        if 'regions' in item:
            r = item['regions']
            ts = r['ts0']
            poly = r['poly0']
            regions[ts].append(poly)
    return regions

def extract_event_features(events, geom_data, region_data):
    features = []
    for e in events:
        actor_counts = len(e['actors'])
        duration = e['duration']

        avg_width, avg_height = [], []
        for obj_id in e['actors']:
            track = geom_data.get(obj_id, [])
            if track:
                bboxes = [b['bbox'] for b in track if e['start'] <= b['frame'] <= e['end']]
                for bbox in bboxes:
                    x1, y1, x2, y2 = bbox
                    avg_width.append(x2 - x1)
                    avg_height.append(y2 - y1)

        w = sum(avg_width) / len(avg_width) if avg_width else 0
        h = sum(avg_height) / len(avg_height) if avg_height else 0

        regions_count = 0
        if region_data:
            for f in range(e['start'], e['end'] + 1):
                if f in region_data:
                    regions_count += len(region_data[f])

        features.append({
            'duration': duration,
            'num_actors': actor_counts,
            'avg_width': w,
            'avg_height': h,
            'regions_in_event': regions_count
        })
    return features

# Carrega classificador treinado
df = pd.read_csv("dados_balanceados.csv")
le = LabelEncoder()
df["label"] = le.fit_transform(df["type"])
X_train = df[["duration", "num_actors", "avg_width", "avg_height", "regions_in_event"]]
y_train = df["label"]

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Lê os vídeos sem anotação
with open("videos_sem_anotacao.txt", "r") as f:
    video_ids = [line.strip() for line in f.readlines()]

# Diretórios onde procurar os .yml
annotation_dirs = [Path("viratannotations/train"), Path("viratannotations/validate")]

# Predição
predicoes = []

for vid in tqdm(video_ids[:80], desc="🔍 Processando vídeos não anotados"):
    # Localiza arquivos de anotação
    a_path, g_path, r_path = None, None, None
    for d in annotation_dirs:
        if (d / f"{vid}.activities.yml").exists():
            a_path = d / f"{vid}.activities.yml"
        if (d / f"{vid}.geom.yml").exists():
            g_path = d / f"{vid}.geom.yml"
        if (d / f"{vid}.regions.yml").exists():
            r_path = d / f"{vid}.regions.yml"

    if not (a_path and g_path and r_path):
        continue  # pula se faltar algum arquivo

    try:
        events_raw = load_yml(a_path)
        geom_raw = load_yml(g_path)
        region_raw = load_yml(r_path)

        events = parse_activity_data(events_raw)
        geom_data = parse_geom_data(geom_raw)
        region_data = parse_region_data(region_raw)

        feats = extract_event_features(events, geom_data, region_data)

        for i, feat in enumerate(feats):
            X_test = [[feat['duration'], feat['num_actors'], feat['avg_width'], feat['avg_height'], feat['regions_in_event']]]
            pred = clf.predict(X_test)[0]
            prob = max(clf.predict_proba(X_test)[0])

            predicoes.append({
                "video": vid,
                "event_index": i,
                "type_predito": le.inverse_transform([pred])[0],
                "probabilidade": prob,
                **feat
            })

    except Exception as e:
        print(f"⚠️ Erro no vídeo {vid}: {e}")
        continue

# Salva resultados
pd.DataFrame(predicoes).to_csv("predicoes.csv", index=False)
print("✅ Predições salvas em predicoes.csv")
