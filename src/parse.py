import os
import glob
import yaml
import pandas as pd
import numpy as np
import re
from tqdm import tqdm

# Diretórios com anotações
ROOT_DIR = r"C:\Users\Pegad\OneDrive\Área de Trabalho\vs\UFU\projeto_final\viratannotations"
SUBDIRS = [os.path.join(ROOT_DIR, d) for d in ("train", "validate")]

# Função para carregar e garantir lista de documentos YAML
def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        docs = list(yaml.safe_load_all(f))
    flat = []
    for d in docs:
        if isinstance(d, list):
            flat.extend(d)
        else:
            flat.append(d)
    return flat

# Parser de um único prefixo de vídeo
def parse_video(prefix, folder):
    base = os.path.join(folder, prefix)
    files = {
        'activities': base + '.activities.yml',
        'regions':    base + '.regions.yml',
        'geom':       base + '.geom.yml',
        'types':      base + '.types.yml'
    }
    acts = load_yaml(files['activities']) if os.path.exists(files['activities']) else []
    regs = load_yaml(files['regions'])    if os.path.exists(files['regions'])    else []
    gms = load_yaml(files['geom'])        if os.path.exists(files['geom'])       else []
    tps = load_yaml(files['types'])       if os.path.exists(files['types'])      else []

    # Monta mapas para regiões, geometrias e tipos
    reg_map = {}
    for e in regs:
        # pular metas e entires sem key 'regions'
        if not isinstance(e, dict) or 'regions' not in e:
            continue
        r = e['regions']
        if 'id1' not in r or 'ts0' not in r:
            continue
        key = (r['id1'], r['ts0'])
        reg_map.setdefault(key, []).append(r)

    geom_map = {}
    for e in gms:
        if not isinstance(e, dict) or 'geom' not in e:
            continue
        g = e['geom']
        if 'id1' not in g or 'ts0' not in g:
            continue
        key = (g['id1'], g['ts0'])
        geom_map.setdefault(key, []).append(g)

    type_map = {}
    for e in tps:
        if not isinstance(e, dict) or 'types' not in e:
            continue
        t = e['types']
        if 'id1' in t and 'cset3' in t:
            cls = max(t['cset3'].items(), key=lambda x: x[1])[0]
            type_map[t['id1']] = cls

    rows = []
    for ev in acts:
        # Trata meta entries como labels (ignora linhas de header de vídeo)
        if 'meta' in ev and isinstance(ev['meta'], str) and not ev['meta'].startswith(prefix):
            raw = ev['meta']
            m = re.match(r"^(.+?)\s+\d", raw)
            label = m.group(1) if m else raw
            rows.append({
                'video': prefix,
                'actor_id': np.nan,
                'subject_type': np.nan,
                'predicate': label,
                'frame': np.nan,
                'x': np.nan, 'y': np.nan, 'w': np.nan, 'h': np.nan,
                'region_id': np.nan, 'region_cx': np.nan, 'region_cy': np.nan
            })
            continue
        # Eventos com atores
        act_entry = ev.get('act')
        if not act_entry:
            continue
        predicate = next(iter(act_entry.get('act2', {})), 'Unknown')
        actors = act_entry.get('actors', [])
        timespans = act_entry.get('timespan', [])
        for tspan in timespans:
            for _, (start_frame, _) in tspan.items():
                frame = start_frame
                for actor in actors:
                    aid = actor.get('id1')
                    row = {
                        'video': prefix,
                        'actor_id': aid,
                        'subject_type': type_map.get(aid, 'Unknown'),
                        'predicate': predicate,
                        'frame': frame,
                        'x': np.nan, 'y': np.nan, 'w': np.nan, 'h': np.nan,
                        'region_id': np.nan, 'region_cx': np.nan, 'region_cy': np.nan
                    }
                    # Preenche região, se existir
                    for r in reg_map.get((aid, frame), []):
                        row['region_id'] = r['id1']
                        pts = r.get('poly0', [])
                        if pts:
                            xs, ys = zip(*pts)
                            row['region_cx'] = np.mean(xs)
                            row['region_cy'] = np.mean(ys)
                        break
                    # Preenche geom, se existir
                    for g in geom_map.get((aid, frame), []):
                        coords = g.get('g0', '')
                        if coords:
                            vals = list(map(float, coords.split()))
                            row['x'], row['y'], row['w'], row['h'] = vals
                        break
                    rows.append(row)
    return rows

# Coleta todos os arquivos .activities.yml
activity_files = []
for folder in SUBDIRS:
    activity_files.extend(glob.glob(os.path.join(folder, '*.activities.yml')))

# Parsing com barra de progresso
tqdm_kwargs = {'desc': 'Parsing videos', 'unit': 'file'}
all_rows = []
for filepath in tqdm(activity_files, **tqdm_kwargs):
    prefix = os.path.basename(filepath).replace('.activities.yml', '')
    folder = os.path.dirname(filepath)
    all_rows.extend(parse_video(prefix, folder))

# Cria DataFrame e mostra colunas
df = pd.DataFrame(all_rows)
print('Colunas carregadas:', df.columns.tolist())

# Garante colunas padrão
required_cols = ['video','actor_id','subject_type','predicate','frame',
                 'x','y','w','h','region_id','region_cx','region_cy']
for col in required_cols:
    df[col] = df.get(col, np.nan)

# Engenharia de features básicas
df['area'] = df['w'] * df['h']
df['speed'] = np.sqrt((df['x'].diff())**2 + (df['y'].diff())**2)

# Preenche NaNs remanescentes com 0
df.fillna(0, inplace=True)

# Gera CSV supervisionado (com predicate)
train_labeled = df.copy()
train_labeled.to_csv('new_train_labeled.csv', index=False)
# Gera CSV não-supervisionado (sem predicate)
train_unlabeled = df.drop(columns=['predicate'], errors='ignore')
train_unlabeled.to_csv('new_train_unlabeled.csv', index=False)

print('Arquivos gerados: new_train_labeled.csv e new_train_unlabeled.csv')
