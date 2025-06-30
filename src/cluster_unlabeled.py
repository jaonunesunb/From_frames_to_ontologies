#!/usr/bin/env python3
# cluster_unlabeled.py

import os
import cv2
import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torchvision.models import resnet18
import torchvision.transforms as T
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv'}

def get_wanted_stems(txt_path):
    """
    Lê o TXT e retorna um set com os stems (nome sem extensão)
    das linhas — ex: 'VIRAT_S_000003' bate com 'VIRAT_S_000003.mp4'.
    """
    stems = set()
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            name = line.strip()
            if not name:
                continue
            p = Path(name)
            stems.add(p.stem)
    return stems

def find_videos(root_dir, wanted_stems):
    """
    Varre recursivamente root_dir e retorna lista de caminhos
    cujos stems estão em wanted_stems.
    """
    found = []
    for path in Path(root_dir).rglob('*'):
        if path.suffix.lower() in VIDEO_EXTS and path.stem in wanted_stems:
            found.append(str(path))
    return sorted(found)

def build_model(device):
    """Carrega ResNet-18 pré-treinada, removendo a última FC."""
    model = resnet18(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.to(device).eval()
    return model

def extract_video_embedding(path, model, device, num_frames=16):
    """Extrai embedding 512-D de um clipe via amostragem de frames."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total <= 0:
        raise ValueError(f"Não foi possível ler frames de {path}")
    idxs = np.linspace(0, total - 1, num_frames, dtype=int)
    cap = cv2.VideoCapture(path)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"Nenhum frame extraído de {path}")

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
    ])
    batch = torch.stack([transform(f) for f in frames]).to(device)
    with torch.no_grad():
        feats = model(batch)  # [N,512,1,1]
    feats = feats.squeeze(-1).squeeze(-1)  # [N,512]
    return feats.mean(dim=0).cpu().numpy()

def extract_all(videos_txt, videos_root, output_csv, device, num_frames):
    """Extrai embeddings de todos os vídeos listados em videos_txt."""
    wanted = get_wanted_stems(videos_txt)
    vids = find_videos(videos_root, wanted)
    if not vids:
        raise RuntimeError(f"Nenhum dos vídeos em {videos_txt} foi encontrado em {videos_root}")
    print(f"Encontrados {len(vids)} vídeos para processar.\n")

    model = build_model(device)
    records = []
    for path in tqdm(vids, desc="Extraindo embeddings", unit="vídeo"):
        try:
            emb = extract_video_embedding(path, model, device, num_frames)
            rec = {'video': os.path.basename(path)}
            rec.update({f'feat_{i}': float(emb[i]) for i in range(emb.shape[0])})
            records.append(rec)
        except Exception as e:
            tqdm.write(f"[ERRO] {path} → {e}")

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"\nSalvo embeddings em '{output_csv}' ({len(df)} vídeos).")
    return output_csv

def cluster_dbscan(features_csv, output_csv, eps, min_samples):
    """Aplica StandardScaler + DBSCAN e salva resultados."""
    df = pd.read_csv(features_csv)
    X = df.drop(columns=['video']).values
    Xs = StandardScaler().fit_transform(X)
    print("\nRodando DBSCAN...")
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=-1)
    labels = db.fit_predict(Xs)
    df['cluster'] = labels
    df.to_csv(output_csv, index=False)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    print(f"DBSCAN → {n_clusters} clusters, {n_noise} outliers.")
    print(f"Salvo clusters em '{output_csv}'")
    return df

def main():
    p = argparse.ArgumentParser(
        description="Extrai embeddings e faz DBSCAN nos vídeos sem anotação (filtrando pela listagem)"
    )
    p.add_argument(
        "--videos_txt", required=True,
        help="TXT com nomes (ou caminhos) dos vídeos sem anotação (um por linha)"
    )
    p.add_argument(
        "--videos_root", required=True,
        help="Diretório raiz contendo raw_videos_1, raw_videos_2, raw_videos_3, raw_videos_4"
    )
    p.add_argument(
        "--embeddings_csv", default="features_unlabeled.csv",
        help="Onde salvar os embeddings extraídos"
    )
    p.add_argument(
        "--clusters_csv", default="features_unlabeled_clusters.csv",
        help="Onde salvar o CSV com rótulos de cluster"
    )
    p.add_argument(
        "--num_frames", type=int, default=16,
        help="Número de frames amostrados por vídeo"
    )
    p.add_argument(
        "--eps", type=float, default=0.5,
        help="eps do DBSCAN"
    )
    p.add_argument(
        "--min_samples", type=int, default=5,
        help="min_samples do DBSCAN"
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}\n")

    feats_csv = extract_all(
        videos_txt=args.videos_txt,
        videos_root=args.videos_root,
        output_csv=args.embeddings_csv,
        device=device,
        num_frames=args.num_frames
    )
    cluster_dbscan(
        features_csv=feats_csv,
        output_csv=args.clusters_csv,
        eps=args.eps,
        min_samples=args.min_samples
    )

if __name__ == "__main__":
    main()
