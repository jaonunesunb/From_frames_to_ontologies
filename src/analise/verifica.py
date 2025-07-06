from pathlib import Path
from collections import defaultdict

# Caminhos
video_dirs = [Path(f"data/raw_videos_{i}") for i in range(1, 5)]
annotation_dirs = [Path("viratannotations/train"), Path("viratannotations/validate")]
expected_suffixes = [".activities.yml", ".geom.yml", ".regions.yml", ".types.yml"]

# Coletar vídeos
video_stems = set()
video_full_paths = dict()
for d in video_dirs:
    for f in d.glob("*.mp4"):
        stem = f.stem
        video_stems.add(stem)
        video_full_paths[stem] = str(f)

# Coletar anotações por prefixo
annotations = defaultdict(set)
annotation_full_paths = defaultdict(dict)
for d in annotation_dirs:
    for suffix in expected_suffixes:
        for f in d.glob(f"*{suffix}"):
            stem = f.stem.replace(suffix.replace(".yml", ""), "")
            annotations[stem].add(suffix)
            annotation_full_paths[stem][suffix] = str(f)

# Agrupar categorias
videos_with_all_annotations = []
videos_with_partial_annotations = []
videos_without_annotations = []
annotations_without_video = []

for v in video_stems:
    if v in annotations:
        if len(annotations[v]) == 4:
            videos_with_all_annotations.append(v)
        else:
            videos_with_partial_annotations.append((v, sorted(list(annotations[v]))))
    else:
        videos_without_annotations.append(v)

for a in annotations:
    if a not in video_stems:
        annotations_without_video.append((a, sorted(list(annotations[a]))))

# Exibir resumo
print(f"🎥 Total de vídeos: {len(video_stems)}")
print(f"📝 Total de prefixos com anotações: {len(annotations)}")
print(f"✅ Vídeos com TODAS as anotações: {len(videos_with_all_annotations)}")
print(f"⚠️ Vídeos com anotações PARCIAIS: {len(videos_with_partial_annotations)}")
print(f"❌ Vídeos SEM anotação: {len(videos_without_annotations)}")
print(f"❌ Anotações SEM vídeo: {len(annotations_without_video)}")

# Exibir exemplos
print("\n✅ Exemplos com todas as anotações:")
for v in videos_with_all_annotations[:5]:
    print(f"- {v}")

print("\n⚠️ Exemplos com anotações parciais:")
for v, parts in videos_with_partial_annotations[:5]:
    print(f"- {v}: {parts}")

print("\n❌ Exemplos de vídeos sem anotação:")
print(videos_without_annotations[:5])

print("\n❌ Exemplos de anotações sem vídeo:")
for a, parts in annotations_without_video[:5]:
    print(f"- {a}: {parts}")
# Salvar lista completa dos vídeos sem anotação
with open("videos_sem_anotacao.txt", "w") as f:
    for v in sorted(videos_without_annotations):
        f.write(v + "\n")

# Salvar lista completa dos vídeos com todas as anotações
with open("videos_completos.txt", "w") as f:
    for v in sorted(videos_with_all_annotations):
        f.write(v + "\n")

# Salvar lista completa de anotações que não têm vídeo correspondente
with open("annotations_sem_video.txt", "w") as f:
    for a, partes in sorted(annotations_without_video):
        f.write(f"{a}: {', '.join(partes)}\n")

print("\n📂 Arquivos gerados:")
print("- videos_sem_anotacao.txt")
print("- videos_completos.txt")
print("- annotations_sem_video.txt")
