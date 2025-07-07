# From Frames to Ontologies

Este repositório contém scripts para extrair anotações do dataset VIRAT, gerar conjuntos de treinamento e executar modelos de classificação e clustering. Os resultados (imagens e CSVs) ficam em `src/outputs/`.

## Requisitos

- Python 3.10 ou superior
- Dependências listadas em `requirements.txt`

Instale os pacotes com:

```bash
pip install -r requirements.txt
```

## Dataset

As anotações do VIRAT não acompanham este repositório. Clone o repositório [`viratannotations`](https://gitlab.kitware.com/viratdata/viratannotations) para obter os arquivos `.yml` e ajuste o caminho em `src/parse.py` para onde você os salvou:

```python
ROOT_DIR = r"/caminho/para/viratannotations"
SUBDIRS = [os.path.join(ROOT_DIR, "train"), os.path.join(ROOT_DIR, "validate")]
```

A estrutura esperada é:

```
viratannotations/
├── train/
│   ├── VIDEO.activities.yml
│   ├── VIDEO.geom.yml
│   ├── VIDEO.regions.yml
│   └── VIDEO.types.yml
└── validate/
    └── ...
```

## Passo a passo

1. **Gerar CSVs a partir das anotações**

   ```bash
   python src/parse.py
   ```

   Isso cria `new_train_labeled.csv` e `new_train_unlabeled.csv` no diretório atual.

2. **Limpar as classes**

   ```bash
   python src/limpeza.py
   ```

   O script gera `src/outputs/paired_train_labeled_cleaned.csv`, corrigindo rótulos `subject_type` igual a `0` para `Unknown`.

3. **Treinar modelos supervisionados**

   - Support Vector Machine:

     ```bash
     python src/SVM_final.py
     ```

   - Random Forest:

     ```bash
     python src/RF_final.py
     ```

   As acurácias e matrizes de confusão são gravadas em `src/outputs/`.

4. **Executar clustering**

   ```bash
   python src/DbScan_final.py
   ```

   Serão gerados arquivos de “outliers” e imagens de clustering em `src/outputs/`.

## Resultados

Imagens, matrizes de confusão e CSVs produzidos ficam organizados em `src/outputs/Images` e `src/outputs/TXTs`.
