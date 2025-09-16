#!/usr/bin/env python3
# normaliza_dataset_classificacao.py

"""
Script para normalizar um dataset de classificação em pastas
para o formato COCO JSON (coco.json).

Estrutura esperada:
data/
└── classificacao/
    ├── mild/
    │   ├── 2_1_mild.jpg
    │   └── ...
    ├── moderate/
    ├── open/
    └── severe/

Cada subpasta dentro de `data/classificacao` será mapeada a uma categoria distinta.
Cada imagem gerará uma entrada em `images` e uma anotação cobrindo toda a imagem
(em `annotations`), adequada para tarefas de classificação.
"""

import argparse
import os
import json
from PIL import Image
from datetime import datetime

def parse_args() -> argparse.Namespace:
    """
    Analisa os argumentos de linha de comando.

    Returns:
        argparse.Namespace: objeto com atributos:
            - dataset_format: formato atual do dataset (somente "pastas")
            - dataset_path: caminho para a pasta raiz das classes
            - output: nome do arquivo JSON de saída
    """
    parser = argparse.ArgumentParser(
        description="Normaliza dataset de classificação (pastas) para COCO format."
    )
    parser.add_argument(
        "-f", "--dataset_format",
        type=str,
        choices=["pastas"],
        required=True,
        help='Formato atual do dataset (atualmente, só "pastas").'
    )
    parser.add_argument(
        "-d", "--dataset_path",
        type=str,
        required=True,
        help="Caminho para a pasta que contém subpastas de classes."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default=".",
        help="Diretório onde o arquivo JSON será salvo."
    )
    return parser.parse_args()

def main() -> None:
    """
    Função principal que:
      1. Mapeia cada subpasta em uma categoria COCO.
      2. Varre todas as imagens dentro de cada pasta de classe.
      3. Gera entradas em 'images' com metadados (tamanho, nome de arquivo).
      4. Gera uma anotação (bbox full-image) para cada imagem.
      5. Serializa o dicionário final no formato COCO e escreve em disco.
    """
    args = parse_args()
    base = args.dataset_path

    # Monta lista de categorias a partir das subpastas
    class_names = sorted([
        d for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d))
    ])
    categories = []
    for idx, name in enumerate(class_names):
        categories.append({
            "id": idx,
            "name": name,
            "supercategory": ""
        })
    cat2id = {c["name"]: c["id"] for c in categories}

    images      = []  # lista de dicionários COCO 'images'
    annotations = []  # lista de dicionários COCO 'annotations'
    ann_id      = 0   # contador global de annotations

    # Para cada classe, percorre suas imagens
    for class_name in class_names:
        folder = os.path.join(base, class_name)
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue

            path = os.path.join(folder, fname)
            # Lê largura e altura com PIL
            with Image.open(path) as img:
                width, height = img.size

            # Extrai IDs do nome do arquivo (ex: "2_1_mild.jpg")
            base_name = os.path.splitext(fname)[0]
            parts    = base_name.split("_")
            prefix   = int(parts[0])      # ID principal
            slice_id = int(parts[1])      # sub-ID ou slice
            # Gera um image_id único (ex: prefix * 100 + slice_id)
            image_id = prefix * 100 + slice_id

            # Adiciona entrada em 'images'
            images.append({
                "id":            image_id,
                "license":       1,
                "file_name":     fname,#os.path.relpath(path, start=base),
                "height":        height,
                "width":         width,
                "date_captured": datetime.now().isoformat()
            })

            # Cria anotação cobrindo toda a imagem (bbox completa)
            annotations.append({
                "id":           ann_id,
                "image_id":     image_id,
                "category_id":  cat2id[class_name],
                "bbox":         [0, 0, width, height],
                "area":         width * height,
                "segmentation": [],
                "iscrowd":      0
            })
            ann_id += 1

    # Monta o dicionário COCO final
    coco_dict = {
        "categories":  categories,
        "images":      images,
        "annotations": annotations
    }

    out_file = os.path.join(args.output_dir, "classificacao.json")

    # Grava o JSON formatado
    with open(out_file, "x", encoding="utf-8") as f:
        json.dump(coco_dict, f, ensure_ascii=False, indent=4)

    print(f"[OK] COCO JSON salvo em: {out_file}")

    return

if __name__ == "__main__":
    main()
