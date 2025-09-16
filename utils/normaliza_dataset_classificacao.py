#!/usr/bin/env python3
import argparse
import os
import json
from PIL import Image
from datetime import datetime

def parse_args() -> argparse.Namespace:
    """
    Analisa e processa os argumentos fornecidos através da linha de comandos.
    """
    parser = argparse.ArgumentParser(
        description="Converte um dataset de classificação (organizado em pastas por classe) para o formato COCO JSON."
    )
    parser.add_argument(
        "-f", "--dataset_format",
        type=str,
        choices=["pastas"],
        required=True,
        help='Formato atual do dataset. Atualmente, só suporta "pastas".'
    )
    parser.add_argument(
        "-d", "--dataset_path",
        type=str,
        required=True,
        help="Caminho para a pasta que contém as subpastas de cada classe (ex: data/classificacao/Narina)."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default=".",
        help="Diretório onde o ficheiro JSON de saída será guardado."
    )
    return parser.parse_args()

def main() -> None:
    """
    Função principal que executa o processo de conversão.
    """
    args = parse_args()
    base_path = args.dataset_path

    # 1. Mapear Nomes de Pastas para Categorias COCO
    # Lista todas as subpastas no diretório do dataset, que correspondem às classes.
    class_names = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    
    # Cria a secção 'categories' do JSON, atribuindo um ID numérico a cada classe.
    categories = [{"id": idx, "name": name, "supercategory": ""} for idx, name in enumerate(class_names)]
    cat2id = {c["name"]: c["id"] for c in categories} # Dicionário para mapeamento rápido de nome para ID.

    images = []
    annotations = []
    image_id_counter = 0

    # 2. Iterar sobre cada classe e as suas imagens
    for class_name in class_names:
        folder_path = os.path.join(base_path, class_name)
        for fname in sorted(os.listdir(folder_path)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            # 3. Gerar a entrada 'images' para cada imagem
            path = os.path.join(folder_path, fname)
            with Image.open(path) as img:
                width, height = img.size

            images.append({
                "id": image_id_counter,
                "file_name": fname,
                "height": height,
                "width": width,
                "date_captured": datetime.now().isoformat()
            })

            # 4. Gerar a entrada 'annotations' para cada imagem
            # Para classificação, a anotação é uma bounding box que cobre a imagem inteira.
            annotations.append({
                "id": image_id_counter, # Usamos o mesmo contador para o ID da anotação
                "image_id": image_id_counter,
                "category_id": cat2id[class_name],
                "bbox": [0, 0, width, height],
                "area": width * height,
                "segmentation": [],
                "iscrowd": 0
            })
            image_id_counter += 1

    # 5. Montar e Guardar o Dicionário COCO Final
    coco_dict = {
        "categories": categories,
        "images": images,
        "annotations": annotations
    }

    out_file = os.path.join(args.output_dir, "classificacao.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(coco_dict, f, ensure_ascii=False, indent=4)

    print(f"[OK] Ficheiro COCO JSON para classificação guardado em: {out_file}")

if __name__ == "__main__":
    main()