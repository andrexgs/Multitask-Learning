import argparse
import cv2
import json
import numpy as np
import os

def parse_args() -> dict:
    """
    Analisa e processa os argumentos fornecidos através da linha de comandos.
    """
    parser = argparse.ArgumentParser(description="Converte um dataset de segmentação (imagens e máscaras) para o formato COCO JSON.")
    parser.add_argument('-f', '--dataset_format', type=str, choices=['imagem_mascara_binaria'], required=True, help='Formato do dataset.')
    parser.add_argument('-d', '--dataset_path', type=str, required=True, help='Caminho para a pasta raiz do dataset (ex: data/segmentacao/Narina).')
    parser.add_argument('-o', '--output_dir', type=str, default='.', help='Diretório onde o ficheiro JSON será guardado.')
    parser.add_argument('-i', '--image_dir', type=str, default='imagens', help='Nome da subpasta de imagens.')
    parser.add_argument('-a', '--annotation_dir', type=str, default='anotacoes', help='Nome da subpasta de anotações (máscaras).')
    
    return vars(parser.parse_args())

def converte_mask_para_pontos(mask_path: str) -> list:
    """
    Lê uma máscara binária e extrai os contornos do objeto.
    Retorna uma lista de pontos (x, y) que definem o polígono de segmentação.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Converte os contornos para o formato de lista de pontos [x1, y1, x2, y2, ...]
    segmentation = []
    if contours:
        segmentation = contours[0].flatten().tolist()
    return segmentation

def main() -> None:
    """
    Função principal que executa o processo de conversão.
    """
    args = parse_args()
    
    image_dir = os.path.join(args['dataset_path'], args['image_dir'])
    annotation_dir = os.path.join(args['dataset_path'], args['annotation_dir'])
    
    images_list = []
    annotations_list = []
    image_id_counter = 0

    # 1. Itera sobre todas as imagens na pasta de imagens
    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(image_dir, fname)
        
        # 2. Procura pela máscara correspondente
        mask_name = os.path.splitext(fname)[0] + ".png"
        mask_path = os.path.join(annotation_dir, mask_name)
        if not os.path.exists(mask_path):
            print(f"Aviso: Não foi encontrada máscara correspondente para a imagem {fname}. A ignorar.")
            continue

        # 3. Extrai metadados da imagem
        image = cv2.imread(img_path)
        if image is None: continue
        height, width, _ = image.shape

        images_list.append({
            "id": image_id_counter,
            "file_name": fname,
            "width": width,
            "height": height
        })

        # 4. Converte a máscara em polígono e cria a anotação
        segmentation_points = converte_mask_para_pontos(mask_path)
        if segmentation_points:
            # Converte o polígono em formato de array numpy para calcular a área e a bounding box
            poly_np = np.array(segmentation_points).reshape(-1, 2)
            x, y, w, h = cv2.boundingRect(poly_np)
            area = cv2.contourArea(poly_np)

            annotations_list.append({
                "id": image_id_counter,
                "image_id": image_id_counter,
                "category_id": 1, # Assumimos uma única categoria "narina"
                "segmentation": [segmentation_points],
                "area": float(area),
                "bbox": [x, y, w, h],
                "iscrowd": 0
            })
        
        image_id_counter += 1

    # 5. Define as categorias (apenas uma neste caso)
    categories = [
        {"id": 1, "name": "narina", "supercategory": "narina"}
    ]

    # 6. Monta e guarda o ficheiro COCO JSON final
    coco_dict = {
        "images": images_list,
        "annotations": annotations_list,
        "categories": categories
    }

    output_path = os.path.join(args["output_dir"], 'segmentacao.json')
    with open(output_path, 'w') as f:
        json.dump(coco_dict, f, indent=4)
        
    print(f"[OK] Ficheiro COCO JSON para segmentação guardado em: {output_path}")

if __name__ == "__main__":
    main()