"""
    Converte um conjunto de dados compostos por imagens e máscaras binárias para o formato COCO JSON.
"""

import argparse
import cv2
import json
import numpy as np
import os


def parse_args() -> dict[str, str]:
    """
    Processa os argumentos da linha de comando para converter máscaras de dataset para o formato CSV.

    Returns:
        dict[str, str]: um dicionário contendo o formato e o caminho do dataset.
    """

    parser = argparse.ArgumentParser(description="Converte dataset de segmentação com imagens e máscaras binárias para o formato COCO JSON.")

    # Argumento para o formato atual do dataset
    parser.add_argument(
        '-f',
        '--dataset_format',
        type=str,
        choices=['imagem_mascara_binaria'],
        required=True,
        help='Formato atual do dataset. Atualmente, apenas "imagem_mascara_binaria" é suportado.'
    ) 

    # Argumento para o caminho do dataset
    parser.add_argument(
        '-d',
        '--dataset_path',
        type=str,
        required=True,
        help='Caminho para a raiz do dataset (ex.: pasta que contém as pastas imagens e anotacoes).'
    )

    parser.add_argument(
        '-o',
        '--output_dir',
        type=str,
        default='.',
        help='Diretório onde o arquivo JSON será salvo.'
    )

    # Argumentos dependentes do formato do dataset
    # Diretório de imagens
    parser.add_argument(
        '-i',
        '--image_dir',
        type=str,
        default='imagens',
        help='Nome do diretório que contém as imagens. Padrão: "imagens".'
    )

    # Diretório de anotações
    parser.add_argument(
        '-a',
        '--annotation_dir',
        type=str,
        default='anotacoes',
        help='Nome do diretório que contém as anotações. Padrão: "anotacoes".'
    )

    # Pega os argumentos e armazena em um dicionário
    args = parser.parse_args()
    return vars(args)


def lista_imagens(dataset_path: str) -> list[str]:
    """
    Lista todas as imagens no diretório do dataset.

    Args:
        dataset_path (str): Caminho para o diretório do dataset.

    Returns:
        list[str]: Lista de caminhos completos das imagens.
    """
    return os.listdir(dataset_path)


def associa_imagens_mascaras(image_files: list[str], annotation_files: list[str]) -> list[tuple[str, str]]:
    """
    Associa imagens com suas respectivas máscaras.

    Args:
        image_files (list[str]): Lista de arquivos de imagem.
        annotation_files (list[str]): Lista de arquivos de máscara.

    Returns:
        list[tuple[str, str]]: Lista de tuplas associando cada imagem à sua máscara.
    """

    imagens_mascaras = list()
    for image_file in image_files:
        # Verifica se a máscara correspondente existe
        mask_file = image_file.replace('.jpg', '.png')  # Supondo que as máscaras sejam PNG
        if mask_file in annotation_files:
            imagens_mascaras.append((image_file, mask_file))

    return imagens_mascaras


def converte_mask_para_pontos(mask_path: str) -> list[list[int]]:
    """
    Converte uma máscara binária em uma lista de pontos (contornos). Atualmente, funciona para máscaras com somente um contorno.

    Args:
        mask_path (str): Caminho para o arquivo da máscara.

    Returns:
        list[list[int]]: Lista de pontos representando o contorno da máscara.
    """
    
    # Lê a máscara
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Extrai os contornos da máscara
    contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contour[0]

    # Converte o contorno para uma lista de pontos
    pontos = []
    for ponto in contour:
        xy = ponto[0].tolist()  # Extrai as coordenadas x e y do ponto
        pontos.append(xy)

    return pontos


def cria_dict_imagem(image_file: str, image_id: int) -> dict[str, int | str]:
    """
    Cria um dicionário representando uma imagem no formato COCO.

    Args:
        image_file (str): Caminho do arquivo de imagem.
        image_id (int): ID único da imagem.

    Returns:
        dict[str, int | str]: Dicionário representando a imagem.
    """

    # Lê a imagem para obter suas dimensões
    image = cv2.imread(image_file)

    if image is None: raise FileNotFoundError(f"Imagem '{image_file}' não encontrada no caminho especificado.")

    height, width = image.shape[:2]

    return {
        "id": image_id,
        "file_name": os.path.basename(image_file),
        "width": width,
        "height": height 
    }


def cria_dict_annotation(image_id: int, annotation_id: int, pontos: list[list[int]]) -> dict[str, int | str | list[int]]:
    """
    Cria um dicionário representando uma anotação no formato COCO.

    Args:
        image_id (int): ID da imagem associada.
        annotation_id (int): ID único da anotação.
        pontos (list[list[int]]): Lista de pontos representando o contorno da máscara.

    Returns:
        dict[str, int | str | list[int]]: Dicionário representando a anotação.
    """

    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,  # Assumindo que todas as máscaras pertencem à mesma categoria
        "segmentation": [pontos],
        "area": cv2.contourArea(np.array(pontos)),  # Área do contorno
        "bbox": list(cv2.boundingRect(np.array(pontos))),  # Caixa delimitadora
        "iscrowd": 0  # Assumindo que não há multidões
    }


def normaliza_dataset_imagem_mascara_binaria(args: dict[str, str]):
    """
    Normaliza um dataset de segmentação composto por imagens e máscaras binárias para o formato COCO JSON. Neste formato, cada imagem tem uma única máscara associada (uma única anotação, portanto).

    Args:
        dataset_path (str): Caminho para o diretório do dataset.

    Returns:
        dict[str, list[dict[str, int | str | list[int]]]]: Dicionário no formato COCO JSON.
    """

    # Lista imagens e máscaras
    image_dir = os.path.join(args['dataset_path'], args['image_dir'])
    annotation_dir = os.path.join(args['dataset_path'], args['annotation_dir'])
    image_files = lista_imagens(image_dir)
    annotation_files = lista_imagens(annotation_dir)

    # Associa imagens com máscaras
    imagens_mascaras = associa_imagens_mascaras(image_files, annotation_files)

    images = []
    annotations = []
    for image_id, (image_file, mask_file) in enumerate(imagens_mascaras):
        # Cria o dicionário da imagem
        image_dict = cria_dict_imagem(os.path.join(image_dir, image_file), image_id)
        images.append(image_dict)

        # Converte a máscara para pontos
        mask_path = os.path.join(annotation_dir, mask_file)
        pontos = converte_mask_para_pontos(mask_path)

        # Cria o dicionário da anotação
        annotation_dict = cria_dict_annotation(image_id, image_id, pontos)
        annotations.append(annotation_dict)

    return images, annotations


def determina_categorias(**kwargs) -> list[dict[str, int | str]]:
    """
    Determina as categorias do dataset. Atualmente, assume que todas as máscaras pertencem à mesma categoria.

    Returns:
        list[dict[str, int | str]]: Lista de categorias no formato COCO.
    """
    return [
        {"id": 0, "name": "background", "supercategory": ""},
        {"id": 1, "name": "object", "supercategory": ""}
    ]


def salva_json_coco(coco_dict: dict[str, list[dict[str, int | str | list[int]]]], output_path: str) -> None:
    """
    Salva o dicionário COCO em um arquivo JSON.

    Args:
        coco_dict (dict[str, list[dict[str, int | str | list[int]]]]): Dicionário no formato COCO.
        output_path (str): Caminho para salvar o arquivo JSON.
    """

    with open(output_path, 'w') as f:
        json.dump(coco_dict, f, indent=4)
    
    return


def main() -> None:
    args = parse_args()

    coco_dict: dict[str, list[dict[str, int | str | list[int]]]]  # Formato normal do dataset
    coco_dict = {
        "images": list(),
        "annotations": list(),
        "categories": list()
    }

    # Chama a função de normalização do dataset
    if args['dataset_format'] == 'imagem_mascara_binaria':
        images, annotations = normaliza_dataset_imagem_mascara_binaria(args)
    else: raise ValueError(f"Formato de dataset '{args['dataset_format']}' não suportado.")
    
    coco_dict['images'] = images
    coco_dict['annotations'] = annotations
    coco_dict['categories'] = determina_categorias()

    # Salva o dicionário COCO em um arquivo JSON
    output_path = os.path.join(args["output_dir"], 'segmentacao.json')
    salva_json_coco(coco_dict, output_path)

    print(f"Dataset normalizado e salvo em: {output_path}")

    return
    

if __name__ == "__main__":
    main()
