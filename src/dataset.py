import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import json # <--- ADICIONADO IMPORT NECESSÁRIO

class MultiTaskDataset(Dataset):
    """
    Classe de Dataset multi-tarefa.
    Esta classe assume que cada tarefa possui:
    - Uma pasta de imagens.
    - Um arquivo CSV com nome do arquivo e seu respectivo rótulo.

    Cada elemento do dataset retornará um dicionário contendo:
    {
        'imagens': imagem transformada,
        'labels': { 'task1': label_task1, 'task2': label_task2, ... }
    }

    Isso permite facilmente adicionar novas tarefas,
    desde que se forneçam os caminhos e labels correspondentes.
    """

    def __init__(self, task_dirs, transform=None) -> None:
        """
        Construtor do dataset multi-tarefa.
        
        Parâmetros:
        - task_dirs (dict): dicionário com o nome da tarefa como chave e 
                            o caminho da pasta de dados como valor.
                            Ex: {
                                'task1': 'data/task1',
                                'task2': 'data/task2'
                            }
        - transform: transformações aplicadas às imagens.
        """
        # Lista de tarefas
        self.tasks = list(task_dirs.keys())

        # Dicionário para armazenar o DataFrame de cada tarefa
        # Cada DataFrame terá as colunas ['filename', 'label']
        self.task_data = {}

        # Carregar informações das tarefas
        for task_name, dir_path in task_dirs.items():
            # Caminho do CSV de labels
            csv_path = os.path.join(dir_path, 'labels.csv')
            df = pd.read_csv(csv_path)
            self.task_data[task_name] = df

        # Supondo que todas as tarefas tenham o mesmo número de exemplos e as imagens
        # estejam pareadas entre tarefas (caso contrário, precisa-se definir outra lógica)
        # Aqui assumimos que o número de amostras para todas as tarefas é o mesmo.
        self.length = len(self.task_data[self.tasks[0]])

        self.transform = transform

        # Armazenamos o caminho das imagens da primeira tarefa apenas
        # assumindo que as imagens base para indexar o dataset estão lá.
        # Caso as tarefas tenham imagens diferentes, será necessário definir um critério.
        self.base_task = self.tasks[0]
        self.base_dir = task_dirs[self.base_task]
        self.base_df = self.task_data[self.base_task]

    def __len__(self):
        # Retorna a quantidade total de amostras no dataset.
        return self.length

    def __getitem__(self, idx) -> dict:
        # Método para obter um item (amostra) específico do dataset.

        # 1. Obter nome do arquivo da imagem base
        img_name = self.base_df.iloc[idx]['filename']
        # Caminho completo da imagem
        img_path = os.path.join(self.base_dir, 'imagens', img_name)
        
        # 2. Carregar a imagem
        image = Image.open(img_path).convert('RGB')

        # 3. Aplicar transformações, se houver
        if self.transform:
            image = self.transform(image)

        # 4. Construir um dicionário de labels para cada tarefa
        labels = {}
        for task_name in self.tasks:
            label = self.task_data[task_name].iloc[idx]['label']
            labels[task_name] = label

        # 5. Retornar um dicionário com a imagem e os labels
        return {
            'imagem': image,
            'labels': labels
        }

# --- CÓDIGO NOVO ADICIONADO ABAIXO ---

class ClassificationCocoDataset(Dataset):
    """
    Dataset para carregar dados de classificação a partir de um
    arquivo COCO JSON, como o gerado pelo script de normalização.
    """
    def __init__(self, json_path, data_root, transform=None):
        """
        Construtor do dataset.
        
        Parâmetros:
        - json_path (str): Caminho para o arquivo COCO JSON.
        - data_root (str): Caminho para a pasta onde as imagens estão.
        - transform: Transformações a serem aplicadas nas imagens.
        """
        with open(json_path, 'r') as f:
            coco_data = json.load(f)

        self.transform = transform
        self.data_root = data_root

        # Encontra a pasta de cada imagem
        self.image_paths = {}
        for root, _, files in os.walk(data_root):
            for file in files:
                self.image_paths[file] = root

        # Mapeia image_id para file_name
        images_info = {img['id']: img for img in coco_data['images']}
        
        self.annotations = []
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id in images_info:
                img_info = images_info[image_id]
                file_name = img_info['file_name']
                
                if file_name in self.image_paths:
                    self.annotations.append({
                        'file_name': file_name,
                        'category_id': ann['category_id']
                    })

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        
        file_name = ann['file_name']
        # Constrói o caminho completo da imagem
        img_path = os.path.join(self.image_paths[file_name], file_name)
        
        image = Image.open(img_path).convert('RGB')
        label = ann['category_id']

        if self.transform:
            image = self.transform(image)
            
        return image, label

class SegmentationDataset(Dataset):
    """
    Dataset para carregar pares de imagem e máscara para tarefas de segmentação.
    Assume que os nomes dos ficheiros de imagem e máscara são os mesmos.
    """
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Construtor do dataset de segmentação.
        
        Parâmetros:
        - image_dir (str): Caminho para a pasta de imagens.
        - mask_dir (str): Caminho para a pasta de máscaras (anotações).
        - transform: Transformações a serem aplicadas na imagem e na máscara.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # Lista apenas os ficheiros de imagem, assumindo que cada um tem uma máscara correspondente
        self.images = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Nome do ficheiro da imagem
        img_name = self.images[idx]
        
        # Caminho completo para a imagem e a máscara
        img_path = os.path.join(self.image_dir, img_name)
        
        # Constrói o caminho da máscara, tentando extensões comuns
        mask_name = os.path.splitext(img_name)[0] + ".png" # Assume que as máscaras são .png
        mask_path = os.path.join(self.mask_dir, mask_name)
        if not os.path.exists(mask_path):
            # Tenta com a mesma extensão da imagem se .png não existir
            mask_name = img_name
            mask_path = os.path.join(self.mask_dir, mask_name)

        # Carrega a imagem e a máscara
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") # Converte para grayscale (preto e branco)

        sample = {'image': image, 'mask': mask}

        if self.transform:
            # NOTA: A transformação para segmentação é mais complexa,
            # vamos usar uma simples por agora e podemos melhorar depois.
            image = self.transform(image)
            # Para a máscara, geralmente só redimensionamos e convertemos para tensor
            mask_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            mask = mask_transform(mask)
            # Normaliza a máscara para ter valores 0 ou 1
            mask = (mask > 0.5).float()
            
            sample = {'image': image, 'mask': mask}

        return sample