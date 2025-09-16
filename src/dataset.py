import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

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
