import torch.nn as nn
from torchvision import models

class MultiTaskModel(nn.Module):
    """
    Modelo multi-tarefa com um "backbone" (corpo) partilhado e "cabeças" específicas para cada tarefa.
    A ideia é que o backbone aprenda características visuais gerais (como texturas, formas, etc.)
    que são úteis para todas as tarefas, enquanto cada cabeça se especializa na sua tarefa final
    (ex: uma para classificar, outra para segmentar).
    """
    def __init__(self, backbone_name, task_heads):
        """
        Construtor do modelo multi-tarefa.

        Parâmetros:
        - backbone_name (str): Nome do backbone a ser usado (ex: 'resnet18').
        - task_heads (dict): Dicionário que define as cabeças de cada tarefa.
                             Ex: {'classificacao': nn.Linear(512, 4), 'segmentacao': SegmentationHead(512, 1)}
        """
        super(MultiTaskModel, self).__init__()
        
        # 1. Carregar o Backbone Pré-treinado
        # Usamos um modelo como o ResNet, pré-treinado na ImageNet, como extrator de características.
        self.backbone = getattr(models, backbone_name)(weights='IMAGENET1K_V1')
        
        # 2. Remover a Camada de Classificação Original do Backbone
        # A última camada (fully-connected) da ResNet original é removida,
        # pois vamos adicionar as nossas próprias cabeças de tarefa.
        # Para a ResNet, o corpo que queremos termina antes da camada 'fc'.
        if hasattr(self.backbone, 'fc'):
            self.backbone.fc = nn.Identity()  # nn.Identity() é uma camada que não faz nada.

        # 3. Criar as Cabeças Específicas para cada Tarefa
        # O nn.ModuleDict permite-nos manter um dicionário de camadas/módulos do PyTorch,
        # o que é perfeito para gerir as diferentes cabeças de tarefa.
        self.task_heads = nn.ModuleDict(task_heads)

    def forward(self, x):
        """
        Define como os dados passam através do modelo (forward pass).
        """
        # 1. Extrair as características partilhadas da imagem usando o backbone.
        features = self.backbone(x)

        # 2. Passar as características por cada cabeça de tarefa para obter as saídas.
        # O resultado é um dicionário onde cada chave é o nome da tarefa e o valor é a sua predição.
        outputs = {task_name: head(features) for task_name, head in self.task_heads.items()}
        
        return outputs