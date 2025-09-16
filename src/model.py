import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class MultiTaskModel(nn.Module):
    def __init__(self, backbone_name, task_heads):
        """
        Modelo multi-tarefa com backbone compartilhado e cabeças específicas.

        Parâmetros:
        - backbone_name: Nome do backbone (ex.: 'resnet18').
        - task_heads: Dicionário com a configuração de cabeças de tarefas:
            {
                'task1': (input_dim, num_classes),
                'task2': (input_dim, num_classes)
            }
        """
        super(MultiTaskModel, self).__init__()
        
        # Definir backbone compartilhado
        self.backbone = getattr(models, backbone_name)(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remover a última camada fully-connected

        # Criar cabeças específicas para cada tarefa
        self.task_heads = nn.ModuleDict({
            task_name: nn.Linear(input_dim, num_classes)
            for task_name, (input_dim, num_classes) in task_heads.items()
        })

    def forward(self, x):
        """
        Forward do modelo.
        Retorna um dicionário com as saídas para cada tarefa.
        """
        # Extrair características compartilhadas
        features = self.backbone(x)

        # Calcular as saídas para cada tarefa
        outputs = {task: head(features) for task, head in self.task_heads.items()}
        return outputs
