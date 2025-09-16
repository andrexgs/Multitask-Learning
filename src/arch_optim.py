import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import models

class SegmentationHead(nn.Module):
    """
    "Cabeça" de rede neural para tarefas de segmentação.
    Esta classe pega na saída de baixa resolução de um backbone (como o ResNet)
    e usa uma série de convoluções transpostas (upsampling) para reconstruir
    uma máscara de segmentação com a mesma resolução da imagem de entrada.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Sequência de camadas para aumentar a resolução passo a passo.
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1)
        )
    def forward(self, x):
        return self.upsample(x)

def get_architecture(architecture, task_params, pretrained=True):
    """
    Função principal que constrói e retorna o modelo de rede neural para uma tarefa.
    
    Parâmetros:
    - architecture (str): Nome do backbone (ex: 'resnet18').
    - task_params (dict): Dicionário de parâmetros da tarefa, vindo do train.py.
    - pretrained (bool): Se deve carregar pesos pré-treinados na ImageNet.
    """
    task_type = task_params.get("task_type", "classification")
    out_classes = task_params["num_classes"]
    
    # --- LÓGICA PARA MODELO DE CLASSIFICAÇÃO ---
    if task_type == 'classification':
        # Carrega a arquitetura especificada a partir do torchvision.
        model = getattr(models, architecture)(weights='IMAGENET1K_V1' if pretrained else None)
        
        # A camada final da ResNet chama-se 'fc'. Vamos substituí-la.
        if hasattr(model, 'fc'):
            num_ftrs = model.fc.in_features # Número de neurónios que saem do corpo da ResNet.
            
            # Se uma cabeça personalizada foi definida nos parâmetros, constrói-a.
            if "classifier_head" in task_params:
                print("A construir cabeça de classificação personalizada...")
                head_params = task_params["classifier_head"]
                
                model.fc = nn.Sequential(
                    nn.Linear(num_ftrs, head_params["layer1_neurons"]),
                    nn.ReLU(),
                    nn.Dropout(head_params["dropout"]),
                    nn.Linear(head_params["layer1_neurons"], head_params["layer2_neurons"]),
                    nn.ReLU(),
                    nn.Dropout(head_params["dropout"]),
                    nn.Linear(head_params["layer2_neurons"], out_classes)
                )
            else:
                # Caso contrário, usa uma cabeça de classificação simples com uma única camada.
                print("A usar cabeça de classificação padrão (camada única).")
                model.fc = nn.Linear(num_ftrs, out_classes)
        return model

    # --- LÓGICA PARA MODELO DE SEGMENTAÇÃO ---
    elif task_type == 'segmentation':
        # Carrega o backbone pré-treinado.
        backbone = getattr(models, architecture)(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Removemos as últimas duas camadas (avgpool e fc) para usar apenas o corpo extrator de características.
        modules = list(backbone.children())[:-2]
        model_body = nn.Sequential(*modules)
        
        # Define o número de canais de saída do corpo do backbone.
        num_backbone_out_channels = 512 if architecture in ['resnet18', 'resnet34'] else 2048
        
        # Cria a cabeça de segmentação.
        segmentation_head = SegmentationHead(in_channels=num_backbone_out_channels, out_channels=out_classes)
        
        # Combina o corpo e a cabeça para formar o modelo final.
        full_model = nn.Sequential(model_body, segmentation_head)
        return full_model
        
    else:
        raise ValueError(f"Tipo de tarefa '{task_type}' não suportado.")

def get_optimizer(optimizer_name, parameters, lr=1e-3):
    """
    Cria um otimizador com base no nome e na taxa de aprendizagem.
    """
    if optimizer_name.lower() == 'adam':
        return optim.Adam(parameters, lr=lr)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(parameters, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Otimizador '{optimizer_name}' não suportado.")