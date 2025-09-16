import torch
import torch.optim as optim
from torchvision import models
import torch.nn as nn

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
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
    Carrega uma arquitetura e adapta-a para a tarefa especificada nos parâmetros.
    """
    task_type = task_params.get("task_type", "classification")
    out_classes = task_params["num_classes"]
    
    if task_type == 'classification':
        model = getattr(models, architecture)(weights='IMAGENET1K_V1' if pretrained else None)
        
        if hasattr(model, 'fc'):
            num_ftrs = model.fc.in_features
            
            if "classifier_head" in task_params:
                print("Construindo cabeça de classificação personalizada...")
                head_params = task_params["classifier_head"]
                layer1_neurons = head_params["layer1_neurons"]
                layer2_neurons = head_params["layer2_neurons"]
                dropout = head_params["dropout"]
                
                model.fc = nn.Sequential(
                    nn.Linear(num_ftrs, layer1_neurons),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(layer1_neurons, layer2_neurons),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(layer2_neurons, out_classes)
                )
            else:
                print("Usando cabeça de classificação padrão (camada única).")
                model.fc = nn.Linear(num_ftrs, out_classes)
        return model

    elif task_type == 'segmentation':
        backbone = getattr(models, architecture)(weights='IMAGENET1K_V1' if pretrained else None)
        modules = list(backbone.children())[:-2]
        
        num_backbone_out_channels = 512 if architecture in ['resnet18', 'resnet34'] else 2048
        
        model_body = nn.Sequential(*modules)
        
        # --- CORREÇÃO AQUI ---
        # Adicionámos o argumento 'out_channels' que estava em falta.
        segmentation_head = SegmentationHead(in_channels=num_backbone_out_channels, out_channels=out_classes)
        
        full_model = nn.Sequential(model_body, segmentation_head)
        return full_model
        
    else:
        raise ValueError(f"Tipo de tarefa '{task_type}' não suportado.")

def get_optimizer(optimizer_name, parameters, lr=1e-3):
    if optimizer_name.lower() == 'adam':
        return optim.Adam(parameters, lr=lr)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(parameters, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Otimizador '{optimizer_name}' não suportado.")