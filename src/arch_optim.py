import torch
import torch.optim as optim
from torchvision import models

def get_architecture(architecture, out_classes, pretrained=True, in_channels=3):
    """
    Carrega uma arquitetura de modelo pré-treinada do torchvision.

    Parâmetros:
    - architecture (str): Nome do modelo (ex: 'resnet18').
    - out_classes (int): Número de classes de saída.
    - pretrained (bool): Se deve usar pesos pré-treinados.
    - in_channels (int): Número de canais de entrada (geralmente 3 para RGB).

    Retorna:
    - model: O modelo PyTorch.
    """
    # Carrega o modelo com os pesos pré-treinados
    model = getattr(models, architecture)(pretrained=pretrained)

    # Modifica a primeira camada se o número de canais de entrada não for 3
    if in_channels != 3:
        # Para ResNet e arquiteturas similares
        if hasattr(model, 'conv1'):
            original_conv1 = model.conv1
            model.conv1 = torch.nn.Conv2d(
                in_channels,
                original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=original_conv1.bias
            )
        # Adicione aqui outras lógicas para diferentes arquiteturas se necessário

    # Modifica a última camada (classificador) para o número de classes desejado
    if hasattr(model, 'fc'): # Para ResNet, etc.
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, out_classes)
    elif hasattr(model, 'classifier'): # Para MobileNetV2, etc.
        # A lógica pode variar um pouco dependendo do modelo
        if isinstance(model.classifier, torch.nn.Sequential):
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(num_ftrs, out_classes)
        else: # Caso seja uma única camada
            num_ftrs = model.classifier.in_features
            model.classifier = torch.nn.Linear(num_ftrs, out_classes)
    else:
        raise Exception("Arquitetura não suportada para modificação automática da última camada.")

    return model

def get_optimizer(optimizer_name, parameters, lr=1e-3):
    """
    Cria um otimizador com base no nome.

    Parâmetros:
    - optimizer_name (str): Nome do otimizador ('adam' ou 'sgd').
    - parameters: Parâmetros do modelo para otimizar (model.parameters()).
    - lr (float): Taxa de aprendizado.

    Retorna:
    - optimizer: O otimizador PyTorch.
    """
    if optimizer_name.lower() == 'adam':
        return optim.Adam(parameters, lr=lr)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(parameters, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Otimizador '{optimizer_name}' não suportado.")