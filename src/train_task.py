import torch
from torch.utils.data import DataLoader
from arch_optim import get_architecture, get_optimizer


def train_task(device, task_name, task_data, task_params):
    """
    Função encapsulada para treinar um modelo para uma tarefa específica.

    Parâmetros:
    - device: dispositivo a ser utilizado (CPU ou GPU).
    - task_name: nome da tarefa (ex.: 'classificacao_nasal').
    - task_data: dados específicos da tarefa (Dataset ou DataLoader).
    - task_params: parâmetros específicos da tarefa (ex.: arquitetura, otimizador).

    Retorna:
    - model: modelo treinado.
    - resultados: métricas de desempenho (ex.: acurácia, loss final).
    """
    print(f"Treinando a tarefa: {task_name}")

    # Obter arquitetura e otimizador
    model = get_architecture(
        task_params["architecture"],
        in_channels=task_params["in_channels"],
        out_classes=task_params["num_classes"],
        pretrained=task_params.get("use_transfer_learning", False)
    )
    model = model.to(device)

    optimizer = get_optimizer(task_params["optimizer"], model.parameters())

    # Definir critério de perda
    criterion = torch.nn.CrossEntropyLoss()

    # Criar DataLoader
    dataloader = DataLoader(task_data, batch_size=task_params["batch_size"], shuffle=True)

    # Treinamento
    num_epochs = task_params["num_epochs"]
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zerar gradientes
            optimizer.zero_grad()

            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Tarefa [{task_name}] - Época [{epoch+1}/{num_epochs}] - Loss: {running_loss / len(dataloader):.4f}")

    print(f"Treinamento finalizado para a tarefa: {task_name}")
    return model, {"loss": running_loss / len(dataloader)}
