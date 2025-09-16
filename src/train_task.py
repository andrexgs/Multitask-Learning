import torch
from torch.utils.data import DataLoader
from arch_optim import get_architecture, get_optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math

def train_task(device, task_name, task_data, task_params):
    print(f"\nIniciando treino e validação para a tarefa: {task_name}")

    task_type = task_params.get("task_type", "classification")

    train_dataloader = task_data['train']
    val_dataloader = task_data['val']

    # --- ALTERAÇÃO CORRIGIDA AQUI ---
    # Agora passamos o dicionário 'task_params' completo, como a função espera.
    model = get_architecture(
        architecture=task_params["architecture"],
        task_params=task_params,
        pretrained=task_params.get("use_transfer_learning", True)
    ).to(device)

    optimizer = get_optimizer(task_params["optimizer"], model.parameters())
    
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

    if task_type == 'classification':
        criterion = torch.nn.CrossEntropyLoss()
    elif task_type == 'segmentation':
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Função de perda para a tarefa '{task_type}' não definida.")

    num_epochs = task_params["num_epochs"]
    paciencia = task_params.get("paciencia", 5)
    melhor_metrica_val = -math.inf
    total_sem_melhora = 0
    melhores_resultados = {}
    caminho_modelo_salvo = f"melhor_modelo_{task_name}.pth"

    for epoch in range(num_epochs):
        print(f"-------------------------------\nÉpoca {epoch+1}/{num_epochs}")
        
        model.train()
        for data in train_dataloader:
            if task_type == 'classification':
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
            elif task_type == 'segmentation':
                inputs, labels = data['image'], data['mask']
                inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss, correct, total_val_samples = 0, 0, 0
        with torch.no_grad():
            for data in val_dataloader:
                if task_type == 'classification':
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_val_samples += labels.size(0)
                    correct += (predicted == labels).sum().item()
                elif task_type == 'segmentation':
                    inputs, labels = data['image'], data['mask']
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        metrica_atual = 0
        
        if task_type == 'classification':
            acuracia_val = correct / total_val_samples
            metrica_atual = acuracia_val
            print(f"\n>> RESULTADOS DA ÉPOCA {epoch+1}: Acurácia de Validação: {(100*acuracia_val):.2f}% | Perda: {avg_val_loss:.4f}")
        elif task_type == 'segmentation':
            metrica_atual = -avg_val_loss
            print(f"\n>> RESULTADOS DA ÉPOCA {epoch+1}: Perda de Validação: {avg_val_loss*100:.2f}%")
        
        scheduler.step(metrica_atual)

        if metrica_atual > melhor_metrica_val:
            print(f">>> Métrica melhorou. Salvando modelo...")
            melhor_metrica_val = metrica_atual
            torch.save(model.state_dict(), caminho_modelo_salvo)
            total_sem_melhora = 0
            melhores_resultados = {
                "metrica_validacao": f"{(100*acuracia_val):.2f}%" if task_type == 'classification' else avg_val_loss,
                "epoca_encontrada": epoch + 1
            }
        else:
            total_sem_melhora += 1
            print(f">>> Métrica não melhorou. Paciência: {total_sem_melhora}/{paciencia}")

        if total_sem_melhora >= paciencia:
            print(f"\nPARAGEM ANTECIPADA!")
            break
            
    print(f"\nTreinamento finalizado para a tarefa: {task_name}")
    print(f"Carregando o melhor modelo salvo em '{caminho_modelo_salvo}'")
    model.load_state_dict(torch.load(caminho_modelo_salvo))
    
    return model, melhores_resultados