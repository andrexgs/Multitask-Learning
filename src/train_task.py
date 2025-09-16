import torch
from torch.utils.data import DataLoader
from arch_optim import get_architecture, get_optimizer
import math

def train_task(device, task_name, task_data, task_params):
    print(f"\nIniciando treino e validação para a tarefa: {task_name}")

    train_dataloader = task_data['train']
    val_dataloader = task_data['val']

    model = get_architecture(
        task_params["architecture"],
        in_channels=task_params["in_channels"],
        out_classes=task_params["num_classes"],
        pretrained=task_params.get("use_transfer_learning", True)
    ).to(device)

    optimizer = get_optimizer(task_params["optimizer"], model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    num_epochs = task_params["num_epochs"]
    paciencia = task_params.get("paciencia", 5)
    melhor_acuracia_val = -math.inf
    total_sem_melhora = 0
    melhores_resultados = {}
    caminho_modelo_salvo = f"melhor_modelo_{task_name}.pth"

    for epoch in range(num_epochs):
        print(f"-------------------------------\nÉpoca {epoch+1}/{num_epochs}")
        
        model.train()
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss, correct = 0, 0
        total_val_samples = 0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_val_samples += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_dataloader)
        acuracia_val = correct / total_val_samples
        
        print(f"\n>> RESULTADOS DA ÉPOCA {epoch+1}: Acurácia de Validação: {(100*acuracia_val):.2f}% | Perda: {avg_val_loss:.4f}")

        if acuracia_val > melhor_acuracia_val:
            print(f">>> Acurácia melhorou. Salvando modelo...")
            melhor_acuracia_val = acuracia_val
            torch.save(model.state_dict(), caminho_modelo_salvo)
            total_sem_melhora = 0
            melhores_resultados = {
                "melhor_acuracia": f"{(100*melhor_acuracia_val):.2f}%",
                "perda_validacao": avg_val_loss,
                "epoca_encontrada": epoch + 1
            }
        else:
            total_sem_melhora += 1
            print(f">>> Acurácia não melhorou. Paciência: {total_sem_melhora}/{paciencia}")

        if total_sem_melhora >= paciencia:
            print(f"\nPARAGEM ANTECIPADA! O modelo não melhora há {paciencia} épocas.")
            break
            
    print(f"\nTreinamento finalizado para a tarefa: {task_name}")
    print(f"Carregando o melhor modelo salvo em '{caminho_modelo_salvo}'")
    model.load_state_dict(torch.load(caminho_modelo_salvo))
    
    return model, melhores_resultados