import torch
from torch.utils.data import DataLoader
from arch_optim import get_architecture, get_optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math

def train_task(device, task_name, task_data, task_params):
    """
    Função principal que executa o ciclo completo de treino e validação para uma única tarefa.
    """
    print(f"\nIniciando treino e validação para a tarefa: {task_name}")

    # Determina o tipo de tarefa a partir dos parâmetros (default é 'classification').
    task_type = task_params.get("task_type", "classification")

    # Separa os dataloaders de treino e validação.
    train_dataloader = task_data['train']
    val_dataloader = task_data['val']

    # Constrói o modelo com base nos parâmetros definidos em train.py.
    model = get_architecture(
        architecture=task_params["architecture"],
        task_params=task_params,
        pretrained=task_params.get("use_transfer_learning", True)
    ).to(device)

    # Cria o otimizador, passando a taxa de aprendizagem definida em train.py.
    lr = task_params.get("learning_rate", 1e-3)
    optimizer = get_optimizer(task_params["optimizer"], model.parameters(), lr=lr)
    
    # Cria o agendador de taxa de aprendizagem.
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

    # Seleciona a função de perda apropriada para a tarefa.
    if task_type == 'classification':
        criterion = torch.nn.CrossEntropyLoss()
    elif task_type == 'segmentation':
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Função de perda para a tarefa '{task_type}' não definida.")

    # Parâmetros para o ciclo de treino e paragem antecipada.
    num_epochs = task_params["num_epochs"]
    paciencia = task_params.get("paciencia", 7)
    melhor_metrica_val = -math.inf
    total_sem_melhora = 0
    melhores_resultados = {}
    caminho_modelo_salvo = f"melhor_modelo_{task_name}.pth"

    # Inicia o ciclo de treino por épocas.
    for epoch in range(num_epochs):
        print(f"-------------------------------\nÉpoca {epoch+1}/{num_epochs}")
        
        # --- FASE DE TREINO ---
        model.train() # Coloca o modelo em modo de treino.
        for data in train_dataloader:
            # Desempacota os dados de acordo com o tipo de tarefa.
            if task_type == 'classification':
                inputs, labels = data
            elif task_type == 'segmentation':
                inputs, labels = data['image'], data['mask']
            
            inputs, labels = inputs.to(device), labels.to(device)

            # Ciclo padrão de treino: zera gradientes, faz a predição, calcula a perda, retropropaga.
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # --- FASE DE VALIDAÇÃO ---
        model.eval() # Coloca o modelo em modo de avaliação.
        val_loss, correct, total_val_samples = 0, 0, 0
        with torch.no_grad(): # Desativa o cálculo de gradientes para poupar memória e tempo.
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

        # --- CÁLCULO DE MÉTRICAS E LÓGICA DE PARAGEM ---
        avg_val_loss = val_loss / len(val_dataloader)
        
        if task_type == 'classification':
            acuracia_val = correct / total_val_samples
            metrica_atual = acuracia_val # Para classificação, a métrica a ser otimizada é a acurácia.
            print(f"\n>> RESULTADOS DA ÉPOCA {epoch+1}: Acurácia de Validação: {(100*acuracia_val):.2f}% | Perda: {avg_val_loss:.4f}")
        elif task_type == 'segmentation':
            metrica_atual = -avg_val_loss # Para segmentação, queremos minimizar a perda. Usamos o negativo para que a lógica de "melhoria" funcione.
            print(f"\n>> RESULTADOS DA ÉPOCA {epoch+1}: Perda de Validação: {avg_val_loss:.4f}")
        
        # Atualiza o agendador com a métrica da época atual.
        scheduler.step(metrica_atual)

        # Verifica se a performance melhorou.
        if metrica_atual > melhor_metrica_val:
            print(f">>> Métrica melhorou. A guardar o modelo...")
            melhor_metrica_val = metrica_atual
            torch.save(model.state_dict(), caminho_modelo_salvo)
            total_sem_melhora = 0 # Zera o contador de paciência.
            melhores_resultados = {
                "metrica_validacao": f"{(100*acuracia_val):.2f}%" if task_type == 'classification' else avg_val_loss,
                "epoca_encontrada": epoch + 1
            }
        else:
            total_sem_melhora += 1
            print(f">>> Métrica não melhorou. Paciência: {total_sem_melhora}/{paciencia}")

        # Se não houver melhoria por 'paciencia' épocas, para o treino.
        if total_sem_melhora >= paciencia:
            print(f"\nPARAGEM ANTECIPADA!")
            break
            
    print(f"\nTreino finalizado para a tarefa: {task_name}")
    print(f"A carregar o melhor modelo guardado em '{caminho_modelo_salvo}'")
    model.load_state_dict(torch.load(caminho_modelo_salvo))
    
    return model, melhores_resultados