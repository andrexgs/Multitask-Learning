# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from itertools import cycle
import math

# Importações dos seus outros ficheiros
from dataset import SegmentationDataset
from model import MultiTaskModel
from arch_optim import SegmentationHead # Importar a cabeça de segmentação

def train_multitask_simultaneously(device, model, dataloaders, criterions, optimizer, num_epochs=50, paciencia=10):
    """
    Função para treinar um modelo multi-tarefa de forma simultânea.
    """
    print("\n{'='*20}\nIniciando treino simultâneo para Múltiplas Tarefas\n{'='*20}")

    # Para early stopping
    melhor_metrica_val = -math.inf
    total_sem_melhora = 0
    caminho_modelo_salvo = "melhor_modelo_multitask.pth"

    # Determina qual dataloader é maior para o ciclo principal
    train_loader_cls = dataloaders['classificacao_narina']['train']
    train_loader_seg = dataloaders['segmentacao_narina']['train']
    
    # Usamos 'cycle' para repetir o dataloader menor
    if len(train_loader_cls) > len(train_loader_seg):
        main_loader = train_loader_cls
        secondary_loader = cycle(train_loader_seg)
        main_task_name = 'classificacao_narina'
        secondary_task_name = 'segmentacao_narina'
    else:
        main_loader = train_loader_seg
        secondary_loader = cycle(train_loader_cls)
        main_task_name = 'segmentacao_narina'
        secondary_task_name = 'classificacao_narina'


    for epoch in range(num_epochs):
        print(f"-------------------------------\nÉpoca {epoch+1}/{num_epochs}")

        # --- FASE DE TREINO ---
        model.train()
        total_loss_epoch = 0

        for i, (data_main, data_sec) in enumerate(zip(main_loader, secondary_loader)):
            optimizer.zero_grad()
            
            # Processar dados de classificação
            if main_task_name == 'classificacao_narina':
                inputs_cls, labels_cls = data_main
            else:
                inputs_cls, labels_cls = data_sec
            inputs_cls, labels_cls = inputs_cls.to(device), labels_cls.to(device)

            # Processar dados de segmentação
            if main_task_name == 'segmentacao_narina':
                 data_seg = data_main
            else:
                data_seg = data_sec
            inputs_seg, masks_seg = data_seg['image'].to(device), data_seg['mask'].to(device)

            # Forward pass para ambas as tarefas (usando um batch combinado para demonstração)
            # Numa implementação mais avançada, poderia fazer passes separados se as imagens não forem as mesmas
            inputs_combined = torch.cat([inputs_cls, inputs_seg], 0)
            
            outputs = model(inputs_combined)
            
            # Separar as saídas
            outputs_cls = outputs['classificacao_narina'][:len(inputs_cls)]
            outputs_seg = outputs['segmentacao_narina'][len(inputs_cls):]

            # Calcular as perdas
            loss_cls = criterions['classificacao_narina'](outputs_cls, labels_cls)
            loss_seg = criterions['segmentacao_narina'](outputs_seg, masks_seg)
            
            # Combinar as perdas (pode-se adicionar pesos aqui, ex: 0.5 * loss_cls + 0.5 * loss_seg)
            total_loss = loss_cls + loss_seg
            
            total_loss.backward()
            optimizer.step()
            total_loss_epoch += total_loss.item()
        
        print(f"Perda de Treino Média da Época: {total_loss_epoch / (i+1):.4f}")


        # --- FASE DE VALIDAÇÃO ---
        model.eval()
        val_loss_cls, correct_cls, total_cls = 0, 0, 0
        val_loss_seg = 0
        
        with torch.no_grad():
            # Validação da classificação
            for data in dataloaders['classificacao_narina']['val']:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)['classificacao_narina']
                loss = criterions['classificacao_narina'](outputs, labels)
                val_loss_cls += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_cls += labels.size(0)
                correct_cls += (predicted == labels).sum().item()

            # Validação da segmentação
            for data in dataloaders['segmentacao_narina']['val']:
                inputs, masks = data['image'].to(device), data['mask'].to(device)
                outputs = model(inputs)['segmentacao_narina']
                loss = criterions['segmentacao_narina'](outputs, masks)
                val_loss_seg += loss.item()

        # --- CÁLCULO DE MÉTRICAS E LÓGICA DE PARAGEM ---
        avg_val_loss_cls = val_loss_cls / len(dataloaders['classificacao_narina']['val'])
        acuracia_val = correct_cls / total_cls
        avg_val_loss_seg = val_loss_seg / len(dataloaders['segmentacao_narina']['val'])
        
        print(f"\n>> RESULTADOS DA ÉPOCA {epoch+1}:")
        print(f"   Classificação -> Acurácia: {(100*acuracia_val):.2f}% | Perda: {avg_val_loss_cls:.4f}")
        print(f"   Segmentação   -> Perda: {avg_val_loss_seg:.4f}")
        
        # Métrica combinada para early stopping (ex: acurácia - perda_seg)
        metrica_atual = acuracia_val - avg_val_loss_seg

        if metrica_atual > melhor_metrica_val:
            print(">>> Métrica de validação melhorou. A guardar o modelo...")
            melhor_metrica_val = metrica_atual
            torch.save(model.state_dict(), caminho_modelo_salvo)
            total_sem_melhora = 0
        else:
            total_sem_melhora += 1
            print(f">>> Métrica não melhorou. Paciência: {total_sem_melhora}/{paciencia}")

        if total_sem_melhora >= paciencia:
            print("\nPARAGEM ANTECIPADA!")
            break

    print(f"\nTreino simultâneo finalizado.")
    print(f"A carregar o melhor modelo guardado em '{caminho_modelo_salvo}'")
    model.load_state_dict(torch.load(caminho_modelo_salvo))
    return model

if __name__ == "__main__":
    # --- 1. CONFIGURAÇÕES GERAIS ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo selecionado: {device}")

    # --- 2. PREPARAÇÃO DOS DATASETS (igual ao seu código original) ---
    
    # Classificação
    transform_classificacao_treino = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_classificacao_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    caminho_classificacao = 'data/classificacao/Narina'
    dataset_temp_cls = datasets.ImageFolder(root=caminho_classificacao)
    targets_cls = dataset_temp_cls.targets
    indices_cls = list(range(len(dataset_temp_cls)))
    train_idx_cls, val_idx_cls = train_test_split(indices_cls, test_size=0.3, random_state=42, stratify=targets_cls)
    dataset_treino_cls = Subset(datasets.ImageFolder(root=caminho_classificacao, transform=transform_classificacao_treino), train_idx_cls)
    dataset_val_cls = Subset(datasets.ImageFolder(root=caminho_classificacao, transform=transform_classificacao_val), val_idx_cls)
    train_dataloader_cls = DataLoader(dataset_treino_cls, batch_size=16, shuffle=True, num_workers=0)
    val_dataloader_cls = DataLoader(dataset_val_cls, batch_size=16, shuffle=False, num_workers=0)
    print(f"Dataset de Classificação: {len(dataset_temp_cls)} imagens ({len(dataset_treino_cls)} treino, {len(dataset_val_cls)} val)")

    # Segmentação
    transform_segmentacao = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    caminho_segmentacao_img = 'data/segmentacao/Narina/imagens'
    caminho_segmentacao_mask = 'data/segmentacao/Narina/anotacoes'
    dataset_completo_seg = SegmentationDataset(image_dir=caminho_segmentacao_img, mask_dir=caminho_segmentacao_mask, transform=transform_segmentacao)
    indices_seg = list(range(len(dataset_completo_seg)))
    train_idx_seg, val_idx_seg = train_test_split(indices_seg, test_size=0.3, random_state=42)
    dataset_treino_seg = Subset(dataset_completo_seg, train_idx_seg)
    dataset_val_seg = Subset(dataset_completo_seg, val_idx_seg)
    train_dataloader_seg = DataLoader(dataset_treino_seg, batch_size=8, shuffle=True, num_workers=0)
    val_dataloader_seg = DataLoader(dataset_val_seg, batch_size=8, shuffle=False, num_workers=0)
    print(f"Dataset de Segmentação: {len(dataset_completo_seg)} imagens ({len(dataset_treino_seg)} treino, {len(dataset_val_seg)} val)")

    # --- 3. CONSTRUÇÃO DO MODELO MULTI-TAREFA ---
    
    # Definir as cabeças de cada tarefa
    num_classes_cls = len(dataset_temp_cls.classes)
    num_ftrs_backbone = 512 # ResNet-18/34
    
    head_classificacao = nn.Sequential(
        nn.Linear(num_ftrs_backbone, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes_cls)
    )

    head_segmentacao = SegmentationHead(in_channels=num_ftrs_backbone, out_channels=1)

    task_heads = {
        'classificacao_narina': head_classificacao,
        'segmentacao_narina': head_segmentacao
    }

    # Instanciar o modelo multi-tarefa
    multitask_model = MultiTaskModel(backbone_name='resnet18', task_heads=task_heads).to(device)

    # --- 4. CONFIGURAÇÃO DO TREINO ---
    
    # Agrupar dataloaders e funções de perda
    dataloaders = {
        'classificacao_narina': {'train': train_dataloader_cls, 'val': val_dataloader_cls},
        'segmentacao_narina': {'train': train_dataloader_seg, 'val': val_dataloader_seg}
    }
    criterions = {
        'classificacao_narina': nn.CrossEntropyLoss(),
        'segmentacao_narina': nn.BCEWithLogitsLoss()
    }
    
    # Otimizador único para todo o modelo
    optimizer = optim.Adam(multitask_model.parameters(), lr=0.0001)

    # --- 5. INICIAR O TREINO ---
    final_model = train_multitask_simultaneously(
        device,
        multitask_model,
        dataloaders,
        criterions,
        optimizer,
        num_epochs=50,
        paciencia=10
    )
    
    print("\nModelo Multi-Tarefa treinado com sucesso!")