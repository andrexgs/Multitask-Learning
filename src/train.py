import torch
from mtl_manager import mtl_training
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from dataset import SegmentationDataset

if __name__ == "__main__":
    # --- 1. CONFIGURAÇÕES GERAIS ---
    # Seleciona automaticamente a GPU se estiver disponível, caso contrário, usa a CPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo selecionado: {device}")

    # --- 2. PREPARAÇÃO DA TAREFA DE CLASSIFICAÇÃO ---

    # Define as transformações de Data Augmentation para o conjunto de treino.
    # Isto ajuda a prevenir o overfitting e a melhorar a generalização do modelo.
    transform_classificacao_treino = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Define transformações mais simples para o conjunto de validação (sem augmentation).
    transform_classificacao_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Carrega o dataset de classificação usando ImageFolder e divide-o em treino e validação.
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

    # --- 3. PREPARAÇÃO DA TAREFA DE SEGMENTAÇÃO ---

    # Para a segmentação, usamos transformações mais simples.
    transform_segmentacao = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Carrega o dataset de segmentação usando a nossa classe personalizada e divide-o.
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


    # --- 4. CONFIGURAÇÃO DAS TAREFAS PARA O TREINO ---
    # Esta lista controla quais tarefas serão treinadas. Pode comentar uma tarefa para a desativar.
    tasks_config = [
        {
            "name": "classificacao_narina",
            "data": {'train': train_dataloader_cls, 'val': val_dataloader_cls},
            "params": {
                # --- Hiperparâmetros da Classificação ---
                "task_type": "classification",
                "architecture": "resnet18",
                "optimizer": "adam",
                "learning_rate": 0.0001,
                "num_classes": len(dataset_temp_cls.classes),
                "num_epochs": 50,
                "paciencia": 10,
                "use_transfer_learning": True,
                
                # Define a cabeça de classificação personalizada com 3 camadas.
                "classifier_head": {
                    "layer1_neurons": 1024,
                    "layer2_neurons": 512,
                    "dropout": 0.5
                }
            }
        },
        {
            "name": "segmentacao_narina",
            "data": {'train': train_dataloader_seg, 'val': val_dataloader_seg},
            "params": {
                # --- Hiperparâmetros da Segmentação ---
                "task_type": "segmentation",
                "architecture": "resnet18",
                "optimizer": "adam",
                "learning_rate": 0.001,
                "num_classes": 1, # A saída é uma única máscara binária (1 canal).
                "num_epochs": 100,
                "paciencia": 10,
                "use_transfer_learning": True
            }
        },
    ]

    # Inicia o gestor de treino multi-tarefa.
    models, results = mtl_training(device, tasks_config)

    # Imprime os resultados finais de cada tarefa.
    print("\nResultados Finais do Treinamento:")
    for task_name, result in results.items():
        print(f"Tarefa {task_name}: {result}")