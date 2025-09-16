import torch
from mtl_manager import mtl_training
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from dataset import SegmentationDataset

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo selecionado: {device}")

    # --- Configurações de Transformação (sem alterações) ---
    transform_classificacao = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_validacao_cls = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # --- Carregamento de Dados de Classificação (sem alterações) ---
    caminho_classificacao = 'data/classificacao/Narina'
    dataset_temp_cls = datasets.ImageFolder(root=caminho_classificacao)
    targets_cls = dataset_temp_cls.targets
    indices_cls = list(range(len(dataset_temp_cls)))
    train_idx_cls, val_idx_cls = train_test_split(indices_cls, test_size=0.3, random_state=42, stratify=targets_cls)
    dataset_treino_cls = Subset(datasets.ImageFolder(root=caminho_classificacao, transform=transform_classificacao), train_idx_cls)
    dataset_val_cls = Subset(datasets.ImageFolder(root=caminho_classificacao, transform=transform_validacao_cls), val_idx_cls)
    train_dataloader_cls = DataLoader(dataset_treino_cls, batch_size=16, shuffle=True, num_workers=0)
    val_dataloader_cls = DataLoader(dataset_val_cls, batch_size=16, shuffle=False, num_workers=0)
    print(f"Dataset de Classificação: {len(dataset_temp_cls)} imagens ({len(dataset_treino_cls)} treino, {len(dataset_val_cls)} val)")

    # --- Carregamento de Dados de Segmentação (sem alterações) ---
    # ... (o código de segmentação continua igual)
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


    # --- CONFIGURAÇÃO DAS TAREFAS ---
    tasks_config = [
        {
            "name": "classificacao_narina",
            "data": {'train': train_dataloader_cls, 'val': val_dataloader_cls},
            "params": {
                "task_type": "classification",
                "architecture": "resnet18",
                "optimizer": "adam",
                "num_classes": len(dataset_temp_cls.classes),
                "batch_size": 32,
                "num_epochs": 50,
                "paciencia": 10, 
                "use_transfer_learning": True,
                
                "classifier_head": {
                    "layer1_neurons": 2048, # 1ª Camada
                    "layer2_neurons": 1024,  # 2ª Camada
                    "dropout": 0.5          # % de neurónios a "desligar" para evitar overfitting
                }
            }
        },
        # TAREFA SEGMENTAÇÃO
        {
            "name": "segmentacao_narina",
            "data": {'train': train_dataloader_seg, 'val': val_dataloader_seg},
            "params": {
                "task_type": "segmentation",
                "architecture": "resnet18",
                "optimizer": "adam",
                "num_classes": 1,
                "batch_size": 8,
                "num_epochs": 100,
                "paciencia": 10,
                "use_transfer_learning": True
            }
        },
    ]

    models, results = mtl_training(device, tasks_config)

    print("\nResultados Finais do Treinamento:")
    for task_name, result in results.items():
        print(f"Tarefa {task_name}: {result}")