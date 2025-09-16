import torch
from mtl_manager import mtl_training
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo selecionado: {device}")

    transform_treino = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_validacao = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    caminho_dataset = 'data/classificacao'
    
    dataset_temp = datasets.ImageFolder(root=caminho_dataset)
    targets = dataset_temp.targets
    
    indices = list(range(len(dataset_temp)))
    train_idx, val_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=targets)

    dataset_treino = datasets.ImageFolder(root=caminho_dataset, transform=transform_treino)
    dataset_validacao = datasets.ImageFolder(root=caminho_dataset, transform=transform_validacao)

    training_data = Subset(dataset_treino, train_idx)
    val_data = Subset(dataset_validacao, val_idx)
    
    # Usamos num_workers=0 para máxima compatibilidade
    train_dataloader = DataLoader(training_data, batch_size=16, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=0)
    
    print(f"Total de imagens: {len(dataset_temp)}")
    print(f"Imagens de treino: {len(training_data)}")
    print(f"Imagens de validação: {len(val_data)}")
    print(f"Classes encontradas: {dataset_temp.classes}")

    tasks_config = [
        {
            "name": "minha_classificacao",
            "data": {'train': train_dataloader, 'val': val_dataloader},
            "params": {
                "architecture": "resnet18",
                "optimizer": "adam",
                "in_channels": 3,
                "num_classes": len(dataset_temp.classes),
                "batch_size": 32,
                "num_epochs": 100,
                "paciencia": 7,
                "use_transfer_learning": True
            }
        },
    ]

    models, results = mtl_training(device, tasks_config)

    print("\nResultados Finais do Treinamento:")
    for task_name, result in results.items():
        print(f"Tarefa {task_name}: {result}")