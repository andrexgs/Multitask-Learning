import torch
from mtl_manager import mtl_training
import data_manager

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Configuração das tarefas
    tasks_config = [
        {
            "name": "classificacao_nasal",
            "data": data_manager.get_dataset("data/task1"),  # Caminho para os dados da tarefa 1
            "params": {
                "architecture": "resnet18",
                "optimizer": "adam",
                "in_channels": 3,
                "num_classes": 2,
                "batch_size": 16,
                "num_epochs": 10,
                "use_transfer_learning": True
            }
        },
        {
            "name": "outra_tarefa",
            "data": data_manager.get_dataset("data/task2"),  # Caminho para os dados da tarefa 2
            "params": {
                "architecture": "mobilenet_v2",
                "optimizer": "sgd",
                "in_channels": 3,
                "num_classes": 5,
                "batch_size": 32,
                "num_epochs": 15,
                "use_transfer_learning": False
            }
        },
        #{ Aqui serão adicionadas quantas tarefas forem interessantes adicionar ao MTL
        #    ...
        #    ...
        #    ...
        #}

    ]

    # Treinamento MTL
    models, results = mtl_training(device, tasks_config)

    print("Resultados do MTL:")
    for task_name, result in results.items():
        print(f"Tarefa {task_name}: {result}")
