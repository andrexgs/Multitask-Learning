from train_task import train_task


def mtl_training(device, tasks_config):
    """
    Gerenciador de aprendizado multi-tarefas.

    Parâmetros:
    - device: Dispositivo (CPU ou GPU) a ser usado.
    - tasks_config: Lista de configurações de tarefas. Cada item é um dicionário:
        {
            'name': Nome da tarefa,
            'data': Dados (Dataset ou DataLoader),
            'params': Parâmetros específicos (ex.: arquitetura, otimizador)
        }

    Retorna:
    - models: Dicionário com os modelos treinados para cada tarefa.
    - results: Dicionário com os resultados de cada tarefa.
    """
    models = {}
    results = {}

    for task in tasks_config:
        task_name = task['name']
        task_data = task['data']
        task_params = task['params']

        print(f"Iniciando o treinamento para a tarefa: {task_name}")
        model, result = train_task(device, task_name, task_data, task_params)

        models[task_name] = model
        results[task_name] = result

    return models, results
