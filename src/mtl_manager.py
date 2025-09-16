from train_task import train_task

def mtl_training(device, tasks_config):
    """
    Gestor de aprendizagem multi-tarefas.
    Esta função recebe uma lista de configurações de tarefas e executa o treino
    para cada uma delas de forma sequencial.

    Parâmetros:
    - device: O dispositivo (CPU ou GPU) a ser usado.
    - tasks_config: Uma lista de dicionários, onde cada dicionário define uma tarefa.

    Retorna:
    - models (dict): Um dicionário com os modelos treinados para cada tarefa.
    - results (dict): Um dicionário com os resultados de cada tarefa.
    """
    models = {}
    results = {}

    # Itera sobre cada configuração de tarefa fornecida no ficheiro train.py.
    for task in tasks_config:
        task_name = task['name']
        task_data = task['data']
        task_params = task['params']

        print(f"\n{'='*20}\nIniciando treino para a tarefa: {task_name}\n{'='*20}")
        
        # Chama a função principal de treino para a tarefa atual.
        model, result = train_task(device, task_name, task_data, task_params)

        # Armazena o modelo treinado e os resultados.
        models[task_name] = model
        results[task_name] = result

    return models, results