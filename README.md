# Aprendizagem Multi-Tarefas em Python

Este projeto demonstra uma arquitetura básica para aprendizagem multi-tarefas (Multi Task Learning - MTL) utilizando Python e PyTorch. A ideia é fornecer uma estrutura modular e facilmente extensível para a inclusão de múltiplas tarefas, cada uma podendo ter sua própria entrada, rótulos e métricas.

# Utils: Normalização de Dataset de Classificação para COCO

## normaliza_dataset_classificacao.py
Este script converte um dataset de classificação organizado em pastas  
(em que cada subpasta é uma classe) para o formato **COCO JSON**, gerando:

- **categories**: lista de classes extraída das subpastas.
- **images**: metadados de cada imagem (tamanho, caminho relativo, timestamp).
- **annotations**: uma caixa delimitadora que cobre toda a imagem, adequada para tarefas de classificação.

## Como usar:
python utils/normaliza_dataset_classificacao.py \
  -f pastas \
  -d data/classificacao \
  -o classificacao_coco.json

## Características

- **Modularidade**: Código organizado em pastas distintas para dados, modelo e treinamento.
- **Flexibilidade**: É simples adicionar novas tarefas, apenas adicionando pastas de dados e ajustando o dicionário `task_dirs`.
- **Backbone compartilhado**: Todas as tarefas compartilham uma camada inicial (backbone) da rede, enquanto as camadas finais (cabeças) são específicas para cada tarefa.
- **Exemplo prático**: O código exemplo utiliza um dataset hipotético de imagens (ex.: bulldogs para classificação de estenose nasal), mas pode ser facilmente adaptado para qualquer outro tipo de tarefa.

## Estrutura de Pastas
```
multi_task_learning/
├── data/
│   ├── task1/
│   │   ├── imagens/
│   │   │   ├── img1.jpg
│   │   │   ├── img2.jpg
│   │   │   └── ...
│   │   └── labels.csv
│   ├── task2/
│   │   ├── imagens/
│   │   │   ├── img1.jpg
│   │   │   ├── img2.jpg
│   │   │   └── ...
│   │   └── labels.csv
│   └── ... (adicione quantas tarefas precisar)
│
├── src/
│   ├── dataset.py       # Classes e métodos para carregar e processar dados
│   ├── model.py         # Definição do modelo multi-tarefa
│   ├── train_task.py    # Função encapsulada para treinamento de uma única tarefa
│   ├── mtl_manager.py   # Gerenciador do MTL que coordena múltiplas tarefas
│   ├── train.py         # Script principal de treinamento
|
├── utils/
|
└── README.md            # Documentação do projeto
```

## Requisitos

- Python 3.7+
- PyTorch
- torchvision
- pandas
- Pillow

Você pode instalar os requisitos com:

```bash
pip install torch torchvision pandas Pillow
