# Aprendizagem Multi-Tarefas com PyTorch

Este projeto é uma estrutura modular e robusta para experimentação com Aprendizagem Multi-Tarefas (Multi-Task Learning - MTL) em Visão Computacional, utilizando Python e PyTorch. O objetivo é fornecer uma base sólida e de fácil expansão para treinar múltiplos modelos simultaneamente, começando com um classificador de imagens e com planos para incluir outras tarefas como segmentação semântica.

## Estado do Projeto
-   [x] **Tarefa de Classificação:** Implementação completa com um classificador de imagens robusto.
-   [ ] **Tarefa de Segmentação:** Próximo passo de desenvolvimento.

## Principais Funcionalidades

-   **Estrutura Modular:** O código é organizado de forma lógica, separando o carregamento de dados, a definição da arquitetura (`arch_optim.py`) e a lógica de treino (`train_task.py`), facilitando a manutenção e expansão.
-   **Treino e Validação Robustos:** Inclui uma divisão automática dos dados em conjuntos de treino e validação, garantindo uma avaliação fiável do desempenho do modelo.
-   **Paragem Antecipada (Early Stopping):** O treino monitoriza a acurácia na validação e para automaticamente se o modelo deixar de melhorar, poupando tempo e prevenindo o sobreajuste (*overfitting*).
-   **Data Augmentation:** Aplica transformações aleatórias às imagens de treino (rotações, inversões) para aumentar a variedade dos dados e melhorar a capacidade de generalização do modelo.
-   **Flexibilidade:** Adicionar uma nova tarefa é simplificado, bastando configurar os dados e os parâmetros no ficheiro principal `train.py`.

## Como Começar

Siga estes passos para configurar e executar o projeto no seu ambiente local.

### 1. Pré-requisitos
-   Python 3.7+
-   Git

### 2. Instalação

Primeiro, clone o repositório para a sua máquina:
```bash
git clone <URL_DO_SEU_REPOSITÓRIO>
cd multitask-learning
```

Crie e ative um ambiente virtual para isolar as dependências do projeto:
```bash
# Criar o ambiente
python3 -m venv multitask

# Ativar o ambiente (Linux/macOS)
source multitask/bin/activate
```

Agora, instale todas as bibliotecas necessárias. Pode criar um ficheiro `requirements.txt` para facilitar:
```bash
# Crie este ficheiro manualmente ou com 'pip freeze > requirements.txt'
# Conteúdo do requirements.txt:
# torch
# torchvision
# scikit-learn
# pillow

pip install -r requirements.txt
```

### 3. Preparação dos Dados

Para a tarefa de **classificação**, o projeto utiliza a estrutura de pastas padrão do `ImageFolder`. Crie uma pasta `data/classificacao` e, dentro dela, uma subpasta para cada classe, contendo as respetivas imagens:

```
multitask-learning/
└── data/
    └── classificacao/
        ├── classe_A/
        │   ├── imagem1.jpg
        │   └── imagem2.jpg
        ├── classe_B/
        │   ├── imagem3.jpg
        │   └── imagem4.jpg
        └── ...
```
O script irá detetar os nomes das pastas como os nomes das classes automaticamente.

### 4. Treino do Modelo

Com o ambiente ativado e os dados no sítio certo, inicie o treino com um simples comando:
```bash
python3 src/train.py
```
O script irá automaticamente dividir os dados (70% para treino, 30% para validação), iniciar o treino e exibir o progresso. O melhor modelo será guardado na raiz do projeto com o nome `melhor_modelo_<nome_da_tarefa>.pth`.

## Estrutura do Código (`src/`)

-   `train.py`: O ponto de entrada principal. Aqui você configura as tarefas, prepara os `DataLoaders` e inicia o processo de treino.
-   `mtl_manager.py`: O gestor que recebe as configurações de todas as tarefas e chama a função de treino para cada uma delas.
-   `train_task.py`: Contém a lógica de treino detalhada para uma única tarefa, incluindo o loop por épocas, as fases de treino e validação, e a paragem antecipada.
-   `arch_optim.py`: Define as funções para obter a arquitetura do modelo (ex: `resnet18`) e o otimizador (ex: `Adam`).
-   `dataset.py`: Contém classes `Dataset` personalizadas para carregar diferentes tipos de dados.
-   `model.py`: Definição do modelo multi-tarefa, com um *backbone* partilhado e "cabeças" específicas para cada tarefa.

## Próximos Passos
-   [ ] Adicionar uma tarefa de **segmentação semântica** à estrutura.
-   [ ] Implementar o modelo `MultiTaskModel` para que as tarefas partilhem um *backbone* comum, treinando de forma verdadeiramente simultânea.
