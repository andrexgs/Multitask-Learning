# Aprendizagem Multi-Tarefas com PyTorch

Este projeto é uma estrutura modular e robusta para experimentação com Aprendizagem Multi-Tarefas (Multi-Task Learning - MTL) em Visão Computacional, utilizando Python e PyTorch. O objetivo é fornecer uma base sólida e de fácil expansão para treinar múltiplos modelos simultaneamente, começando com um classificador de imagens e com planos para incluir outras tarefas como segmentação semântica.

## Estado do Projeto
-   [x] **Tarefa de Classificação:** Implementação completa com um classificador de imagens robusto.
-   [ ] **Tarefa de Segmentação:** Próximo passo de desenvolvimento.

## Principais Funcionalidades

-   **Estrutura Modular:** O código é organizado de forma lógica, separando o carregamento de dados, a definição da arquitetura (`arch_optim.py`) e a lógica de treino (`train_task.py`), facilitando a manutenção e expansão.
-   **Carregamento de Dados Simplificado:** Utiliza o `ImageFolder` do PyTorch para carregar datasets de classificação diretamente da estrutura de pastas, sem a necessidade de ficheiros de anotação adicionais.
-   **Treino e Validação Robustos:** Inclui uma divisão automática dos dados em conjuntos de treino e validação, garantindo uma avaliação fiável do desempenho do modelo.
-   **Paragem Antecipada (Early Stopping):** O treino monitoriza a acurácia na validação e para automaticamente se o modelo deixar de melhorar, poupando tempo e prevenindo o sobreajuste (*overfitting*).
-   **Data Augmentation:** Aplica transformações aleatórias às imagens de treino (rotações, inversões) para aumentar a variedade dos dados e melhorar a capacidade de generalização do modelo.

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
# Crie este ficheiro com 'pip freeze > requirements.txt'
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
O script `src/train.py` irá detetar os nomes das pastas como os nomes das classes automaticamente.

### 4. Treino do Modelo

Com o ambiente ativado e os dados no sítio certo, inicie o treino com um simples comando:
```bash
python3 src/train.py
```
O script irá automaticamente dividir os dados (70% para treino, 30% para validação), iniciar o treino e exibir o progresso. O melhor modelo será guardado na raiz do projeto com o nome `melhor_modelo_<nome_da_tarefa>.pth`.

## Ferramentas de Pré-processamento (`utils/`)

O projeto inclui scripts na pasta `utils/` para converter outros formatos de dataset para o padrão COCO JSON. Embora **não sejam necessários para o fluxo de trabalho de classificação atual**, são ferramentas úteis para outras tarefas ou tipos de dados.

### Normalização de Datasets de Classificação
O script `normaliza_dataset_classificacao.py` converte um dataset de classificação organizado em pastas para o formato COCO JSON.
**Como usar:**
```bash
python3 utils/normaliza_dataset_classificacao.py \
  -f pastas \
  -d data/classificacao \
  -o .
```

### Normalização de Datasets de Segmentação
O script `normaliza_dataset_segmentacao.py` converte um dataset de segmentação (imagens e máscaras) para o formato COCO JSON.
**Como usar:**
```bash
python3 utils/normaliza_dataset_segmentacao.py \
  -f imagem_mascara_binaria \
  -d data/segmentacao \
  -i imagens \
  -a anotacoes \
  -o .
```

## Próximos Passos
-   [ ] Adicionar uma tarefa de **segmentação semântica** à estrutura.
-   [ ] Implementar o modelo `MultiTaskModel` para que as tarefas partilhem um *backbone* comum, treinando de forma verdadeiramente simultânea.
