# MTL para Classificação e Segmentação de Narinas de Buldogue

Este projeto é uma estrutura modular e robusta para experimentação com Aprendizagem Multi-Tarefas (Multi-Task Learning - MTL) em Visão Computacional, utilizando Python e PyTorch. O objetivo principal é treinar dois modelos distintos para analisar imagens de narinas de buldogues:

1.  **Classificação:** Determinar o grau de estenose (ex: `mild`, `severe`).
2.  **Segmentação:** Identificar a área exata da narina na imagem.

## Estado do Projeto
-   [x] **Tarefa de Classificação:** Implementação completa e otimizada.
-   [x] **Tarefa de Segmentação:** Implementação completa e otimizada.

## Principais Funcionalidades

-   **Estrutura Modular:** O código é organizado de forma lógica, facilitando a manutenção e a adição de novas tarefas.
-   **Carregamento de Dados Simplificado:** Utiliza o `ImageFolder` do PyTorch para carregar datasets de classificação diretamente da estrutura de pastas, eliminando a necessidade de ficheiros de anotação JSON para esta tarefa.
-   **Treino e Validação Robustos:** Inclui uma divisão automática dos dados em conjuntos de treino e validação para uma avaliação fiável do desempenho do modelo.
-   **Paragem Antecipada (Early Stopping):** O treino para automaticamente se o modelo deixar de melhorar na validação, poupando tempo e prevenindo o sobreajuste (*overfitting*).
-   **Otimização Avançada:** Utiliza *Data Augmentation* para melhorar a generalização e um Agendador de Taxa de Aprendizagem (*Learning Rate Scheduler*) para um ajuste mais fino do modelo.
-   **Configuração Flexível:** Todos os hiperparâmetros importantes (taxa de aprendizagem, número de neurónios, arquitetura do modelo) podem ser facilmente ajustados no ficheiro `src/train.py`.

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

Crie e ative um ambiente virtual:
```bash
# Criar o ambiente
python3 -m venv multitask

# Ativar o ambiente (Linux/macOS)
source multitask/bin/activate
```

Crie um ficheiro `requirements.txt` na raiz do projeto com o seguinte conteúdo:
```
torch
torchvision
scikit-learn
pillow
opencv-python
```

Depois, instale todas as dependências com um único comando:
```bash
pip install -r requirements.txt
```

### 3. Preparação dos Dados

A estrutura de pastas é crucial para o funcionamento automático do projeto.

**Para a Classificação:**
Crie uma subpasta para cada classe dentro de `data/classificacao/Narina/`.
```
data/
└── classificacao/
    └── Narina/
        ├── MILD/
        │   ├── imagem1.jpg
        │   └── ...
        └── SEVERE/
            ├── imagem2.jpg
            └── ...
```

**Para a Segmentação:**
Coloque as imagens originais em `imagens/` e as suas respetivas máscaras (a preto e branco) em `anotacoes/`.
```
data/
└── segmentacao/
    └── Narina/
        ├── imagens/
        │   ├── img1.jpg
        │   └── ...
        └── anotacoes/
            ├── img1.png  <-- Máscara correspondente
            └── ...
```

### 4. Treino dos Modelos

Com o ambiente ativado e os dados no sítio certo, inicie o treino com um simples comando:
```bash
python3 src/train.py
```
O script irá treinar sequencialmente cada tarefa configurada na lista `tasks_config` dentro do próprio script. O melhor modelo para cada tarefa será guardado na raiz do projeto (ex: `melhor_modelo_classificacao_narina.pth`).

## Configuração e Hiperparâmetros

O ficheiro `src/train.py` é o seu centro de controlo. Dentro da lista `tasks_config`, pode facilmente ajustar os seguintes parâmetros para cada tarefa:
-   `"architecture"`: Mude de `"resnet18"` para `"resnet34"` ou `"resnet50"` para usar um modelo mais potente.
-   `"learning_rate"`: Ajuste a taxa de aprendizagem para otimizar a convergência do modelo.
-   `"paciencia"`: Defina o número de épocas a esperar sem melhoria antes de parar o treino.
-   `"classifier_head"`: Para tarefas de classificação, pode definir uma cabeça de rede neural personalizada com o número de neurónios que desejar.

## Ferramentas de Pré-processamento (`utils/`)

O projeto inclui scripts na pasta `utils/` para converter outros formatos de dataset para o padrão COCO JSON. Embora **não sejam necessários para o fluxo de trabalho atual**, são ferramentas úteis.
-   `normaliza_dataset_classificacao.py`: Converte um dataset de classificação em pastas para o formato COCO JSON.
-   `normaliza_dataset_segmentacao.py`: Converte um dataset de segmentação (imagens e máscaras) para o formato COCO JSON.

## Próximos Passos
-   [ ] Implementar o `MultiTaskModel` de `src/model.py` para que as tarefas partilhem um *backbone* comum e treinem de forma verdadeiramente simultânea.