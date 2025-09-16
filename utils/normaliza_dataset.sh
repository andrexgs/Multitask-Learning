#!/bin/bash

# Script de conveniência para executar os normalizadores de dataset.
# Este script foi mantido para referência, mas o fluxo de treino principal
# agora carrega os dados diretamente das pastas, não sendo necessário
# executar este script para treinar os modelos.

# --- PARÂMETROS ---
# -d : Diretório do dataset
# -t : Tipo de tarefa ('classificacao' ou 'segmentacao')
# -f : Formato atual do dataset ('pastas' ou 'imagem_mascara_binaria')

while getopts "d:t:f:" opt; do
    case $opt in
        d) DATASET_DIR="$OPTARG" ;;
        t) TASK_TYPE="$OPTARG" ;;
        f) CURRENT_FORMAT="$OPTARG" ;;
        *) echo "Opção inválida: -$OPTARG" >&2; exit 1 ;;
    esac
done

# Verifica se os parâmetros necessários foram fornecidos
if [ -z "$DATASET_DIR" ] || [ -z "$TASK_TYPE" ] || [ -z "$CURRENT_FORMAT" ]; then
    echo "Uso: $0 -d <diretório_do_dataset> -t <tarefa> -f <formato_atual>"
    echo "Exemplo (Classificação): $0 -d data/classificacao/Narina -t classificacao -f pastas"
    echo "Exemplo (Segmentação):   $0 -d data/segmentacao/Narina -t segmentacao -f imagem_mascara_binaria"
    exit 1
fi

echo "A processar o dataset em: $DATASET_DIR"

# Chama o script Python apropriado com base no tipo de tarefa
if [ "$TASK_TYPE" == "classificacao" ]; then
    python3 normaliza_dataset_classificacao.py -d "$DATASET_DIR" -f "$CURRENT_FORMAT" -o .

elif [ "$TASK_TYPE" == "segmentacao" ]; then
    python3 normaliza_dataset_segmentacao.py -d "$DATASET_DIR" -f "$CURRENT_FORMAT" -o . -i "imagens" -a "anotacoes"

else
    echo "Tarefa inválida: $TASK_TYPE. Use 'classificacao' ou 'segmentacao'."
    exit 1
fi

echo "Processo concluído."