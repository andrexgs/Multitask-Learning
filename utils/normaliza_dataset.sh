#!/bin/bash

# Pega três parâmetros: diretório do dataset (-d), tipo de tarefa (-t) e formato atual (-f)
while getopts "d:t:f:" opt; do
    case $opt in
        d) DATASET_DIR="$OPTARG" ;;
        t) TASK_TYPE="$OPTARG" ;;
        f) CURRENT_FORMAT="$OPTARG" ;;
        *) echo "Opção inválida: -$OPTARG" >&2; exit 1 ;;
    esac
done

# Informa os parâmetros necessários
if [ -z "$DATASET_DIR" ] || [ -z "$CURRENT_FORMAT" ]; then
    echo "Uso: $0 <diretório_do_dataset> <task> <formato_atual>"
    echo "Exemplo: $0 /caminho/para/dataset <classificacao> csv"
    echo "Tarefas suportadas: <classificacao>, <segmentacao>."
    echo "Formatação atual suportada para classificação: imagens separadas em pastas por classe (<pastas>)."
    echo "Formatação atual suportada para segmentação: imagens e máscaras em pastas separadas (anotacoes e imagens) (<imagem_mascara_binaria>); somente segmentação binária."
    exit 1
fi

# Separa o nome da pasta pai e do diretório do dataset e entra no diretório do dataset
PARENT_DIR=$(dirname "$DATASET_DIR")
DATASET_NAME=$(basename "$DATASET_DIR")
cd $PARENT_DIR || { echo "Erro ao entrar no diretório pai: $PARENT_DIR"; exit 1; }

# Cria um diretório temporário para o novo dataset
TEMP_DIR="${DATASET_DIR}_temp"
mkdir -p "$TEMP_DIR"

# Volta para a pasta utils e chama um script Python para normalizar o dataset
cd ../utils
if [ "$TASK_TYPE" == "classificacao" ]; then
    python3 normaliza_dataset_classificacao.py -d "$DATASET_DIR" -f "$CURRENT_FORMAT" -o "$TEMP_DIR"

    if [ "$CURRENT_FORMAT" == "pastas" ]; then
        # Copia as imagens para o diretório temporário
        cp $DATASET_DIR/*/* $TEMP_DIR
    fi

elif [ "$TASK_TYPE" == "segmentacao" ]; then
    python3 normaliza_dataset_segmentacao.py -d "$DATASET_DIR" -f "$CURRENT_FORMAT" -o "$TEMP_DIR" -i "imagens" -a "anotacoes"

    if [ "$CURRENT_FORMAT" == "imagem_mascara_binaria" ]; then
        # Copia as imagens para o diretório temporário
        cp $DATASET_DIR/imagens/* $TEMP_DIR
    fi

else
    echo "Tarefa inválida: $TASK_TYPE. Use 'classificacao' ou 'segmentacao'."
    exit 1
fi

# Remove o diretório original do dataset
rm -rf "$DATASET_DIR"
# Renomeia o diretório temporário para o nome original do dataset
mv "$TEMP_DIR" "$DATASET_DIR"