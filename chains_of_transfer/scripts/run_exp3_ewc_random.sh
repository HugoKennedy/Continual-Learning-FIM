#!/bin/bash

# Expérience 3 : Régularisation (Stratégie EWC + Ordre Aléatoire)

export CUDA_VISIBLE_DEVICES=0
model="simplecnn"

# Le fichier contenant l'ordre aléatoire
task_order_file="./task_order/results/20260125_ModelAgnostic_SplitCIFAR10_5times2classTasks_random_orderings_task_order.json"

echo "=== Lancement Expérience 3 : EWC + RANDOM ==="
python -m chains_of_transfer.main \
    --benchmark SplitCIFAR10 \
    --metric_type random \
    --metric random \
    --ordering_heuristic random1 \
    --task_orders_file $task_order_file \
    --model $model \
    --strategy ewc \
    --ewc_lambda 0.1 \
    --forWhat chains_of_transfer \
    --num_tasks 5 \
    --lr 0.001 \
    --epochs 15 \
    --batch_size 64