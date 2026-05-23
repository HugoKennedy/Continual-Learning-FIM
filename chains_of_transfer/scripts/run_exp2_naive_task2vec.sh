#!/bin/bash

# Expérience 2 : Curriculum Learning (Stratégie Naïve + Ordre Task2Vec)

export CUDA_VISIBLE_DEVICES=0
model="simplecnn"

# Le fichier contenant l'ordre intelligent (Task2Vec)
task_order_file="./task_order/results/20260127_microsoft_resnet50_SplitCIFAR10_5times2classTasks_task2vec_orderings_task_order.json"

echo "=== Lancement Expérience 2 : NAIVE + TASK2VEC ==="
python -m chains_of_transfer.main \
    --benchmark SplitCIFAR10 \
    --metric_type embedding \
    --metric task2vec \
    --ordering_heuristic hamiltonian \
    --task_orders_file $task_order_file \
    --model $model \
    --strategy naive \
    --forWhat chains_of_transfer \
    --num_tasks 5 \
    --lr 0.001 \
    --epochs 15 \
    --batch_size 64