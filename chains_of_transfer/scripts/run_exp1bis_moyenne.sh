#!/bin/bash

# Expérience 1 (Multi-Runs) : Baseline moyennée sur 5 ordres aléatoires

export CUDA_VISIBLE_DEVICES=0
model="simplecnn"
task_order_file="./task_order/results/20260125_ModelAgnostic_SplitCIFAR10_5times2classTasks_random_orderings_task_order.json"

# On va boucler sur 5 ordres aléatoires différents
declare -a heuristics=("random1" "random2" "random3" "random4" "random5")

LOG_FILE="temp_log_exp1_multi.txt"
SUMMARY_FILE="summary_exp1_multi.txt"

echo "RÉSULTATS MULTIPLES (NAIVE + ORDRES ALEATOIRES)" > $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

echo "Lancement Expérience 1 sur plusieurs ordres"

for h in "${heuristics[@]}"
do
    echo ""

    echo "Stratégie Naïve avec Ordre = $h"

    
    python -m chains_of_transfer.main \
        --benchmark SplitCIFAR10 \
        --metric_type random \
        --metric random \
        --ordering_heuristic $h \
        --task_orders_file $task_order_file \
        --model $model \
        --strategy naive \
        --forWhat chains_of_transfer \
        --num_tasks 5 \
        --lr 0.001 \
        --epochs 15 \
        --batch_size 64 2>&1 | tee $LOG_FILE
        
    # --- EXTRACTION AUTOMATIQUE DES RÉSULTATS ---
    ACC=$(grep "Top1_Acc_Stream/eval_phase/test_stream/Task004" $LOG_FILE | tail -1 | awk -F' = ' '{print $2}')
    FORG=$(grep "StreamForgetting/eval_phase/test_stream" $LOG_FILE | tail -1 | awk -F' = ' '{print $2}')
    MAG=$(grep "Weight Magnitude" $LOG_FILE | tail -1 | awk -F' : ' '{print $2}')
    DEAD=$(grep "Dead Units Ratio" $LOG_FILE | tail -1 | awk -F' : ' '{print $2}' | tr -d '%')
    
    echo "Ordre $h -> Acc: $ACC | Forgetting: $FORG | Mag: $MAG | Dead: $DEAD%" >> $SUMMARY_FILE
    
    echo "Terminé pour $h"
done

# --- AFFICHAGE DU RÉSUMÉ FINAL ---
echo ""
echo "TOUTES LES EXPÉRIENCES SONT TERMINÉES !"
echo ""
cat $SUMMARY_FILE

# Nettoyage
rm $LOG_FILE
