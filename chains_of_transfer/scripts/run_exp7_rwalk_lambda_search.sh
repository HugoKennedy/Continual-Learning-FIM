#!/bin/bash

# Expérience 7 : Recherche du Lambda optimal pour RWalk (Synaptic Intelligence)

export CUDA_VISIBLE_DEVICES=0
model="simplecnn"
task_order_file="./task_order/results/20260127_microsoft_resnet50_SplitCIFAR10_5times2classTasks_task2vec_orderings_task_order.json"

# L'échelle logarithmique adaptée pour RWalk (plus faible que EWC)
declare -a lambdas=(0.001 0.005 0.01 0.05 0.1)

LOG_FILE="temp_log_exp7.txt"
SUMMARY_FILE="summary_exp7_rwalk.txt"

echo "RECHERCHE EN GRILLE : LAMBDA OPTIMAL POUR RWALK" > $SUMMARY_FILE


echo "Expérience 7 (Grid Search RWalk)"

for l in "${lambdas[@]}"
do
    echo ""

    echo "RWalk avec Lambda = $l"

    
    python -m chains_of_transfer.main \
        --benchmark SplitCIFAR10 \
        --metric_type embedding \
        --metric task2vec \
        --ordering_heuristic hamiltonian \
        --task_orders_file $task_order_file \
        --model $model \
        --strategy rwalk \
        --ewc_lambda $l \
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
    
    echo "Lambda $l -> Acc: $ACC | Forgetting: $FORG | Mag: $MAG | Dead: $DEAD%" >> $SUMMARY_FILE
    
    echo "Terminé pour Lambda $l"
done

# --- AFFICHAGE DU RÉSUMÉ FINAL ---
echo ""
echo "RECHERCHE EN GRILLE TERMINÉE !"

echo ""
cat $SUMMARY_FILE

# Nettoyage
rm $LOG_FILE