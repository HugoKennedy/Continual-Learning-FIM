#!/bin/bash

# Expérience 5 : Grid Search pour trouver le lambda optimal (EWC + Task2Vec)

export CUDA_VISIBLE_DEVICES=0
model="simplecnn"
task_order_file="./task_order/results/20260127_microsoft_resnet50_SplitCIFAR10_5times2classTasks_task2vec_orderings_task_order.json"

# Les lambdas à tester
declare -a lambdas=(0.005 0.01 0.05 0.1 0.5)

# Fichiers temporaires pour extraire les données
LOG_FILE="temp_log_exp5.txt"
SUMMARY_FILE="summary_exp5.txt"

echo "RÉSULTATS DU GRID SEARCH (Lambda)" > $SUMMARY_FILE


echo "Expérience 5 : RECHERCHE DU LAMBDA OPTIMAL"

for l in "${lambdas[@]}"
do
    echo ""

    echo "EWC avec Lambda = $l"

    
    # On lance Python, on affiche à l'écran ET on sauvegarde dans temp_log_exp5.txt
    python -m chains_of_transfer.main \
        --benchmark SplitCIFAR10 \
        --metric_type embedding \
        --metric task2vec \
        --ordering_heuristic hamiltonian \
        --task_orders_file $task_order_file \
        --model $model \
        --strategy ewc \
        --ewc_lambda $l \
        --forWhat chains_of_transfer \
        --num_tasks 5 \
        --lr 0.001 \
        --epochs 15 \
        --batch_size 64 2>&1 | tee $LOG_FILE
        
    # --- EXTRACTION AUTOMATIQUE DES RÉSULTATS ---
    # On cherche la dernière ligne contenant l'Accuracy et le Forgetting
    ACC=$(grep "Top1_Acc_Stream/eval_phase/test_stream/Task004" $LOG_FILE | tail -1 | awk -F' = ' '{print $2}')
    FORG=$(grep "StreamForgetting/eval_phase/test_stream" $LOG_FILE | tail -1 | awk -F' = ' '{print $2}')
    
    # On sauvegarde ça dans notre fichier de résumé
    echo "Lambda $l -> Acc: $ACC | Forgetting: $FORG" >> $SUMMARY_FILE
    
    echo "Terminé pour Lambda = $l"
done

# --- AFFICHAGE DU RÉSUMÉ FINAL ---
echo ""
echo "TOUTES LES EXPÉRIENCES SONT TERMINÉES !"

echo ""
cat $SUMMARY_FILE

# On nettoie le fichier log temporaire
rm $LOG_FILE