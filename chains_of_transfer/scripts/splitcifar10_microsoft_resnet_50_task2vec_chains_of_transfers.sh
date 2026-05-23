# chains of transfers
# SplitCIFAR10
# 5 two-class classification tasks

export CUDA_VISIBLE_DEVICES=0

## model used in the chain of transfer (not to be confused with the model used to compute transferability and, ultimately, the ordering).
## BTW, the title of this script, i.e., *_microsoft_resnet50_*, corresponds to the model used to compute transferability and the orderings.
model="microsoft_resnet50"
#model="simplecnn"

declare -a arr1=(
    "task2vec"
)
declare -a arr2=(
    #"centrality"
    #"cluster"
    #"eccentricity"
    "hamiltonian"
    #"spectral"
    #"typicality"
)
task_order_file="./task_order/results/20260127_microsoft_resnet50_SplitCIFAR10_5times2classTasks_task2vec_orderings_task_order.json"

for i in "${arr1[@]}"
do
    for j in "${arr2[@]}"
    do
        echo "$i - $j"
        python -m chains_of_transfer.main \
            --benchmark SplitCIFAR10 \
            --metric_type embedding \
            --metric $i \
            --ordering_heuristic $j \
            --task_orders_file $task_order_file \
            --model $model \
            --forWhat chains_of_transfer \
            --num_tasks 5 \
            --k_shot 20 \
            --lr 0.0001 \
            --epochs 15 \
            --batch_size 10
    done
done
