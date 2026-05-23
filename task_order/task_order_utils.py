import json


def get_task_order(
        metric_type="embedding",
        metric="task2vec",
        ordering_heuristic="hamiltonian",
        task_orders_file=None
    ):
    if task_orders_file is not None:
        with open(task_orders_file, "r") as f:
            data = json.load(f)
            for entry in data:

                # new format
                #[
                #  {
                #    "metric_type": "embedding",
                #    "metric": "task2vec",
                #    "ordering_heuristic": "centrality",
                #    "order": [2, 1, 3, 4, 0],
                #    "scores": [
                #      0.3035046291351319,
                #      0.3028356218338012,
                #      0.3025400114059448,
                #      0.3029856109619141,
                #      0.30307522773742684
                #    ]
                if entry.get("metric_type") == metric_type and \
                    entry.get("metric") == metric and \
                        entry.get("ordering_heuristic") == ordering_heuristic:
                        return entry["order"]


if __name__ == "__main__":
    # Example usage
    task_orders_file = "task_order/results/20260127_microsoft_resnet50_SplitMNIST_5times2classTasks_task2vec_orderings_task_order.json"
    order = get_task_order(
        metric_type="embedding",
        metric="task2vec",
        ordering_heuristic="centrality",
        task_orders_file=task_orders_file
    )
    print("Retrieved task order:", order)

    # Example usage
    task_orders_file = "task_order/results/20260125_ModelAgnostic_SplitMNIST_5times2classTasks_random_orderings_task_order.json"
    order = get_task_order(
        metric_type="random",
        metric="random",
        ordering_heuristic="random1",
        task_orders_file=task_orders_file
    )
    print("Retrieved task order:", order)