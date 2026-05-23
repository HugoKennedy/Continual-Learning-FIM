import argparse
import os
import json

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
import numpy as np
import matplotlib.pyplot as plt # <-- NOUVEL IMPORT

# Avalanche
from avalanche.training.supervised import Naive, EWC
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics, confusion_matrix_metrics
from avalanche.training.plugins import EvaluationPlugin, SupervisedPlugin
from avalanche.logging import InteractiveLogger, TextLogger, CSVLogger

# custom imports
from benchmarks.benchmarks_utils import get_benchmark
from nets.nets_utils import get_net
from task_order.task_order_utils import get_task_order
from utils import get_random_string


# Rwalk/SI Plugin personnalisé avec clamping pour éviter les NaN

class CustomSIPlugin(SupervisedPlugin):
    """
    Implémentation manuelle de Synaptic Intelligence (RWalk simplifié).
    Intègre un 'clamping' pour éviter l'explosion des gradients (NaN) 
    qui tue les neurones sur des architectures non normalisées.
    """
    def __init__(self, si_lambda=0.1):
        super().__init__()
        self.si_lambda = si_lambda
        self.saved_params = {}
        self.omega = {}  
        self.delta_params = {} 
        self.task_count = 0

    def before_training_exp(self, strategy, **kwargs):
        for name, param in strategy.model.named_parameters():
            if param.requires_grad:
                self.saved_params[name] = param.clone().detach()
                if name not in self.omega:
                    self.omega[name] = torch.zeros_like(param)
                if name not in self.delta_params:
                    self.delta_params[name] = torch.zeros_like(param)

    def before_backward(self, strategy, **kwargs):
        if self.task_count > 0:
            penalty = torch.tensor(0.0).to(strategy.device)
            for name, param in strategy.model.named_parameters():
                if param.requires_grad and name in self.omega:
                    penalty += (self.omega[name] * (param - self.saved_params[name]) ** 2).sum()
            strategy.loss += self.si_lambda * penalty

    def after_update(self, strategy, **kwargs):
        for name, param in strategy.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name in self.saved_params:
                    delta_p = param.detach() - self.saved_params[name]
                    self.delta_params[name] += param.grad.detach() * delta_p

    def after_training_exp(self, strategy, **kwargs):
        for name, param in strategy.model.named_parameters():
            if param.requires_grad and name in self.saved_params:
                p_diff = param.detach() - self.saved_params[name]
                new_omega = self.delta_params[name] / (p_diff ** 2 + 1e-3)
                new_omega = torch.clamp(new_omega, min=0.0, max=10.0) 
                self.omega[name] += new_omega 
                self.delta_params[name].zero_()
                self.saved_params[name] = param.clone().detach()
        self.task_count += 1


# plasticity_metrics

def compute_plasticity_metrics(model, dataloader, device):
    """Calcule la magnitude moyenne des poids et le ratio de neurones morts sur TOUT le dataset."""
    model.eval()

    total_sum, total_count = 0.0, 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_sum += param.data.abs().sum().item()
            total_count += param.numel()
    avg_weight_mag = total_sum / total_count if total_count > 0 else 0.0

    active_counts = {}
    hooks = []

    def make_hook(layer_name):
        def hook(module, inp, out):
            if out.dim() == 4:
                out = out.amax(dim=(2, 3))
            is_active = (out > 0).float()
            is_active = is_active.view(is_active.size(0), -1) 
            
            if layer_name not in active_counts:
                active_counts[layer_name] = is_active.sum(dim=0).detach().cpu()
            else:
                active_counts[layer_name] += is_active.sum(dim=0).detach().cpu()
        return hook

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            if 'fc2' not in name and 'classifier' not in name:
                hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)
            model(inputs)

    for hook in hooks:
        hook.remove()

    dead_ratios = []
    if not active_counts: 
        return avg_weight_mag, 0.0 

    for name, counts in active_counts.items():
        dead = (counts == 0).float().mean().item()
        dead_ratios.append(dead)

    avg_dead = sum(dead_ratios) / len(dead_ratios) if dead_ratios else 0.0

    return avg_weight_mag, avg_dead


# graphiques Accuracy, Magnitude, Neurones morts

def generate_and_save_plots(exp_dir, task_order, final_results, plasticity_log, strategy_name):
    """
    Génère et sauvegarde automatiquement les 3 graphiques (Accuracy, Magnitude, Neurones morts)
    dans le dossier de l'expérience.
    """
    print("\n>>> Génération automatique des graphiques...")
    
    # 1. Extraction des données
    steps = list(range(1, len(task_order) + 1))
    x_labels = [f"Tâche {t}" for t in task_order]
    
    # Extraction Accuracy finale (depuis le dict Avalanche du dernier step)
    final_accs = []
    for t in task_order:
        # La clé Avalanche ressemble à : 'Top1_Acc_Stream/eval_phase/test_stream/Task00X'
        # On cherche la valeur correspondant à la tâche dans les résultats de la dernière évaluation
        acc_key = f"Top1_Acc_Exp/eval_phase/test_stream/Task{t:03d}/Exp{t:03d}"
        if acc_key in final_results:
            final_accs.append(final_results[acc_key] * 100) # Conversion en %
        else:
            final_accs.append(0.0)

    # Extraction Plasticity
    magnitudes = [log['weight_magnitude'] for log in plasticity_log]
    dead_units = [log['dead_units_ratio'] * 100 for log in plasticity_log] # Conversion en %

    # 2. Plot 1 : Accuracy par tâche (Bar chart)
    plt.figure(figsize=(8, 5))
    bars = plt.bar(x_labels, final_accs, color='royalblue', edgecolor='black', zorder=3)
    plt.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Seuil de Hasard (50%)', zorder=2)
    plt.title(f"Précision finale par tâche\n({strategy_name})", fontsize=12, fontweight='bold')
    plt.xlabel("Ordre d'apprentissage", fontsize=10)
    plt.ylabel('Accuracy Finale (%)', fontsize=10)
    plt.ylim(0, 110)
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    plt.legend()
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'accuracy_finale.png'), dpi=300)
    plt.close()

    # 3. Plot 2 : Évolution de la Magnitude
    plt.figure(figsize=(8, 5))
    plt.plot(steps, magnitudes, marker='o', color='darkorange', linewidth=2.5, markersize=8)
    plt.xticks(steps, x_labels)
    plt.title(f"Évolution de la Magnitude des Poids\n({strategy_name})", fontsize=12, fontweight='bold')
    plt.xlabel("Ordre d'apprentissage", fontsize=10)
    plt.ylabel('Magnitude moyenne absolue', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'magnitude_evolution.png'), dpi=300)
    plt.close()

    # 4. Plot 3 : Évolution des Neurones Morts
    plt.figure(figsize=(8, 5))
    plt.plot(steps, dead_units, marker='s', color='crimson', linewidth=2.5, markersize=8)
    plt.xticks(steps, x_labels)
    plt.title(f"Perte de Plasticité (Neurones Morts)\n({strategy_name})", fontsize=12, fontweight='bold')
    plt.xlabel("Ordre d'apprentissage", fontsize=10)
    plt.ylabel('Ratio de Neurones Morts (%)', fontsize=10)
    plt.ylim(bottom=0)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'dead_units_evolution.png'), dpi=300)
    plt.close()

    print(f">>> Graphiques sauvegardés dans : {exp_dir}")


# MAIN

def main(args):

    print("CUDA available:", torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    trial_id = get_random_string(5)
    print("trial_id:", trial_id)

    # Reproductibilité
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)

    # Benchmark 
    train_loaders, test_loaders, benchmark = get_benchmark(
        name=args.benchmark,
        num_tasks=args.num_tasks,
        degree=args.degree,
        batch_size=args.batch_size,
        k_shot=args.k_shot
    )

    original_classes = {}
    for task_id in range(args.num_tasks):
        original_classes[task_id] = benchmark.original_classes_in_exp[task_id]

    # Ordre des tâches
    task_order = get_task_order(
        metric_type=args.metric_type,
        metric=args.metric,
        ordering_heuristic=args.ordering_heuristic,
        task_orders_file=args.task_orders_file
    )

    # Modèle
    net = get_net(
        name=args.model,
        weights_path=args.pretrained_weights,
        num_classes=len(benchmark.original_classes_in_exp[0]),
        gray_scale=True if args.benchmark in [
            "SplitMNIST", "SplitFashionMNIST", "RotatedMNIST", "RotatedDSprites"
        ] else False,
        device=device,
        forWhat=args.forWhat
    )

    class LogitsExtractor(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, x):
            output = self.model(x)
            return output.logits if hasattr(output, 'logits') else output
    
    if hasattr(net, 'config') or "microsoft" in args.model:
        net = LogitsExtractor(net)

    print(f"Optimizer: SGD, lr={args.lr}")
    optimizer = SGD(net.parameters(), lr=args.lr, momentum=0.9)
    criterion = CrossEntropyLoss()

    original_classes_json = {
        str(task_id): list(classes) if isinstance(classes, set) else classes
        for task_id, classes in original_classes.items()
    }

    config = {
        'benchmark': args.benchmark,
        'model': args.model,
        'strategy': args.strategy,
        'ewc_lambda': args.ewc_lambda if args.strategy in ['ewc', 'rwalk'] else None,
        'metric_type': args.metric_type,
        'metric': args.metric,
        'ordering_heuristic': args.ordering_heuristic,
        'task_order': task_order,
        'original_classes': original_classes_json,
        'num_tasks': args.num_tasks,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'epochs': args.epochs,
        'k_shot': args.k_shot,
        'trial': trial_id,
        'exp_dir': './chains_of_transfer/results/{}'.format(trial_id)
    }

    os.makedirs(config['exp_dir'], exist_ok=True)
    with open(config['exp_dir'] + '/config.json', 'w') as f:
        json.dump(config, f, indent=4)

    loggers = [
        TextLogger(open(config['exp_dir'] + '/log.txt', 'a')),
        CSVLogger(log_folder=config['exp_dir']),
        InteractiveLogger(),
    ]

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        confusion_matrix_metrics(num_classes=benchmark.n_classes, save_image=True, stream=True),
        loggers=loggers
    )

    # Sélection de la stratégie
    strat_name_for_plot = f"{args.strategy.upper()}"
    if args.strategy == 'naive':
        print("\n>>> Strategy: NAIVE (No regularization)")
        cl_strategy = Naive(
            model=net, optimizer=optimizer, criterion=criterion,
            train_mb_size=args.batch_size, train_epochs=args.epochs,
            eval_mb_size=args.batch_size, evaluator=eval_plugin, device=device
        )
    elif args.strategy == 'ewc':
        strat_name_for_plot += f" ($\lambda={args.ewc_lambda}$)"
        print(f"\n>>> Strategy: EWC (lambda={args.ewc_lambda})")
        cl_strategy = EWC(
            model=net, optimizer=optimizer, criterion=criterion,
            ewc_lambda=args.ewc_lambda,
            train_mb_size=args.batch_size, train_epochs=args.epochs,
            eval_mb_size=args.batch_size, evaluator=eval_plugin, device=device
        )
    elif args.strategy == 'rwalk':
        strat_name_for_plot += f" ($\lambda={args.ewc_lambda}$)"
        print(f"\n>>> Strategy: RWALK/SI (si_lambda={args.ewc_lambda}) [CUSTOM PLUGIN + CLAMPING]")
        cl_strategy = Naive(
            model=net, optimizer=optimizer, criterion=criterion,
            train_mb_size=args.batch_size, train_epochs=args.epochs,
            eval_mb_size=args.batch_size, evaluator=eval_plugin, device=device,
            plugins=[CustomSIPlugin(si_lambda=args.ewc_lambda)]
        )
    else:
        raise ValueError(f"Stratégie inconnue : {args.strategy}")

    # Boucle d'entraînement
    results = []
    plasticity_log = []

    for step, task_id in enumerate(task_order):
        print(f"\n{'='*60}")
        print(f"Task order: {task_order}")
        print(f"Training on task {task_id} (step {step+1}/{len(task_order)})")
        print('='*60)

        train_exp = benchmark.train_stream[task_id]

        cl_strategy.train(train_exp, num_workers=4)
        
        # Évaluation sur TOUT le stream de test
        eval_res = cl_strategy.eval(benchmark.test_stream)
        results.append(eval_res)

        wm, dead = compute_plasticity_metrics(net, test_loaders[task_id], device)
        
        print("\n" + "★"*60)
        print(f"PLASTICITY METRICS AFTER TASK {task_id}:")
        print(f" -> Weight Magnitude : {wm:.6f}")
        print(f" -> Dead Units Ratio : {dead*100:.2f}%")
        print("★"*60 + "\n")

        plasticity_log.append({
            'step': step,
            'task_trained': task_id,
            'weight_magnitude': wm,
            'dead_units_ratio': dead
        })

    # --- SAUVEGARDE JSON ---
    plasticity_path = config['exp_dir'] + '/plasticity_metrics.json'
    with open(plasticity_path, 'w') as f:
        json.dump(plasticity_log, f, indent=4)
    print(f"\n Metrics saved to: {plasticity_path}")

    # --- GÉNÉRATION AUTOMATIQUE DES GRAPHIQUES ---
    # results[-1] contient le dictionnaire complet de la dernière évaluation (après toutes les tâches)
    generate_and_save_plots(config['exp_dir'], task_order, results[-1], plasticity_log, strat_name_for_plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--pretrained_weights", type=str, default=None)
    parser.add_argument("--forWhat", type=str, default="chains_of_transfer")
    
    parser.add_argument("--strategy", type=str, default="naive", choices=["naive", "ewc", "rwalk"])
    parser.add_argument("--ewc_lambda", type=float, default=0.001)

    parser.add_argument("--metric_type", type=str, default="random")
    parser.add_argument("--metric", type=str, default="random")
    parser.add_argument("--ordering_heuristic", type=str, default="random1")
    parser.add_argument("--task_orders_file", type=str, default="task_orderings.json")

    parser.add_argument("--num_tasks", type=int, required=True)
    parser.add_argument("--degree", type=int, default=None)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    
    parser.add_argument("--k_shot", type=int, default=None, help="Nombre d'images par classe à conserver (few-shot)")

    args = parser.parse_args()
    main(args)