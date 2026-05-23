# Régularisation par importance et ordonnancement des tâches en apprentissage continu

## Étude empirique à travers le prisme de la Matrice d'Information de Fisher

Ce dépôt contient le code source, les scripts d'expérimentation et les outils d'analyse associés à un projet de recherche en **apprentissage continu** (*Continual Learning*).

L'objectif du projet est d'étudier l'interaction entre :

- les méthodes de **régularisation par importance des paramètres**, comme **Elastic Weight Consolidation (EWC)** et une variante custom de **RWalk / Synaptic Intelligence** ;
- l'**ordonnancement des tâches** à partir de distances entre tâches calculées via **Task2Vec** ;
- les métriques classiques de performance, comme l'accuracy finale et l'oubli moyen ;
- les métriques internes de plasticité du réseau, comme le ratio de neurones morts et la magnitude moyenne des poids.

Le fil conducteur scientifique du projet est la **Matrice d'Information de Fisher** (*Fisher Information Matrix*, FIM), utilisée à deux niveaux :

1. comme mesure d'importance des paramètres dans les méthodes de régularisation ;
2. comme signature de tâche via Task2Vec pour construire des curricula d'apprentissage.

---

## Table des matières

1. [Contexte scientifique](#1-contexte-scientifique)
2. [Question de recherche](#2-question-de-recherche)
3. [Méthodes étudiées](#3-méthodes-étudiées)
4. [Structure du projet](#4-structure-du-projet)
5. [Installation](#5-installation)
6. [Utilisation générale](#6-utilisation-générale)
7. [Arguments principaux de main.py](#7-arguments-principaux-de-mainpy)
8. [Exemples d'expériences](#8-exemples-dexpériences)
9. [Ordonnancement des tâches avec Task2Vec](#9-ordonnancement-des-tâches-avec-task2vec)
10. [Métriques calculées](#10-métriques-calculées)
11. [Fichiers de sortie](#11-fichiers-de-sortie)
12. [Analyse des résultats](#12-analyse-des-résultats)
13. [Implémentation custom RWalk / SI](#13-implémentation-custom-rwalk--si)
14. [Reproductibilité](#14-reproductibilité)
15. [Problèmes fréquents](#15-problèmes-fréquents)
16. [Auteurs](#16-auteurs)
17. [Résumé scientifique du projet](#17-résumé-scientifique-du-projet)

---

## 1. Contexte scientifique

L'apprentissage profond classique suppose souvent que toutes les données sont disponibles dès le départ, dans un cadre fixe et stationnaire. Le modèle est entraîné sur un jeu de données complet, puis évalué sur des données issues d'une distribution similaire.

En pratique, cette hypothèse est rarement réaliste. Dans de nombreuses situations, les données arrivent progressivement :

- de nouvelles classes apparaissent ;
- la distribution des données évolue ;
- les anciennes données ne sont plus accessibles ;
- il est trop coûteux de réentraîner le modèle depuis zéro ;
- il peut être impossible de stocker les anciennes données pour des raisons de confidentialité ou de mémoire.

C'est le cadre de l'**apprentissage continu** (*Continual Learning*).

Dans ce cadre, le modèle apprend une suite de tâches :

```text
Tâche 1 -> Tâche 2 -> Tâche 3 -> ... -> Tâche N
```

Le problème principal est que l'apprentissage d'une nouvelle tâche peut dégrader fortement les performances sur les anciennes tâches. Ce phénomène est appelé **oubli catastrophique**.

Il existe aussi un second problème, plus interne au réseau : la **perte de plasticité**.

La plasticité désigne la capacité du modèle à continuer à apprendre de nouvelles tâches. Un réseau peut devenir progressivement trop rigide, perdre des neurones actifs ou accumuler des poids de grande magnitude, ce qui peut rendre l'apprentissage futur plus difficile.

Le projet étudie donc simultanément deux phénomènes :

- **la stabilité**, c'est-à-dire la capacité à conserver les anciennes tâches ;
- **la plasticité**, c'est-à-dire la capacité à apprendre les nouvelles tâches.

---

## 2. Question de recherche

La question centrale du projet est la suivante :

> La similarité entre tâches mesurée à partir de la Matrice d'Information de Fisher peut-elle servir de proxy utile pour prédire ou améliorer les performances des méthodes de régularisation par importance en apprentissage continu ?

Autrement dit, le projet cherche à savoir si la Fisher peut être utilisée à la fois :

1. pour protéger les paramètres importants du réseau ;
2. pour organiser les tâches dans un ordre plus intelligent.

L'hypothèse principale est que la FIM peut être utile si elle est utilisée à deux niveaux :

- au niveau des **paramètres**, avec EWC ;
- au niveau des **tâches**, avec Task2Vec.

Le projet teste notamment si un ordre de tâches basé sur Task2Vec, comme l'ordre Hamiltonien, permet de réduire l'oubli lorsqu'il est combiné à une régularisation EWC.

---

## 3. Méthodes étudiées

### 3.1 Stratégie naïve

La stratégie naïve consiste à entraîner le modèle séquentiellement sur les tâches, sans mécanisme de mémoire.

Le modèle apprend la tâche courante, puis passe à la suivante. Aucune contrainte ne l'empêche de modifier les paramètres utiles aux anciennes tâches.

C'est la baseline principale du projet. Elle permet de mesurer l'oubli catastrophique naturel du modèle.

---

### 3.2 Elastic Weight Consolidation, EWC

EWC est une méthode de régularisation par importance.

L'idée est la suivante :

- après avoir appris une tâche, on estime quels paramètres sont importants ;
- lors des tâches suivantes, on pénalise les modifications de ces paramètres ;
- cette importance est calculée à partir de la diagonale de la Matrice d'Information de Fisher.

La loss d'EWC peut s'écrire schématiquement :

```text
Loss totale = Loss nouvelle tâche + pénalité EWC
```

avec une pénalité du type :

```text
lambda * somme_i Fisher_i * (theta_i - theta_i_old)^2
```

Le coefficient `lambda` contrôle la force de la régularisation.

- Si `lambda` est trop faible, EWC se comporte presque comme la stratégie naïve.
- Si `lambda` est trop fort, le modèle peut devenir trop rigide ou instable.
- Une valeur intermédiaire peut produire un meilleur compromis stabilité-plasticité.

---

### 3.3 Synaptic Intelligence / RWalk custom

Le projet inclut une stratégie appelée `rwalk`.

En pratique, cette stratégie correspond à une implémentation custom inspirée de **Synaptic Intelligence** et de **Riemannian Walk**.

L'idée générale est de mesurer l'importance des paramètres non seulement à la fin d'une tâche, mais aussi pendant la trajectoire d'apprentissage.

Un paramètre est considéré comme important s'il a contribué de manière significative à la diminution de la loss pendant l'entraînement.

Cette stratégie est plus délicate à stabiliser numériquement que EWC. Le projet inclut donc une version custom avec :

- un epsilon plus grand pour éviter les divisions instables ;
- un clamping des importances pour éviter les valeurs extrêmes ;
- une gestion plus robuste des dictionnaires de paramètres.

---

### 3.4 Task2Vec

Task2Vec permet de représenter une tâche par un vecteur d'embedding.

Dans ce projet, chaque tâche est représentée à partir d'une signature liée à la Fisher. Deux tâches proches dans l'espace Task2Vec sont supposées mobiliser des paramètres similaires.

Task2Vec permet ensuite de construire plusieurs ordres de tâches :

- ordre aléatoire ;
- ordre Hamiltonien ;
- ordre par centralité ;
- ordre par clustering.

L'objectif est d'étudier si un meilleur ordre de présentation des tâches peut réduire l'oubli.

---

## 4. Structure du projet

La structure exacte du dépôt peut varier légèrement selon l'organisation locale, mais le projet suit généralement cette logique :

```text
.
├── README.md
├── requirements.txt
├── main.py
├── benchmarks/
│   └── ...
├── nets/
│   └── ...
├── task_order/
│   ├── ...
│   └── results/
├── scripts/
│   ├── run_naive_random.sh
│   ├── run_naive_task2vec.sh
│   ├── run_ewc_random.sh
│   ├── run_ewc_task2vec.sh
│   ├── run_rwalk_task2vec.sh
│   └── ...
├── results/
│   ├── ...
│   └── trial_id/
│       ├── metrics.csv
│       ├── plasticity_metrics.json
│       ├── accuracy_*.png
│       ├── magnitude_*.png
│       └── NM_*.png
└── figures/
    └── ...
```

Selon la version du projet, `main.py` peut être situé à la racine ou dans un sous-dossier comme `chains_of_transfer/`.

Les scripts Bash permettent de lancer automatiquement les expériences principales.

Les résultats sont stockés dans des dossiers associés à chaque run ou à chaque `trial_id`.

---

## 5. Installation

### 5.1 Créer un environnement virtuel

Il est recommandé d'utiliser un environnement virtuel Python.

Avec `venv` :

```bash
python -m venv .venv
source .venv/bin/activate
```

Sous Windows :

```bash
python -m venv .venv
.venv\Scripts\activate
```

Avec `conda` :

```bash
conda create -n continual-learning python=3.10
conda activate continual-learning
```

---

### 5.2 Installer les dépendances

Si un fichier `requirements.txt` est fourni :

```bash
pip install -r requirements.txt
```

Sinon, installer les dépendances principales manuellement :

```bash
pip install torch torchvision
pip install avalanche-lib
pip install numpy pandas matplotlib seaborn
pip install scikit-learn tqdm
```

Selon l'implémentation Task2Vec utilisée, il peut aussi être nécessaire d'installer :

```bash
pip install task2vec
```

ou d'utiliser une version locale du code Task2Vec.

---

### 5.3 Vérifier l'installation

Pour vérifier que PyTorch fonctionne :

```bash
python -c "import torch; print(torch.__version__)"
```

Pour vérifier qu'Avalanche est installé :

```bash
python -c "import avalanche; print('Avalanche OK')"
```

---

## 6. Utilisation générale

Le script principal du projet est `main.py`.

Une expérience se lance typiquement avec :

```bash
python main.py \
  --model simplecnn \
  --strategy ewc \
  --ewc_lambda 0.1 \
  --ordering_heuristic hamiltonian
```

Selon l'organisation du dépôt, il peut être nécessaire de se placer dans le bon dossier :

```bash
cd chains_of_transfer
python main.py ...
```

ou d'appeler directement le script depuis la racine :

```bash
python chains_of_transfer/main.py ...
```

---

## 7. Arguments principaux de main.py

Les arguments peuvent varier légèrement selon la version finale du code, mais les principaux paramètres sont les suivants.

### 7.1 Modèle

Argument : `--model`

Choix du modèle utilisé.

Exemples possibles :

```bash
--model simplecnn
--model microsoft_resnet50
--model resnet50
```

Le modèle principal du projet est `simplecnn`.

ResNet50 est utilisé pour étudier l'effet de la capacité du modèle et de la sur-paramétrisation.

---

### 7.2 Stratégie de continual learning

Argument : `--strategy`

Choix de la stratégie d'apprentissage continu.

Valeurs typiques :

```bash
--strategy naive
--strategy ewc
--strategy rwalk
```

- `naive` : entraînement séquentiel sans mémoire ;
- `ewc` : régularisation Elastic Weight Consolidation ;
- `rwalk` : stratégie custom inspirée de SI / RWalk.

---

### 7.3 Force de régularisation

Argument : `--ewc_lambda`

Coefficient de régularisation.

Exemples :

```bash
--ewc_lambda 0.001
--ewc_lambda 0.01
--ewc_lambda 0.1
--ewc_lambda 1.0
```

Pour EWC, ce coefficient contrôle la force de la pénalité Fisher.

Pour la stratégie `rwalk`, il est également utilisé comme coefficient de régularisation dans le plugin custom.

---

### 7.4 Ordonnancement des tâches

Argument : `--ordering_heuristic`

Choix de l'ordre des tâches.

Exemples :

```bash
--ordering_heuristic random
--ordering_heuristic random1
--ordering_heuristic hamiltonian
--ordering_heuristic centrality
--ordering_heuristic clustering
```

Les ordres Task2Vec sont généralement pré-calculés et stockés dans des fichiers de résultats.

---

### 7.5 Nombre d'échantillons par classe

Argument : `--k_shot`

Permet de limiter le nombre d'exemples par classe.

Exemple :

```bash
--k_shot 100
```

Dans les expériences principales du projet, la restriction `k_shot` a été retirée afin d'utiliser tout le dataset disponible.

---

### 7.6 Autres arguments possibles

Selon la version de `main.py`, d'autres arguments peuvent être disponibles :

```bash
--epochs
--batch_size
--lr
--seed
--num_tasks
--dataset
--output_dir
--trial_id
```

Il est possible d'afficher tous les arguments avec :

```bash
python main.py --help
```

---

## 8. Exemples d'expériences

### 8.1 Baseline naïve avec ordre aléatoire

Objectif : mesurer l'oubli catastrophique naturel du modèle.

```bash
python main.py \
  --model simplecnn \
  --strategy naive \
  --ordering_heuristic random
```

Cette expérience sert de baseline.

On s'attend généralement à observer :

- une bonne performance sur les tâches récentes ;
- une forte dégradation sur les anciennes tâches ;
- une plasticité globalement préservée ;
- un oubli catastrophique marqué.

---

### 8.2 Naïve avec ordre Hamiltonien Task2Vec

Objectif : tester si l'ordre Task2Vec seul suffit à réduire l'oubli.

```bash
python main.py \
  --model simplecnn \
  --strategy naive \
  --ordering_heuristic hamiltonian
```

Cette expérience permet de distinguer :

- l'effet du curriculum ;
- l'effet d'un mécanisme explicite de mémoire.

Résultat attendu :

```text
Task2Vec structure le stream, mais ne protège pas les paramètres.
```

---

### 8.3 EWC avec lambda faible

Objectif : tester une régularisation trop faible.

```bash
python main.py \
  --model simplecnn \
  --strategy ewc \
  --ewc_lambda 0.001 \
  --ordering_heuristic random
```

Interprétation typique :

```text
La pénalité est trop faible pour modifier significativement la trajectoire d'apprentissage.
```

---

### 8.4 EWC avec lambda 0.1 et ordre aléatoire

Objectif : observer l'effet d'une régularisation EWC plus forte.

```bash
python main.py \
  --model simplecnn \
  --strategy ewc \
  --ewc_lambda 0.1 \
  --ordering_heuristic random
```

Cette expérience permet d'observer l'apparition d'une mémoire paramétrique.

---

### 8.5 EWC avec lambda 0.1 et ordre Hamiltonien

Objectif : tester la synergie entre Fisher-paramètres et Fisher-tâches.

```bash
python main.py \
  --model simplecnn \
  --strategy ewc \
  --ewc_lambda 0.1 \
  --ordering_heuristic hamiltonian
```

C'est l'une des configurations centrales du projet.

Elle teste l'hypothèse suivante :

```text
La FIM est plus utile lorsqu'elle sert à la fois à protéger les paramètres et à lisser les transitions entre tâches.
```

---

### 8.6 EWC avec centralité

```bash
python main.py \
  --model simplecnn \
  --strategy ewc \
  --ewc_lambda 0.1 \
  --ordering_heuristic centrality
```

Objectif : tester un ordre Task2Vec fondé sur une tâche centrale du graphe.

---

### 8.7 EWC avec clustering

```bash
python main.py \
  --model simplecnn \
  --strategy ewc \
  --ewc_lambda 0.1 \
  --ordering_heuristic clustering
```

Objectif : tester l'effet d'un regroupement des tâches similaires.

Cette expérience est importante car elle montre qu'exploiter la similarité Task2Vec de manière naïve peut dégrader les performances.

Le clustering peut créer des transitions brutales entre groupes de tâches.

---

### 8.8 RWalk / SI custom

```bash
python main.py \
  --model simplecnn \
  --strategy rwalk \
  --ewc_lambda 0.01 \
  --ordering_heuristic hamiltonian
```

Objectif : tester une régularisation trajectorielle inspirée de SI / RWalk.

Cette stratégie peut être plus instable que EWC et doit être interprétée comme une variante custom stabilisée.

---

### 8.9 Expérience avec ResNet50

```bash
python main.py \
  --model microsoft_resnet50 \
  --strategy ewc \
  --ewc_lambda 0.1 \
  --ordering_heuristic hamiltonian
```

Objectif : étudier l'effet de la capacité du modèle.

ResNet50 permet d'observer si une architecture plus grande et normalisée change la dynamique de l'oubli, des poids et des neurones morts.

---

## 9. Ordonnancement des tâches avec Task2Vec

Le projet étudie plusieurs manières d'organiser les tâches.

### 9.1 Ordre aléatoire

L'ordre aléatoire sert de baseline.

Il ne tient pas compte de la similarité entre tâches.

---

### 9.2 Ordre Hamiltonien

L'ordre Hamiltonien cherche à minimiser les distances Task2Vec entre tâches successives.

Intuition : éviter les transitions brutales entre tâches.

C'est l'ordre qui donne généralement le meilleur compromis dans le projet lorsqu'il est combiné à EWC.

---

### 9.3 Centralité

La centralité consiste à commencer par une tâche centrale dans le graphe de similarité.

Intuition : commencer par une tâche représentative des autres.

---

### 9.4 Clustering

Le clustering regroupe les tâches proches en blocs.

Intuition initiale : apprendre ensemble les tâches similaires.

Mais les résultats montrent que cette stratégie peut être problématique, car elle peut créer de fortes ruptures entre clusters.

---

## 10. Métriques calculées

Le projet ne se limite pas à l'accuracy. Il mesure aussi l'oubli et l'état interne du réseau.

### 10.1 Accuracy finale

L'accuracy finale mesure la performance du modèle après apprentissage de toutes les tâches.

Elle est calculée sur les tâches du stream de test.

---

### 10.2 Oubli moyen

L'oubli mesure la perte de performance sur une tâche après apprentissage des tâches suivantes.

Pour une tâche `j`, on peut le définir comme :

```text
oubli_j = performance après apprentissage de j - performance finale sur j
```

Un oubli élevé indique que le modèle a fortement perdu la tâche.

---

### 10.3 Ratio de neurones morts

Le ratio de neurones morts mesure la proportion de neurones ReLU qui ne s'activent jamais sur l'ensemble évalué.

Formule :

```text
D = nombre de neurones ReLU jamais activés / nombre total de neurones ReLU observés
```

Un neurone mort est une unité dont la sortie reste toujours nulle.

Pour les couches convolutionnelles, un filtre est considéré actif s'il produit au moins une activation positive sur une image.

Cette métrique est calculée à l'aide de forward hooks dans le modèle.

---

### 10.4 Magnitude moyenne des poids

La magnitude moyenne des poids mesure la moyenne des valeurs absolues des poids du réseau.

Formule :

```text
M = moyenne(|w_i|)
```

Cette métrique permet d'observer si les poids augmentent fortement au cours de l'apprentissage.

Une forte croissance peut indiquer une tension structurelle ou une rigidification du réseau.

---

### 10.5 Croissance relative de magnitude

Pour comparer des modèles de tailles différentes, comme SimpleCNN et ResNet50, il est plus pertinent d'utiliser la croissance relative :

```text
croissance_relative = (M_t - M_0) / M_0
```

Cela permet de comparer l'évolution des poids indépendamment de l'échelle initiale du modèle.

---

## 11. Fichiers de sortie

Chaque expérience génère généralement un dossier de résultats.

Ce dossier peut contenir :

```text
metrics.csv
plasticity_metrics.json
accuracy_*.png
magnitude_*.png
NM_*.png
logs.txt
config.json
```

### 11.1 metrics.csv

Contient les métriques principales de performance, par exemple :

- accuracy ;
- forgetting ;
- stream accuracy ;
- métriques Avalanche.

---

### 11.2 plasticity_metrics.json

Contient les métriques internes calculées après chaque tâche.

Exemple de structure :

```json
[
  {
    "step": 0,
    "task_trained": 0,
    "weight_magnitude": 0.0123,
    "dead_units_ratio": 0.004
  },
  {
    "step": 1,
    "task_trained": 1,
    "weight_magnitude": 0.0131,
    "dead_units_ratio": 0.006
  }
]
```

---

### 11.3 Graphiques générés

Les graphiques principaux peuvent inclure :

```text
accuracy_naive_random.png
accuracy_ewc_0.1_t2v_hamiltonian.png
accuracy_rwalk_0.01_t2v_hamiltonian.png

magnitude_naive_random.png
magnitude_ewc_0.1_t2v_hamiltonian.png
magnitude_rawlk_0.01_t2v_hamiltonian.png

NM_naive_random.png
NM_ewc_0.1_t2v_hamiltonian.png
NM_rwalk_0.01_t2v_hamiltonian.png
```

Dans les noms de fichiers :

```text
NM = neurones morts
```

---

## 12. Analyse des résultats

L'analyse du projet repose sur plusieurs comparaisons.

### 12.1 Naïve vs EWC

Cette comparaison permet de mesurer l'intérêt d'une régularisation par importance.

- Naïve apprend les tâches récentes mais oublie les anciennes.
- EWC cherche à protéger les paramètres importants.
- L'effet dépend fortement de `lambda`.

---

### 12.2 Ordre aléatoire vs Task2Vec

Cette comparaison permet de mesurer l'effet de l'ordre des tâches.

Résultat important : Task2Vec seul ne suffit pas à empêcher l'oubli.

Un bon ordre peut lisser les transitions, mais il ne remplace pas un mécanisme de mémoire.

---

### 12.3 EWC aléatoire vs EWC Hamiltonien

Cette comparaison est centrale.

Elle teste si la Fisher est plus utile lorsqu'elle agit à deux niveaux :

- protection des paramètres ;
- choix des transitions de tâches.

Résultat principal : EWC + ordre Hamiltonien donne le meilleur compromis observé entre oubli et plasticité.

---

### 12.4 Hamiltonien vs clustering

Le clustering peut sembler intuitivement bon car il regroupe les tâches similaires.

Mais dans un stream séquentiel, ce qui compte n'est pas seulement la similarité à l'intérieur d'un groupe.

Ce qui compte aussi, c'est la transition entre deux tâches consécutives.

Le clustering peut créer des ruptures brutales entre groupes, ce qui augmente l'interférence.

---

### 12.5 SimpleCNN vs ResNet50

ResNet50 permet d'étudier l'effet de la capacité du modèle.

Un modèle plus grand peut mieux absorber certaines contraintes, mais il ne supprime pas automatiquement l'oubli.

Les métriques internes, comme les neurones morts, doivent être interprétées différemment selon l'architecture.

---

## 13. Implémentation custom RWalk / SI

La stratégie `rwalk` du projet ne doit pas être interprétée comme une reproduction officielle exhaustive de RWalk.

Elle correspond à une variante custom inspirée de SI / RWalk.

### 13.1 Motivation

Certaines implémentations de SI ou RWalk peuvent être instables dans un cadre expérimental avec :

- têtes de classification dynamiques ;
- petits modèles non normalisés ;
- valeurs d'importance très grandes ;
- divisions par des déplacements très faibles ;
- gradients explosifs ;
- valeurs `NaN`.

---

### 13.2 Solution implémentée

Le projet utilise un plugin custom qui stabilise l'accumulation d'importance.

La mise à jour de l'importance peut se comprendre schématiquement comme :

```text
Omega_i <- Omega_i + clamp(Delta_i / ((theta_i - theta_i_old)^2 + epsilon), 0, 10)
```

avec :

- `Omega_i` : importance accumulée du paramètre ;
- `Delta_i` : contribution liée aux gradients et au déplacement du paramètre ;
- `epsilon` : terme de stabilisation ;
- `clamp` : limitation des valeurs extrêmes.

---

### 13.3 Interprétation

Cette correction rend la méthode testable, mais elle constitue aussi une limite méthodologique.

Les résultats doivent être lus comme ceux d'une variante stabilisée inspirée de RWalk / SI, et non comme ceux d'une implémentation officielle exacte de RWalk.

---

## 14. Reproductibilité

Pour assurer la reproductibilité des expériences, il est recommandé de :

1. fixer les seeds ;
2. sauvegarder les arguments de chaque run ;
3. conserver les fichiers JSON d'ordre des tâches ;
4. conserver les logs bruts ;
5. sauvegarder les graphiques générés ;
6. documenter la version de PyTorch et d'Avalanche.

Exemple :

```bash
python main.py \
  --model simplecnn \
  --strategy ewc \
  --ewc_lambda 0.1 \
  --ordering_heuristic hamiltonian \
  --seed 0
```

Pour vérifier les versions :

```bash
python -c "import torch; print('torch:', torch.__version__)"
python -c "import avalanche; print('avalanche ok')"
```

---

## 15. Problèmes fréquents

### 15.1 Erreur liée à Avalanche

Si Avalanche n'est pas installé :

```bash
pip install avalanche-lib
```

Si l'erreur persiste, vérifier la version de Python et de PyTorch.

---

### 15.2 CUDA out of memory

Réduire la taille du batch :

```bash
--batch_size 32
```

ou utiliser un modèle plus petit :

```bash
--model simplecnn
```

---

### 15.3 Loss qui devient NaN

Causes possibles :

- `lambda` trop grand ;
- gradients explosifs ;
- importance RWalk / SI trop élevée ;
- learning rate trop élevé.

Solutions possibles :

```bash
--ewc_lambda 0.01
```

ou réduire le learning rate :

```bash
--lr 0.0001
```

---

### 15.4 RWalk instable

Utiliser une valeur de régularisation plus faible :

```bash
--strategy rwalk --ewc_lambda 0.01
```

Éviter les valeurs trop grandes comme :

```bash
--ewc_lambda 0.1
```

si elles provoquent des NaN ou une mort massive des neurones.

---

### 15.5 Les figures ne sont pas générées

Vérifier que :

- le dossier de sortie existe ;
- `plasticity_metrics.json` est bien créé ;
- `matplotlib` est installé ;
- le script a terminé sans erreur ;
- les chemins de sauvegarde sont corrects.

---

## 16. Auteurs

Projet réalisé dans le cadre du Projet Cassiopée n°18.

Auteurs :

- Raphaël Seran
- Hugo Kennedy
- Othmane Himmi

---

## 17. Résumé scientifique du projet

Le projet montre que la Matrice d'Information de Fisher est un outil utile mais insuffisant pour résoudre l'apprentissage continu.

Elle est utile parce qu'elle permet :

- de mesurer l'importance des paramètres ;
- de construire des signatures de tâches ;
- de guider un curriculum d'apprentissage.

Mais elle est insuffisante parce que :

- Task2Vec seul ne protège pas les paramètres ;
- une mauvaise heuristique, comme le clustering, peut aggraver l'oubli ;
- EWC dépend fortement du choix de `lambda` ;
- l'architecture du modèle change l'interprétation des métriques ;
- accuracy, oubli, neurones morts et magnitude mesurent des phénomènes différents.

La conclusion principale est donc :

```text
La FIM n'est pas un oracle de performance.
Elle est un proxy local de l'interférence entre tâches.
Elle devient réellement utile lorsqu'elle est combinée à une régularisation bien calibrée,
un curriculum progressif et une architecture adaptée.
```
