from typing import Any, Optional, Sequence
import torch
from torch.utils.data import random_split
from torchvision.transforms import Compose, Normalize

from avalanche.benchmarks import NCScenario, nc_benchmark
from avalanche.benchmarks.utils import (
    _make_taskaware_classification_dataset,
    DefaultTransformGroups,
)
from avalanche.benchmarks.utils.data import make_avalanche_dataset
from avalanche.training.determinism.rng_manager import RNGManager

from benchmarks.rotateddsprites.rotated_dsprites import dSpritesPerRotationTask

# dSprites uses 40 discrete rotation bins across 360 degrees.
_ROTATION_BIN_SIZE_DEG = 360 // 40

# Default transforms (dSprites is grayscale 64x64, values in [0, 1])
_default_dsprites_train_transform = Compose([])
_default_dsprites_eval_transform = Compose([])



def RotatedDSprites(
    n_experiences: int = 10,
    rotations_list: Optional[Sequence[int]] = None,
    dsprites_root: str = "./benchmarks/data/",
    train_transform: Optional[Any] = _default_dsprites_train_transform,
    eval_transform: Optional[Any] = _default_dsprites_eval_transform,
    train_fraction: float = 0.8,
    seed: int = 1234,
    return_task_id: bool = False,
    class_ids_from_zero_in_each_exp: bool = True,
) -> NCScenario:
    """
    Build a Rotated dSprites benchmark akin to Avalanche's RotatedMNIST.

    Each experience contains all shapes from a single rotation bin of the
    dSprites dataset. ``rotations_list`` defines the rotation degrees for each
    experience (multiples of 9 degrees are valid, since dSprites has 40 bins).
    
    :param n_experiences: Number of experiences (rotation bins).
    :param rotations_list: List of rotation degrees. If None, rotations are
        spaced uniformly from 0 to 360.
    :param dsprites_root: Path to dSprites data root.
    :param train_transform: Transform applied after rotation on train data.
    :param eval_transform: Transform applied after rotation on eval data.
    :param train_fraction: Fraction of data used for training (vs test).
    :param seed: Random seed for determinism.
    :param return_task_id: If True, each experience has a task ID.
    :param class_ids_from_zero_in_each_exp: Map class IDs to [0, n_classes).
    
    :return: An NCScenario benchmark.
    """

    RNGManager.set_random_seeds(seed)

    if rotations_list is None:
        step = 360 // n_experiences
        rotations_list = [step * i for i in range(n_experiences)]
    else:
        if len(rotations_list) != n_experiences:
            raise ValueError(
                f"rotations_list length ({len(rotations_list)}) must match "
                f"n_experiences ({n_experiences})"
            )

    list_train_dataset = []
    list_test_dataset = []
    split_generator = torch.Generator().manual_seed(seed)

    for task_id, rotation_degree in enumerate(rotations_list):
        rotation_bin = int((rotation_degree // _ROTATION_BIN_SIZE_DEG) % 40)

        # Load base dSprites dataset for this rotation
        base_dataset = dSpritesPerRotationTask(
            path=dsprites_root,
            target_rotation=rotation_bin,
            transform=None,
        )

        # Split into train/test
        train_size = int(train_fraction * len(base_dataset))
        test_size = len(base_dataset) - train_size
        train_ds, test_ds = random_split(
            base_dataset, [train_size, test_size], generator=split_generator
        )

        # Add targets for Avalanche compatibility
        train_ds.targets = [base_dataset.labels[i] for i in train_ds.indices]
        test_ds.targets = [base_dataset.labels[i] for i in test_ds.indices]

        # Wrap with task awareness and make avalanche datasets with frozen transforms
        # (no rotation transform needed since we already selected the rotation bin)
        train_avalanche_ds = make_avalanche_dataset(
            _make_taskaware_classification_dataset(train_ds),
            frozen_transform_groups=DefaultTransformGroups((None, None)),
        )

        test_avalanche_ds = make_avalanche_dataset(
            _make_taskaware_classification_dataset(test_ds),
            frozen_transform_groups=DefaultTransformGroups((None, None)),
        )

        list_train_dataset.append(train_avalanche_ds)
        list_test_dataset.append(test_avalanche_ds)

    return nc_benchmark(
        list_train_dataset,
        list_test_dataset,
        n_experiences=len(list_train_dataset),
        task_labels=return_task_id,
        shuffle=False,
        class_ids_from_zero_in_each_exp=class_ids_from_zero_in_each_exp,
        one_dataset_per_exp=True,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )
