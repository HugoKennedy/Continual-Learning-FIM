import torch
from torchvision import transforms
import numpy as np
from collections import defaultdict

from avalanche.benchmarks import RotatedMNIST, SplitCIFAR10, SplitCIFAR100, SplitImageNet, SplitTinyImageNet, SplitMNIST
from avalanche.benchmarks.classic.ctiny_imagenet import _default_train_transform, _default_eval_transform
from avalanche.training.determinism.rng_manager import RNGManager

#from benchmarks.rotated_fashionmnist import RotatedFashionMNIST
from benchmarks.rotateddsprites.rotated_dsprites_avlnch import RotatedDSprites


def get_benchmark(
        name: str,
        num_tasks: int,
        degree: int,
        batch_size: int = 10,
        k_shot: int = None
    ):

    # Set a fixed seed: must be kept the same across save/resume operations
    RNGManager.set_random_seeds(1234)

    match name:
        case "RotatedFashionMNIST" | "RotatedMNIST" | "RotatedDSprites":
            per_task_rotation = degree
            rotations_list = [per_task_rotation * i for i in range(num_tasks)]

            if name == "RotatedFashionMNIST":
                benchmark = RotatedFashionMNIST(
                    n_experiences=num_tasks,
                    rotations_list=rotations_list,
                    seed=1234,
                )
            elif name == "RotatedMNIST":
                benchmark = RotatedMNIST(
                    n_experiences=num_tasks,
                    rotations_list=rotations_list,
                    return_task_id=True,
                    train_transform=transforms.Compose([]),
                    eval_transform=transforms.Compose([]),
                    seed=1234,
                )
            elif name == "RotatedDSprites":
                benchmark = RotatedDSprites(
                    n_experiences=num_tasks,
                    rotations_list=rotations_list,
                    return_task_id=True,
                    class_ids_from_zero_in_each_exp=True,
                    seed=1234,
                )

            benchmark.rotations_list = rotations_list  # add attribute to benchmark for later use

        case "SplitMNIST":
            # * [:-1] remove normalization from _default_train_transform and _default_eval_transform
            # as ViT processor will handle normalization internally
            # otherwise BUG
            # * [:-2] otherwise bug with MNIST: ToTensor() expects PIL Image or ndarray, got Tensor
            # * NOT WORKNIG! add transforms.Grayscale(num_output_channels=3) to convert 1 channel to 3 channels. The Processor expects 3 channels.
            # All in all, not to be used with finetuning models pretrained on RGB images
            # Use training from scratch instead for SplitMNIST
            # ==> use SplitMNIST in conjunction with models with adapted first conv layer for gray scale images
            # ==> see gray_scale=True in get_net() in nets_utils.py
            #train_transform = transforms.Compose(_default_train_transform.transforms[:])
            #eval_transform = transforms.Compose(_default_eval_transform.transforms[:])

            benchmark = SplitMNIST(
                n_experiences=num_tasks,
                #return_task_id=False,  # BUG when training with MTL setting, we need task ids
                return_task_id=True,
                train_transform=transforms.Compose([]),
                eval_transform=transforms.Compose([]),
                class_ids_from_zero_in_each_exp=True,
                seed=1234
            )

        case "SplitCIFAR10":
            # remove normalization from _default_train_transform and _default_eval_transform
            # as ViT processor will handle normalization internally
            # otherwise BUG
            train_transform = transforms.Compose(_default_train_transform.transforms[:-1])
            eval_transform = transforms.Compose(_default_eval_transform.transforms[:-1])

            benchmark = SplitCIFAR10(
                n_experiences=num_tasks,
                #return_task_id=False,
                return_task_id=True,
                train_transform=train_transform,
                eval_transform=eval_transform,
                class_ids_from_zero_in_each_exp=True,
                seed=1234
            )
    
        case "SplitCIFAR100":
            # remove normalization from _default_train_transform and _default_eval_transform
            # as ViT processor will handle normalization internally
            # otherwise BUG
            train_transform = transforms.Compose(_default_train_transform.transforms[:-1])
            eval_transform = transforms.Compose(_default_eval_transform.transforms[:-1])

            benchmark = SplitCIFAR100(
                n_experiences=num_tasks,
                #return_task_id=False,
                return_task_id=True,
                train_transform=train_transform,
                eval_transform=eval_transform,
                class_ids_from_zero_in_each_exp=True,
                seed=1234
            )

        case "SplitImageNet":
            benchmark = SplitImageNet(
                dataset_root='benchmarks/data/',
                n_experiences=num_tasks,
                return_task_id=False,
                seed=1234
            )

        case "SplitTinyImageNet":
            # remove normalization from _default_train_transform and _default_eval_transform
            # as ViT processor will handle normalization internally
            # otherwise BUG
            train_transform = transforms.Compose(_default_train_transform.transforms[:-1])
            eval_transform = transforms.Compose(_default_eval_transform.transforms[:-1])

            benchmark = SplitTinyImageNet(
                n_experiences=num_tasks,
                return_task_id=False,
                train_transform=train_transform,
                eval_transform=eval_transform,
                class_ids_from_zero_in_each_exp=True,
                # param class_ids_from_zero_in_each_exp:
                # If True, original class IDs will be mapped to range [0, n_classes_in_exp)
                # for each experience. Defaults to False. Mutually exclusive with the
                # ``class_ids_from_zero_from_first_exp`` parameter.
                seed=1234
            )

        case _:
            raise ValueError(f"Unknown benchmark name: {name}")

    # get dataloaders for all tasks from train_stream
    train_loaders = [
        torch.utils.data.DataLoader(
            make_k_shot_subset(exp.dataset, k=k_shot, seed=1234) if k_shot is not None else exp.dataset,
            batch_size=batch_size,
            shuffle=False
            )
            for exp in benchmark.train_stream
        ]
    '''
    train_loaders = [
        torch.utils.data.DataLoader(
            exp.dataset, batch_size=batch_size, shuffle=False)
            for exp in benchmark.train_stream]
    '''
    test_loaders = [  # even if kshot, use all test samples for evaluation
        torch.utils.data.DataLoader(
            exp.dataset, batch_size=batch_size, shuffle=False)
            for exp in benchmark.test_stream]

    print('number of batches in train_loaders[0]: {} (where batch size is {})'.format(len(train_loaders[0]), batch_size))  # sanity check

    return train_loaders, test_loaders, benchmark




def make_k_shot_subset(dataset, k, labels=None, seed=1234):
    """
    dataset: any PyTorch dataset
    k: number of samples per class
    labels: optional pre-extracted labels (list or tensor)
    """
    print("Making k_shot subsets...", end=' ')
    
    rng = np.random.default_rng(seed)
    class_to_indices = defaultdict(list)

    if labels is None:
        for i in range(len(dataset)):
            _, y, _ = dataset[i]
            class_to_indices[y].append(i)
    else:
        for i, y in enumerate(labels):
            class_to_indices[int(y)].append(i)

    selected_indices = []
    for cls, idxs in class_to_indices.items():
        if len(idxs) < k:
            raise ValueError(f"Class {cls} has only {len(idxs)} samples (< {k})")
        selected_indices.extend(rng.choice(idxs, size=k, replace=False))

    print("DONE")

    return torch.utils.data.Subset(dataset, selected_indices)

