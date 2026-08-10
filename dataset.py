import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import Subset, DataLoader
from torchvision.transforms import AutoAugmentPolicy
from PIL import Image
from pathlib import Path
import numpy as np
import os

def CIFAR10(root = "./data", download = False, val = False, batch_size=128, config = None):
    transform_aug = []

    train_transform = transforms.Compose(
        transform_aug + [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if val :
        np.random.seed(0)
        split_permutation = list(np.random.permutation(50000))

        train_set = Subset(torchvision.datasets.CIFAR10(root = root, train=True, transform = train_transform, download=download), split_permutation[:45000])
        val_set = Subset(torchvision.datasets.CIFAR10(root = root, train=True, transform = test_transform, download=download), split_permutation[45000:])
        test_set = torchvision.datasets.CIFAR10(root = root , train=False, transform = test_transform, download=download)
    
        train_loader = DataLoader(train_set,batch_size=batch_size,  pin_memory=True, shuffle = True)
        val_loader   = DataLoader(val_set,batch_size=batch_size,  pin_memory=True, shuffle = False)
        test_loader   = DataLoader(test_set,batch_size=batch_size,  pin_memory=True, shuffle = False)


        return train_loader, val_loader, test_loader
    else:
        train_set = torchvision.datasets.CIFAR10(root = root, train=True, transform = train_transform, download=download)
        test_set = torchvision.datasets.CIFAR10(root = root , train=False, transform = test_transform, download=download)
    
        train_loader = DataLoader(train_set,batch_size=batch_size,  pin_memory=True, shuffle = True)
        val_loader   = None
        test_loader   = DataLoader(test_set,batch_size=batch_size,  pin_memory=True, shuffle = False)


        return train_loader, val_loader, test_loader


def CIFAR100(root = "./data", download = False, val = False, batch_size=128, config = None):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if val:
        np.random.seed(0)
        split_permutation = list(np.random.permutation(50000))
        train_set = Subset(torchvision.datasets.CIFAR100(root=root, train=True, transform=train_transform, download=download), split_permutation[:45000])
        val_set   = Subset(torchvision.datasets.CIFAR100(root=root, train=True, transform=test_transform, download=download), split_permutation[45000:])
        test_set  = torchvision.datasets.CIFAR100(root=root, train=False, transform=test_transform, download=download)
        train_loader = DataLoader(train_set, batch_size=batch_size, pin_memory=True, shuffle=True)
        val_loader   = DataLoader(val_set, batch_size=batch_size, pin_memory=True, shuffle=False)
        test_loader  = DataLoader(test_set, batch_size=batch_size, pin_memory=True, shuffle=False)
        return train_loader, val_loader, test_loader
    else:
        train_set = torchvision.datasets.CIFAR100(root=root, train=True, transform=train_transform, download=download)
        test_set  = torchvision.datasets.CIFAR100(root=root, train=False, transform=test_transform, download=download)
        train_loader = DataLoader(train_set, batch_size=batch_size, pin_memory=True, shuffle=True)
        val_loader   = None
        test_loader  = DataLoader(test_set, batch_size=batch_size, pin_memory=True, shuffle=False)
        return train_loader, val_loader, test_loader


def TinyImageNet(root = "./data", download = False, val = False, batch_size=128, config = None):
    """Tiny-ImageNet-200: 100k train / 10k val, 200 classes, 64x64. Same signature and same
    (train_loader, val_loader, test_loader) contract as CIFAR10/CIFAR100 above.

    Transforms match ADR's `src/dataset/build_dataset.py` exactly (RandomCrop(64, padding=4) +
    horizontal flip + ToTensor; test is ToTensor only) so the comparison stays like-for-like.
    As with CIFAR here, normalization is NOT applied in the loader -- it lives in the model, and
    the attack operates in [0,1] pixel space.

    Layout expected under `root/tiny-imagenet-200`:
        train/<wnid>/images/*.JPEG      (as shipped; ImageFolder recurses, so this works as-is)
        val/<wnid>/*.JPEG               (NOT as shipped -- the official val/ is flat images/ plus
                                         val_annotations.txt, and must be regrouped by class first;
                                         see scripts/prepare_tiny_imagenet.py)
    `download` is accepted for signature compatibility but ignored: there is no torchvision
    downloader for this dataset.

    ------------------------------------------------------------------------------------------
    READ THIS BEFORE TOUCHING THE SPLITS -- the naming is genuinely confusing here.

    Tiny-ImageNet ships THREE directories, but only two are usable:

        train/   100,000 imgs   labelled     -> training
        val/      10,000 imgs   labelled     -> THIS IS OUR TEST SET
        test/     10,000 imgs   NO LABELS    -> unusable, competition holdout, never touched

    The directory named `test/` has no public labels, so accuracy cannot be computed on it.
    ADR (`src/dataset/build_dataset.py`, `is_train=False` -> val/) and the AT literature all
    report their numbers on the labelled `val/` split. We do the same, so our clean/PGD/CW/AA
    are directly comparable to theirs. Hence: **official val/ is returned as `test_loader`.**

    The `val` ARGUMENT of this function is a different thing entirely. It mirrors CIFAR10/
    CIFAR100 above and means "carve a held-out split out of train" (for methods that need a
    meta/validation batch). It never touches the evaluation set.

        val=False (default)   train_loader 100,000 | val_loader None   | test_loader 10,000
        val=True              train_loader  90,000 | val_loader 10,000 | test_loader 10,000

    `test_loader` is the official val/ split in BOTH cases. Note that val=True trains on 10%
    less data than ADR did, so leave it off for any run that goes into a comparison table.
    ------------------------------------------------------------------------------------------
    """
    # main.py calls this as root=<data_root>/<config.dataset>, i.e. ./data/TinyImageNet, while a
    # direct call or the prepare script uses ./data. The extracted archive is always named
    # tiny-imagenet-200, so accept either nesting rather than forcing a 120k-file move.
    root = str(root).rstrip("/\\")
    candidates = [
        os.path.join(root, "tiny-imagenet-200"),                       # root=./data
        os.path.join(os.path.dirname(root), "tiny-imagenet-200"),      # root=./data/TinyImageNet
        root,                                                          # root already the dataset dir
    ]
    data_dir = next(
        (c for c in candidates
         if os.path.isdir(os.path.join(c, "train")) and os.path.isdir(os.path.join(c, "val"))),
        None)
    if data_dir is None:
        raise FileNotFoundError(
            "Tiny-ImageNet not found. Looked for train/ and val/ under:\n  "
            + "\n  ".join(candidates)
            + "\nRun: python scripts/prepare_tiny_imagenet.py --root ./data")

    train_dir = os.path.join(data_dir, "train")
    # Deliberately val/, not test/ -- see the block above. Named `eval_dir` so no later reader
    # has to decide whether "test_dir" meant the unlabelled test/ directory.
    eval_dir = os.path.join(data_dir, "val")

    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_full = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
    eval_set   = torchvision.datasets.ImageFolder(eval_dir, transform=test_transform)
    # ImageFolder assigns label indices by sorted directory name independently per directory, so
    # a missing/extra class on either side would silently shift every label. Cheap to rule out.
    if train_full.classes != eval_set.classes:
        raise RuntimeError(
            f"train/ and val/ class lists differ ({len(train_full.classes)} vs "
            f"{len(eval_set.classes)}) -- labels would be wrong. Re-run "
            f"scripts/prepare_tiny_imagenet.py.")

    test_loader = DataLoader(eval_set, batch_size=batch_size, pin_memory=True, shuffle=False)

    if val:
        np.random.seed(0)
        split_permutation = list(np.random.permutation(len(train_full)))
        n_keep = int(len(train_full) * 0.9)
        train_set = Subset(train_full, split_permutation[:n_keep])
        # Same underlying directory, but eval-time transform (no crop/flip) on the held-out part.
        val_set   = Subset(torchvision.datasets.ImageFolder(train_dir, transform=test_transform),
                           split_permutation[n_keep:])
        train_loader = DataLoader(train_set, batch_size=batch_size, pin_memory=True, shuffle=True)
        val_loader   = DataLoader(val_set, batch_size=batch_size, pin_memory=True, shuffle=False)
        return train_loader, val_loader, test_loader
    else:
        train_loader = DataLoader(train_full, batch_size=batch_size, pin_memory=True, shuffle=True)
        val_loader   = None
        return train_loader, val_loader, test_loader
