#!/usr/bin/env python
"""Download Tiny-ImageNet-200 and regroup its val split into ImageFolder layout.

The official archive ships val/ as a FLAT images/ directory plus a val_annotations.txt mapping
file, which torchvision's ImageFolder cannot read. train/ is already usable as shipped
(train/<wnid>/images/*.JPEG -- ImageFolder recurses into the images/ level).

    python scripts/prepare_tiny_imagenet.py [--root ./data]

Idempotent: re-running after a successful prepare is a no-op.

Note on speed: the archive is ~120k small files. On a Windows drive mounted through WSL
(/mnt/d, DrvFs) extraction takes tens of minutes -- it is file-creation bound, not bandwidth
bound. This is expected, not a hang.
"""
import argparse
import collections
import os
import shutil
import sys
import urllib.request
import zipfile

URL = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"
EXPECTED_BYTES = 248100043


def _download(zip_path):
    if os.path.exists(zip_path) and os.path.getsize(zip_path) == EXPECTED_BYTES:
        print(f"archive already present: {zip_path}")
        return
    print(f"downloading {URL}")

    def hook(blocks, bs, total):
        done = blocks * bs
        pct = 100.0 * done / total if total > 0 else 0
        sys.stdout.write(f"\r  {done / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.0f}%)")
        sys.stdout.flush()

    urllib.request.urlretrieve(URL, zip_path, reporthook=hook)
    print()
    size = os.path.getsize(zip_path)
    if size != EXPECTED_BYTES:
        raise RuntimeError(f"size mismatch: got {size}, expected {EXPECTED_BYTES}")


def _extract(zip_path, root, data_dir):
    if os.path.isdir(os.path.join(data_dir, "train")):
        print("already extracted")
        return
    print("extracting (120k small files -- slow on /mnt drives, expect tens of minutes)")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(root)


def _regroup_val(data_dir):
    """val/images/*.JPEG + val_annotations.txt  ->  val/<wnid>/*.JPEG"""
    val_dir = os.path.join(data_dir, "val")
    ann = os.path.join(val_dir, "val_annotations.txt")
    if not os.path.exists(ann):
        print("val already regrouped")
        return
    mapping = {}
    with open(ann) as f:
        for line in f:
            parts = line.split("\t")
            mapping[parts[0]] = parts[1]
    print(f"regrouping val: {len(mapping)} images, {len(set(mapping.values()))} classes")

    counts = collections.Counter()
    moved = 0
    for img, wnid in mapping.items():
        dest = os.path.join(val_dir, wnid)
        os.makedirs(dest, exist_ok=True)
        src = os.path.join(val_dir, "images", img)
        if os.path.exists(src):
            shutil.move(src, os.path.join(dest, img))
            moved += 1
        counts[wnid] += 1
    print(f"  moved {moved}, per-class min/max {min(counts.values())}/{max(counts.values())}")

    leftover_dir = os.path.join(val_dir, "images")
    if os.path.isdir(leftover_dir):
        leftover = os.listdir(leftover_dir)
        if leftover:
            raise RuntimeError(f"{len(leftover)} images left unmapped in {leftover_dir}")
        os.rmdir(leftover_dir)
    # Keep val_annotations.txt out of the way: ImageFolder ignores stray files, but its presence
    # is what this function keys off to decide whether regrouping already happened.
    os.rename(ann, ann + ".done")


def _verify(data_dir):
    import torchvision
    tr = torchvision.datasets.ImageFolder(os.path.join(data_dir, "train"))
    ev = torchvision.datasets.ImageFolder(os.path.join(data_dir, "val"))
    print(f"train/ {len(tr):>6} imgs / {len(tr.classes)} classes   -> training")
    print(f"val/   {len(ev):>6} imgs / {len(ev.classes)} classes   -> EVALUATION "
          f"(returned as test_loader; this is what ADR reports on)")
    print(f"test/  {'10000':>6} imgs / unlabelled          -> NOT USED (no public labels)")
    assert tr.classes == ev.classes, "train/val class order differs -- labels would be wrong"
    assert (len(tr), len(ev)) == (100000, 10000), f"unexpected counts {len(tr)}, {len(ev)}"
    print("OK: dataset.TinyImageNet() is ready to use")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data")
    args = ap.parse_args()

    os.makedirs(args.root, exist_ok=True)
    zip_path = os.path.join(args.root, "tiny-imagenet-200.zip")
    data_dir = os.path.join(args.root, "tiny-imagenet-200")

    _download(zip_path)
    _extract(zip_path, args.root, data_dir)
    _regroup_val(data_dir)
    _verify(data_dir)


if __name__ == "__main__":
    main()
