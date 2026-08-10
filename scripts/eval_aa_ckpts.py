# AA evaluation on archived ckpts (2026-07-15): the cw −0.30 (3/3) deficit of k350+WA vs
# baseline+WA needs the AA arbiter. Same convention as prior project AA numbers
# (evaluate_final_aa: apgd-ce + apgd-t, eps 8/255) -> comparable to baseline_10step AA 24.76.
# Seed-matched pair: baseline+WA(s2, folder last) vs k350+WA(s2, archived); plus eps10(s0), pure k350(s2).
import sys, os
sys.path.insert(0, "/mnt/d/research/FAT")
os.chdir("/mnt/d/research/FAT")
import torch
import dataset as dataset_mod
from CIFAR10.models.resnet_z import ResNet18_z
from converter import Converter
from utils import evaluate_final_aa

class A: pass
args = A(); args.eps = 8/255.0; args.batch_size = 512

mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
_, _, test_loader = dataset_mod.CIFAR100(root="./data/CIFAR100", download=False, batch_size=512, val=False)

CKPTS = [
    ("baseline+WA (s2)",  "CIFAR100/checkpoint/temp_baseline_10step_wa/temperature_last.pkl"),
    ("k350+WA (s2)",      "CIFAR100/checkpoint/featdir_span_random_10step_wa/k350wa_seed2_last.pkl"),
    ("k350+WA+eps10 (s0)","CIFAR100/checkpoint/featdir_k350wa_eps10/feat_direction_last.pkl"),
    ("k350 pure (s2)",    "CIFAR100/checkpoint/featdir_span_random_10step/k350_seed2_last.pkl"),
]
for name, path in CKPTS:
    m = Converter(ResNet18_z(num_classes=100), mean, std)
    r = m.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
    m.cuda().eval()
    aa = evaluate_final_aa(m, test_loader, args)
    print(f"AA_RESULT | {name} | {path} | aa_acc {aa:.2f}", flush=True)
print("AA_ALL_DONE")
