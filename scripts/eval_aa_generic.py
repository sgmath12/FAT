# Generic AA evaluator (2026-07-16): pass "label|path" pairs via argv, prints AA_RESULT lines.
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

for pair in sys.argv[1:]:
    label, path = pair.split("|", 1)
    m = Converter(ResNet18_z(num_classes=100), mean, std)
    m.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
    m.cuda().eval()
    aa = evaluate_final_aa(m, test_loader, args)
    print(f"AA_RESULT | {label} | {path} | aa_acc {aa:.2f}", flush=True)
