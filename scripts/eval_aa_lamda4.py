import sys, os
sys.path.insert(0, "/mnt/d/research/FAT"); os.chdir("/mnt/d/research/FAT")
import torch, dataset as dataset_mod
from CIFAR10.models.resnet_z import ResNet18_z
from converter import Converter
from utils import evaluate_final_aa
class A: pass
args = A(); args.eps = 8/255.0; args.batch_size = 512
mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
_, _, tl = dataset_mod.CIFAR100(root="./data/CIFAR100", download=False, batch_size=512, val=False)
m = Converter(ResNet18_z(num_classes=100), mean, std)
m.load_state_dict(torch.load("CIFAR100/checkpoint/featdir_span_random_10step_wa/k350wa_lamda4_last.pkl", map_location="cpu"), strict=False)
m.cuda().eval()
print("AA_RESULT | k350+WA+lamda4 (s0) | aa_acc %.2f" % evaluate_final_aa(m, tl, args), flush=True)
