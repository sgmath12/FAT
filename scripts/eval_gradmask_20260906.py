# GRADIENT-MASKING CHECKS ON THE SHIPPED MODEL (2026-09-06).
#
# Every AutoAttack number in this paper is the `standard` version restricted to apgd-ce and apgd-t
# (utils.py:856).  Both are white-box gradient attacks, and the objective we train has no labels and
# no logits, so "is this gradient masking?" is the first question a reviewer asks.  This script runs
# the checks Athalye et al. list, on the shipped CIFAR-100 cell:
#
#   1. black box vs white box   Square (5000 queries) must NOT beat apgd-ce + apgd-t
#   2. FAB-T                    the other white-box attack of the full AA suite
#   3. radius sweep             accuracy must fall monotonically in eps and reach ~0 well above 8/255
#   4. step sweep               PGD-10/20/50/100 must be monotone and flatten, not oscillate
#   5. transfer from the teacher  the student inherits the teacher's classifier, so a reviewer will
#                               ask whether examples built on the teacher carry over
#
# Attacks 1-2 run on the first 1000 test examples (AA convention for the expensive attacks); the rest
# on the full test set.
import sys, os, time
sys.path.insert(0, "/mnt/d/research/FAT"); os.chdir("/mnt/d/research/FAT")
import torch, torch.nn as nn, torch.nn.functional as F
import dataset as dataset_mod
from CIFAR10.models.resnet import ResNet18
from converter import Converter
from autoattack import AutoAttack

MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
STD  = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
STUDENT = "CIFAR100/checkpoint/l2_bestrecipe_freezehead/feat_direction_last.pkl"
TEACHER = "CIFAR100/checkpoint/clean_200ep/clean_last.pkl"

def build(path):
    m = Converter(ResNet18(num_classes=100), MEAN, STD)
    sd = torch.load(path, map_location="cpu")
    m.load_state_dict(sd, strict=False)
    return m.cuda().eval()

def accuracy(model, x, y, bs=512):
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x), bs):
            correct += (model(x[i:i+bs].cuda()).argmax(1) == y[i:i+bs].cuda()).sum().item()
    return 100.0 * correct / len(x)

def pgd(model, x, y, eps, steps, alpha=None, bs=256):
    alpha = alpha or max(eps / 4, 1/255)
    out = []
    for i in range(0, len(x), bs):
        xb, yb = x[i:i+bs].cuda(), y[i:i+bs].cuda()
        d = (torch.rand_like(xb) * 2 - 1) * eps
        d = (xb + d).clamp(0, 1) - xb
        for _ in range(steps):
            d.requires_grad_(True)
            loss = F.cross_entropy(model(xb + d), yb)
            g, = torch.autograd.grad(loss, d)
            d = (d.detach() + alpha * g.sign()).clamp(-eps, eps)
            d = (xb + d).clamp(0, 1) - xb
        with torch.no_grad():
            out.append((model(xb + d).argmax(1) == yb).float().cpu())
    return 100.0 * torch.cat(out).mean().item()

_, _, test_loader = dataset_mod.CIFAR100(root="./data/CIFAR100", download=False, batch_size=512, val=False)
x_test = torch.cat([x for x, _ in test_loader], 0)
y_test = torch.cat([y for _, y in test_loader], 0)
student, teacher = build(STUDENT), build(TEACHER)
print(f"GM | clean | student {accuracy(student, x_test, y_test):.2f} | teacher {accuracy(teacher, x_test, y_test):.2f}", flush=True)

# 1-2. the two attacks the paper has never run, on the first 1000 examples
xs, ys = x_test[:1000], y_test[:1000]
for name in ["apgd-ce", "apgd-t", "fab-t", "square"]:
    a = AutoAttack(student, norm="Linf", eps=8/255, version="standard")
    a.attacks_to_run = [name]
    t0 = time.time()
    adv = a.run_standard_evaluation(xs, ys, bs=256)
    acc = accuracy(student, adv.cpu(), ys)
    print(f"GM | attack {name:8s} | n 1000 | acc {acc:.2f} | {time.time()-t0:.0f}s", flush=True)

# 3. radius sweep, full test set
for e in [2, 4, 8, 12, 16, 32, 64]:
    print(f"GM | pgd20 eps {e:2d}/255 | acc {pgd(student, x_test, y_test, e/255, 20):.2f}", flush=True)

# 4. step sweep at 8/255
for s in [10, 20, 50, 100]:
    print(f"GM | pgd steps {s:3d} | acc {pgd(student, x_test, y_test, 8/255, s):.2f}", flush=True)

# 5. transfer from the natural teacher: build on the teacher, evaluate on the student
bs, correct = 256, 0
for i in range(0, len(x_test), bs):
    xb, yb = x_test[i:i+bs].cuda(), y_test[i:i+bs].cuda()
    d = torch.zeros_like(xb)
    for _ in range(20):
        d.requires_grad_(True)
        g, = torch.autograd.grad(F.cross_entropy(teacher(xb + d), yb), d)
        d = (d.detach() + (2/255) * g.sign()).clamp(-8/255, 8/255)
        d = (xb + d).clamp(0, 1) - xb
    with torch.no_grad():
        correct += (student(xb + d).argmax(1) == yb).sum().item()
print(f"GM | transfer teacher->student pgd20 | acc {100.0*correct/len(x_test):.2f}", flush=True)
