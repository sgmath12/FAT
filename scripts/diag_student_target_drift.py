"""
CORRECTED framing (per user): the teacher is NEVER attacked in the real pipeline -- it only ever
sees clean x and emits target = teacher(x)/tau. The real question is whether the STUDENT (raw vs
normalized head, same backbone) can keep matching that FIXED clean target once ITS OWN input is
adversarially perturbed -- using the EXACT attack objective from training (inner_loss_only_return):
    x_adv = argmax_{||x_adv-x||<=eps} KL( student(x_adv) || target )   (student self-attacked to
            diverge from the FIXED clean teacher target, steps=3, matching train_temperature)
Compare raw-head vs normalized-head student (same clean-loaded weights) on:
    KL(student(x)     || target)   -- clean divergence from target (should be ~0, same net as teacher)
    KL(student(x_adv) || target)   -- divergence AFTER the attack succeeds at pulling it away
Full CIFAR100 test set, tau=16 (matches temp_studentNorm_teacherRaw.yaml).
"""
import sys, torch, torch.nn.functional as F
sys.path.insert(0, "/mnt/d/research/FAT")
import dataset
from converter import Converter
from CIFAR10.models.resnet import ResNet18
from CIFAR10.models.resnet_z import ResNet18_z

torch.manual_seed(0)
mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
std  = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
CKPT = "/mnt/d/research/FAT/CIFAR100/checkpoint/clean/clean_last.pkl"
EPS, STEP, STEPS, TAU = 8/255, 4/255, 3, 16.0

ck = torch.load(CKPT, map_location="cuda")
teacher = Converter(ResNet18(num_classes=100), mean, std).cuda(); teacher.load_state_dict(ck); teacher.eval()
raw     = Converter(ResNet18(num_classes=100), mean, std).cuda(); raw.load_state_dict(ck); raw.eval()
norm    = Converter(ResNet18_z(num_classes=100, scale=1.0), mean, std).cuda(); norm.load_state_dict(ck); norm.eval()

_, _, test_loader = dataset.CIFAR100(root="/mnt/d/research/FAT/data/CIFAR100", download=True, batch_size=250)
criterion_kl = torch.nn.KLDivLoss(reduction='none')

def attack_to_diverge(model, target, x):
    x_adv = x.detach() + 0.001 * torch.randn_like(x)
    for _ in range(STEPS):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss = criterion_kl(F.log_softmax(model(x_adv), dim=1), F.softmax(target, dim=1)).sum()
        grad = torch.autograd.grad(loss, [x_adv])[0]
        x_adv = x_adv.detach() + STEP * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x - EPS), x + EPS).clamp(0, 1)
    return x_adv.detach()

def run(model, name):
    kl_clean_all, kl_adv_all = [], []
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            target = (teacher(x) / TAU).detach()          # ALWAYS the clean raw-teacher target
        x_adv = attack_to_diverge(model, target, x)
        with torch.no_grad():
            kl_clean = criterion_kl(F.log_softmax(model(x), dim=1), F.softmax(target, dim=1)).sum(1)
            kl_adv   = criterion_kl(F.log_softmax(model(x_adv), dim=1), F.softmax(target, dim=1)).sum(1)
        kl_clean_all.append(kl_clean.cpu()); kl_adv_all.append(kl_adv.cpu())
    kc, ka = torch.cat(kl_clean_all), torch.cat(kl_adv_all)
    print(f"\n=== {name} ===")
    print(f"  KL(student(x)     || target): mean={kc.mean():.4f}")
    print(f"  KL(student(x_adv) || target): mean={ka.mean():.4f}  (this is exactly the training LOSS the inner-max attack achieves)")
    print(f"  drift opened by attack (adv - clean): {ka.mean()-kc.mean():.4f}")
    return kc, ka

kc_raw, ka_raw   = run(raw,  "RAW-head student (loaded teacher's weights, no normalize)")
kc_norm, ka_norm = run(norm, "NORMALIZED-head student (loaded teacher's weights, L2-normalize)")

print("\n=== COMPARISON ===")
print(f"post-attack KL ratio (norm/raw): {ka_norm.mean()/ka_raw.mean():.4f}")
