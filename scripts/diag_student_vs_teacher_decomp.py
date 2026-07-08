"""
Diagnostic (no training): decompose Delta-logit under PGD attack into MAGNITUDE-channel vs
ROTATION-channel, comparing RAW (teacher, no norm) vs NORMALIZED (student, L2-norm head) --
using the EXACT SAME backbone weights (student loads the teacher's clean checkpoint before AT).
Full CIFAR100 test set (10000 images).

For each variant, PGD-attack ITS OWN prediction (standard robustness attack, eps 8/255, steps 20),
then for the raw penultimate feature Phi(x) -> Phi(x_adv):
    s      = ||Phi_adv|| / ||Phi_clean||                         (norm ratio actually observed)
    z_mag  = classifier_head( magnitude-only counterfactual feature )
             raw:  head(s * Phi_clean)
             norm: head(scale * Phi_clean/||Phi_clean||)  == z_clean EXACTLY (norm change is a
                   null op once normalized -- this is the structural point, not a bug)
    rot    = || z_adv - z_mag ||
    r      = rot / || z_adv - z_clean ||         (rotation SHARE)
Also reports || z_adv - z_clean || (TOTAL vulnerability magnitude) -- the more decisive number:
does normalization shrink the attack's overall leverage, even though by construction 100% of
whatever remains is "rotation" for the normalized variant.
"""
import sys, torch, torchattacks
sys.path.insert(0, "/mnt/d/research/FAT")
import dataset
from converter import Converter
from CIFAR10.models.resnet import ResNet18
from CIFAR10.models.resnet_z import ResNet18_z

torch.manual_seed(0)
mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
std  = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
CKPT = "/mnt/d/research/FAT/CIFAR100/checkpoint/clean/clean_last.pkl"
EPS = 8/255

ck = torch.load(CKPT, map_location="cuda")

raw = Converter(ResNet18(num_classes=100), mean, std).cuda()
raw.load_state_dict(ck); raw.eval()

norm = Converter(ResNet18_z(num_classes=100, scale=1.0), mean, std).cuda()
norm.load_state_dict(ck); norm.eval()

_, _, test_loader = dataset.CIFAR100(root="/mnt/d/research/FAT/data/CIFAR100", download=True, batch_size=250)

def run(model, name):
    attack = torchattacks.PGD(model, eps=EPS, alpha=2/255, steps=20, random_start=True)
    r_all, dz_all, correct_clean, correct_adv, n = [], [], 0, 0, 0
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()
        x_adv = attack(x, y)
        with torch.no_grad():
            phi_c, z_c = model.extract_feature(x)
            phi_a, z_a = model.extract_feature(x_adv)
            s = (phi_a.norm(dim=1) / phi_c.norm(dim=1).clamp_min(1e-8)).unsqueeze(1)
            z_mag = model.linear(s * phi_c)
            rot = (z_a - z_mag).norm(dim=1)
            dz = (z_a - z_c).norm(dim=1)
            r = rot / dz.clamp_min(1e-8)
            r_all.append(r.cpu()); dz_all.append(dz.cpu())
            correct_clean += (z_c.argmax(1) == y).sum().item()
            correct_adv   += (z_a.argmax(1) == y).sum().item()
            n += x.size(0)
    r_cat, dz_cat = torch.cat(r_all), torch.cat(dz_all)
    q = lambda t, p: torch.quantile(t, torch.tensor(p)).tolist()
    print(f"\n=== {name} (n={n}) ===")
    print(f"  clean_acc={100*correct_clean/n:.2f}  pgd20_acc={100*correct_adv/n:.2f}")
    print(f"  rot_share:  mean={r_cat.mean():.4f}  p5/p50/p95={q(r_cat,[.05,.5,.95])}")
    print(f"  ||dz_total||: mean={dz_cat.mean():.4f}  p5/p50/p95={q(dz_cat,[.05,.5,.95])}")
    return r_cat, dz_cat

r_raw, dz_raw = run(raw, "RAW (teacher, no norm)")
r_norm, dz_norm = run(norm, "NORMALIZED (student head, same backbone)")

print("\n=== COMPARISON ===")
print(f"total-vulnerability ratio (norm/raw), mean ||dz||: {dz_norm.mean()/dz_raw.mean():.4f}")
print(f"median ratio: {dz_norm.median()/dz_raw.median():.4f}")
