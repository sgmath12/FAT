# Mechanism battery (1)(2)(3) (2026-07-15): eval-only, on archived 10-step pure-k350 ckpts.
# (1) SUBSPACE ABLATION: at inference, feed the head only the span(Q) component / only the
#     complement component of the normalized feature -> where does clean/robust acc live?
# (2) PER-SUBSPACE LINEAR PROBE: train a linear classifier on span-projected vs complement
#     features (clean & adv) -> does the FREE subspace carry robust discriminative info the
#     teacher doesn't have? (teacher-feature probes as reference)
# (3) ATTACK DISPLACEMENT: how much does PGD move the feature direction inside span(Q) vs the
#     complement? (share of ||Q^T(f_adv - f_clean)||^2 in the total)
# Q is reproduced bit-exactly (randn(512,k,gen(seed0)) float32 -> double QR -> float32),
# matching train_feat_direction's _Q_cache construction.
import sys, os
sys.path.insert(0, "/mnt/d/research/FAT")
os.chdir("/mnt/d/research/FAT")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchattacks
import dataset as dataset_mod
from CIFAR10.models.resnet_z import ResNet18_z
from converter import Converter

K = 350
mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
_, _, test_loader = dataset_mod.CIFAR100(root="./data/CIFAR100", download=False, batch_size=512, val=False)

g = torch.Generator().manual_seed(0)
base = torch.randn(512, K, generator=g)
Qm, _ = torch.linalg.qr(base.double())
Q = Qm.float().cuda()                      # 512 x 350, orthonormal

def load(path):
    m = Converter(ResNet18_z(num_classes=100), mean, std)
    m.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
    return m.cuda().eval()

class Ablated(nn.Module):
    """Full model but the head sees only the span / complement component of Phi_hat."""
    def __init__(self, conv_model, mode):
        super().__init__()
        self.m = conv_model
        self.mode = mode
    def forward(self, x):
        feat, _ = self.m(x, feat=True)
        f = F.normalize(feat, dim=1)
        if self.mode == "span":
            f = (f @ Q) @ Q.T
        elif self.mode == "comp":
            f = f - (f @ Q) @ Q.T
        enc = self.m.encoder
        return enc.head_from_feat(enc.scale * f)

def acc_clean_pgd(model):
    atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=20, random_start=True)
    c = r = n = 0
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            c += (model(x).argmax(1) == y).sum().item()
        xa = atk(x, y)
        with torch.no_grad():
            r += (model(xa).argmax(1) == y).sum().item()
        n += y.numel()
    return 100*c/n, 100*r/n

def collect_feats(model):
    """clean & adv (PGD20 on the FULL model) normalized features + labels."""
    atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=20, random_start=True)
    Fc, Fa, Y = [], [], []
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()
        xa = atk(x, y)
        with torch.no_grad():
            fc, _ = model(x, feat=True)
            fa, _ = model(xa, feat=True)
        Fc.append(F.normalize(fc, dim=1)); Fa.append(F.normalize(fa, dim=1)); Y.append(y)
    return torch.cat(Fc), torch.cat(Fa), torch.cat(Y)

def probe(feats_tr, y_tr, feats_te, y_te, dim):
    head = nn.Linear(dim, 100).cuda()
    opt = torch.optim.Adam(head.parameters(), lr=1e-2, weight_decay=0)
    for ep in range(60):
        idx = torch.randperm(feats_tr.shape[0], device="cuda")
        for i in range(0, len(idx), 2048):
            b = idx[i:i+2048]
            loss = F.cross_entropy(head(feats_tr[b]), y_tr[b])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return 100 * (head(feats_te).argmax(1) == y_te).float().mean().item()

def run_battery(name, path):
    m = load(path)
    print(f"\n===== {name} ({path}) =====", flush=True)
    # (1) ablation
    for mode in ["full", "span", "comp"]:
        am = Ablated(m, mode).eval()
        c, r = acc_clean_pgd(am)
        print(f"[1 ablation] head sees {mode:4s}: clean {c:.2f}  pgd20 {r:.2f}", flush=True)
    # (2)+(3) features
    Fc, Fa, Y = collect_feats(m)
    d = Fa - Fc
    span_share = (d @ Q).pow(2).sum(1) / d.pow(2).sum(1).clamp_min(1e-12)
    print(f"[3 displacement] PGD feature shift: span share mean {span_share.mean():.3f} "
          f"(chance {K/512:.3f}); total ||d|| mean {d.norm(dim=1).mean():.3f}", flush=True)
    ntr = 8000
    for tag, proj, dim in [("span", lambda f: f @ Q, K), ("comp", lambda f: f - (f @ Q) @ Q.T, 512)]:
        for ftag, FT in [("clean", Fc), ("adv", Fa)]:
            a = probe(proj(FT[:ntr]), Y[:ntr], proj(FT[ntr:]), Y[ntr:], dim)
            print(f"[2 probe] {tag:4s} features, {ftag:5s}: acc {a:.2f}", flush=True)

run_battery("k350 pure s2 (student)", "CIFAR100/checkpoint/featdir_span_random_10step/k350_seed2_last.pkl")
run_battery("k512 featdir 3step s0 (no-free-space control)", "CIFAR100/checkpoint/featdir/feat_direction_last.pkl")
run_battery("TEACHER (natural, reference)", "CIFAR100/checkpoint/clean/clean_last.pkl")
print("BATTERY_DONE", flush=True)
