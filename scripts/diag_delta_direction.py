"""Is the pilot's learned delta a GLOBAL direction or per-sample structure? (2026-07-06)

Pilot symptom: ||delta|| marched to the cap with near-zero dispersion (p5 0.283 / p50 0.297) --
uniform magnitude for every sample smells like one systematic logit-space push, not per-sample
triage. This dumps delta over the full CIFAR100 TEST set (teacher-wrong rate ~24% there vs 0.3%
on train -> far better probe statistics) and measures:
  1. alignment ratio ||mean(d_hat)|| (1.0 = a single global direction, ~0 = idiosyncratic)
  2. cos(d_i, mean_dir) distribution
  3. WHAT the mean direction does: top prob-mass gainer/loser classes on the average target
  4. dtrue on teacher-wrong vs teacher-right samples (swap-rediscovery, well-powered here)
  5. per-sample residual (after removing the global component): does ITS norm correlate with
     teacher entropy / correctness -- i.e. is there any second-order per-sample signal at all?
Only needs the teacher ckpt + the pilot ckpt's delta_net.* keys (delta path never sees the student).
"""
import sys, torch, torch.nn.functional as F
sys.path.insert(0, "/mnt/d/research/FAT")
import dataset
from converter import Converter
from CIFAR10.models.resnet import ResNet18
from utils import DeltaNet

torch.manual_seed(0)
mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
std  = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
TAU, DELTA_R = 16.0, 0.3
TEACHER_CKPT = "/mnt/d/research/FAT/CIFAR100/checkpoint/clean/clean_last.pkl"
PILOT_CKPT   = "/mnt/d/research/FAT/CIFAR100/checkpoint/temp_deltanet_bilevel/temperature_deltanet_bilevel_last.pkl"

teacher = Converter(ResNet18(num_classes=100), mean, std).cuda()
teacher.load_state_dict(torch.load(TEACHER_CKPT, map_location="cuda")); teacher.eval()

pk = torch.load(PILOT_CKPT, map_location="cuda")
dstate = {k[len("delta_net."):]: v for k, v in pk.items() if k.startswith("delta_net.")}
assert dstate, "no delta_net keys in pilot ckpt"
dnet = DeltaNet(100, hidden=128, depth=1, use_bn=False).cuda()
dnet.load_state_dict(dstate); dnet.eval()

_, _, test_loader = dataset.CIFAR100(root="/mnt/d/research/FAT/data/CIFAR100", download=False, batch_size=500)

D, T0, Y, WRONG = [], [], [], []
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()
        _, z = teacher(x, feat=True)
        t0 = z / TAU
        d = dnet(t0)
        d = d - d.mean(dim=1, keepdim=True)
        dn = d.norm(dim=1, keepdim=True)
        d = d * (DELTA_R / dn.clamp_min(1e-12)).clamp(max=1.0)
        D.append(d.cpu()); T0.append(t0.cpu()); Y.append(y.cpu())
        WRONG.append((z.argmax(dim=1) != y).cpu())
D = torch.cat(D); T0 = torch.cat(T0); Y = torch.cat(Y); WRONG = torch.cat(WRONG)
N = D.shape[0]

# 1-2. global alignment
d_hat = D / D.norm(dim=1, keepdim=True).clamp_min(1e-12)
mean_dhat = d_hat.mean(dim=0)
align = mean_dhat.norm().item()
mean_dir = mean_dhat / mean_dhat.norm()
cos = d_hat @ mean_dir
q = torch.quantile(cos, torch.tensor([0.05, 0.5, 0.95]))
print(f"[1] alignment ratio ||mean(d_hat)|| = {align:.4f}  (1.0 = one global direction)")
print(f"[2] cos(d_i, mean_dir): mean {cos.mean():.4f}  p5 {q[0]:.4f}  p50 {q[1]:.4f}  p95 {q[2]:.4f}")

# 3. what the global direction does to the average target's prob mass
p0 = F.softmax(T0, dim=1)
p1 = F.softmax(T0 + D, dim=1)
dmass = (p1 - p0).mean(dim=0)
top = torch.argsort(dmass, descending=True)
print(f"[3] avg prob-mass shift: top gainers {[(int(i), round(dmass[i].item()*1e4,2)) for i in top[:5]]} (x1e-4)")
print(f"    top losers  {[(int(i), round(dmass[i].item()*1e4,2)) for i in top[-5:]]} (x1e-4)")
print(f"    total |mass moved| per sample: {(p1-p0).abs().sum(dim=1).mean().item():.5f}")

# 4. swap-rediscovery on the test set (well-powered: wrong_frac ~24%)
d_true = D.gather(1, Y.unsqueeze(1)).squeeze(1)
print(f"[4] wrong_frac {WRONG.float().mean():.4f} | dtrue_wrong {d_true[WRONG].mean():.5f} "
      f"| dtrue_right {d_true[~WRONG].mean():.5f}")
# also in PROB mass on the true class
pm_true0 = p0.gather(1, Y.unsqueeze(1)).squeeze(1); pm_true1 = p1.gather(1, Y.unsqueeze(1)).squeeze(1)
dm_true = pm_true1 - pm_true0
print(f"    true-class mass shift: wrong {dm_true[WRONG].mean()*1e4:.3f}e-4 | right {dm_true[~WRONG].mean()*1e4:.3f}e-4")

# 5. per-sample residual after removing the global component
resid = D - (D @ mean_dir).unsqueeze(1) * mean_dir
rn = resid.norm(dim=1)
H = -(p0 * p0.clamp_min(1e-12).log()).sum(dim=1)
def corr(a, b):
    a = (a - a.mean()) / a.std().clamp_min(1e-12); b = (b - b.mean()) / b.std().clamp_min(1e-12)
    return (a * b).mean().item()
print(f"[5] residual norm: mean {rn.mean():.4f} (vs total {D.norm(dim=1).mean():.4f}) "
      f"| corr(resid_norm, teacher_entropy) {corr(rn, H):.3f} "
      f"| resid_norm wrong {rn[WRONG].mean():.4f} vs right {rn[~WRONG].mean():.4f}")
