# Head-rotation diagnostic (user, 2026-07-13): before building the gain-only head
# (w_s,c = g_c * w_hat_t,c, direction FROZEN at the teacher's), measure how much the
# baseline-trained student head ACTUALLY rotated away from the teacher head it was
# initialized from (finetune init => w_s == w_t exactly at step 0).
#
# Decomposition per class c:   w_s,c = g_c * w_hat_t,c + r_c,   r_c ⊥ w_t,c
#   g_c   = <w_s,c, w_hat_t,c>          (gain = component along teacher direction)
#   cos_c = cos(w_s,c, w_t,c)           (rotation; residual share = sqrt(1-cos^2))
# Prediction rule: cos_c high (~>0.9) across classes  -> gain-only head should TIE
#                  cos_c low / difficulty-structured   -> direction learning is load-bearing.
# Extra: is the rotation residual inside span(W_t) (100-dim) or in the orthogonal
# 412-dim complement? In-span => a "teacher-span head" would still capture it.
# Controls: no_init run (head never tied to teacher; its cos = no-shared-basis null).
import os, sys
sys.path.insert(0, "/mnt/d/research/FAT")
os.chdir("/mnt/d/research/FAT")
import torch
import torch.nn.functional as F
import numpy as np

W_KEY, B_KEY = "encoder.linear.weight", "encoder.linear.bias"

def head(path):
    sd = torch.load(path, map_location="cpu")
    return sd[W_KEY].double(), sd[B_KEY].double()

Wt, bt = head("CIFAR100/checkpoint/clean/clean_last.pkl")
What = F.normalize(Wt, dim=1)
# orthonormal basis of span(W_t) for the in-span/out-of-span split
Q, _ = torch.linalg.qr(Wt.T)          # 512 x 100

stats = np.load("results/CIFAR100/diag_perclass_teacher.npz")

def rankz(v):
    r = np.empty_like(v); r[np.argsort(v)] = np.arange(len(v), dtype=np.float64)
    return (r - r.mean()) / (r.std() + 1e-12)

def spearman(a, b):
    return float((rankz(np.asarray(a, np.float64)) * rankz(np.asarray(b, np.float64))).mean())

def report(name, path):
    Ws, bs = head(path)
    g = (Ws * What).sum(1)                       # gain along teacher direction
    cos = F.cosine_similarity(Ws, Wt, dim=1)
    ns, nt = Ws.norm(dim=1), Wt.norm(dim=1)
    resid = Ws - g.unsqueeze(1) * What           # ⊥ w_t,c per class
    rshare = resid.norm(dim=1) / ns.clamp_min(1e-12)
    in_span = (resid @ Q).norm(dim=1) / resid.norm(dim=1).clamp_min(1e-12)

    q = lambda v: [round(float(x), 4) for x in torch.quantile(v, torch.tensor([0.05, 0.5, 0.95], dtype=v.dtype))]
    print(f"\n=== {name} ===")
    print(f"cos(w_s,c, w_t,c)     mean {cos.mean():.4f}  p5/50/95 {q(cos)}  min {cos.min():.4f}")
    print(f"residual share        mean {rshare.mean():.4f}  p5/50/95 {q(rshare)}")
    print(f"resid in span(W_t)    mean frac {in_span.mean():.4f}  p5/50/95 {q(in_span)}")
    print(f"||w_s,c||             mean {ns.mean():.4f}  spread max/min {float(ns.max()/ns.min()):.3f}x   (teacher {nt.mean():.4f}, {float(nt.max()/nt.min()):.3f}x)")
    print(f"gain g_c              mean {g.mean():.4f}  spread max/min {float(g.max()/g.min()):.3f}x  (g/||w_s|| mean {float((g/ns).mean()):.4f})")
    print(f"bias                  ||b_s-b_t|| {float((bs-bt).norm()):.4f}  std(b_s) {float(bs.std()):.4f} (teacher {float(bt.std()):.4f})")
    gl, cn = np.log(np.clip(g.numpy(), 1e-9, None)), cos.numpy()
    for sn in ["gnorm", "acc", "entropy", "margin"]:
        if sn in stats.files:
            print(f"  spearman vs {sn:8s}: log g_c {spearman(gl, stats[sn]):+.3f}   cos_c {spearman(cn, stats[sn]):+.3f}")

report("baseline 3-step (temp_studentNorm_teacherRaw, last)", "CIFAR100/checkpoint/temp_studentNorm_teacherRaw/temperature_last.pkl")
report("baseline 10-step (temp_studentNorm_teacherRaw_10step, last)", "CIFAR100/checkpoint/temp_studentNorm_teacherRaw_10step/temperature_last.pkl")
report("CONTROL no_init (head never teacher-tied; no-shared-basis null)", "CIFAR100/checkpoint/temp_studentNorm_teacherRaw_no_init/temperature_last.pkl")

# --- Procrustes follow-up (same session): is the rotation a SINGLE shared feature-space
# rotation (backbone drift bookkeeping -> gain-only w/ free backbone can compensate) or
# per-class idiosyncratic (true direction learning -> gain-only should LOSE)?
# Fit orthogonal R (512x512) minimizing ||W_s - W_t R||_F, re-measure per-class cos.
def procrustes(name, path):
    Ws, _ = head(path)
    U, S, Vh = torch.linalg.svd(Wt.T @ Ws)
    R = U @ Vh
    WtR = Wt @ R
    cos0 = F.cosine_similarity(Ws, Wt, dim=1)
    cosR = F.cosine_similarity(Ws, WtR, dim=1)
    gR = (Ws * F.normalize(WtR, dim=1)).sum(1)
    q = lambda v: [round(float(x), 4) for x in torch.quantile(v, torch.tensor([0.05, 0.5, 0.95], dtype=v.dtype))]
    print(f"\n=== Procrustes {name} ===")
    print(f"cos raw  mean {cos0.mean():.4f} -> cos after shared-R alignment  mean {cosR.mean():.4f}  p5/50/95 {q(cosR)}  min {cosR.min():.4f}")
    print(f"residual share after R: mean {float((1-cosR**2).clamp_min(0).sqrt().mean()):.4f}")
    print(f"gain-after-R spread max/min {float(gR.max()/gR.min()):.3f}x ; spearman(log gR, gnorm) {spearman(np.log(np.clip(gR.numpy(),1e-9,None)), stats['gnorm']):+.3f}")

procrustes("baseline 3-step", "CIFAR100/checkpoint/temp_studentNorm_teacherRaw/temperature_last.pkl")
procrustes("baseline 10-step", "CIFAR100/checkpoint/temp_studentNorm_teacherRaw_10step/temperature_last.pkl")
procrustes("CONTROL no_init", "CIFAR100/checkpoint/temp_studentNorm_teacherRaw_no_init/temperature_last.pkl")
