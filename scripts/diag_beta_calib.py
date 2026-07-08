"""Calibrate beta for decorrelated carve. Teacher-only.
need_rel = class_need / mean_dim(class_need); gate = exp(-beta*need_rel); w = exp(-tau*frag*gate).
Reports need_rel distribution, and per (tau,beta): carved teacher acc / softness / how much carve survives.
beta too big -> gate~0 everywhere -> w~1 -> reverts to RAW sharp teacher (acc high, entropy low, ~no carve).
"""
import sys, torch, numpy as np
import torch.nn.functional as F
sys.path.insert(0, "/mnt/d/research/FAT")
from CIFAR10.models.resnet import ResNet18
from converter import Converter
import torchvision, torchvision.transforms as T

mean=(0.5070751592371323,0.48654887331495095,0.4409178433670343)
std=(0.2673342858792401,0.2564384629170883,0.27615047132568404)
EPS,CS=8/255,2; CSTEP=EPS/CS; N,BS=2000,100
teacher=Converter(ResNet18(num_classes=100),mean,std).cuda()
teacher.load_state_dict(torch.load("/mnt/d/research/FAT/CIFAR100/checkpoint/clean/clean_last.pkl"))
teacher.eval(); W=teacher.encoder.linear.weight.detach()
ds=torchvision.datasets.CIFAR100("/mnt/d/research/FAT/data/CIFAR100",train=False,download=False,transform=T.ToTensor())
loader=torch.utils.data.DataLoader(torch.utils.data.Subset(ds,range(N)),batch_size=BS)
FR,NR,FT,Y=[],[],[],[]
for x,y in loader:
    x,y=x.cuda(),y.cuda()
    with torch.no_grad(): fc,_=teacher(x,feat=True)
    xa=x.clone().detach()
    for _ in range(CS):
        xa.requires_grad_(True); _,lo=teacher(xa,feat=True)
        g=torch.autograd.grad(F.cross_entropy(lo,y),xa)[0]
        xa=torch.min(torch.max(xa.detach()+CSTEP*g.sign(),x-EPS),x+EPS).clamp(0,1)
    with torch.no_grad(): fa,_=teacher(xa,feat=True)
    FR.append((fc-fa).abs().cpu()); NR.append((W[y]*fc).abs().cpu()); FT.append(fc.cpu()); Y.append(y.cpu())
frag=torch.cat(FR); need=torch.cat(NR); feat=torch.cat(FT).cuda(); y=torch.cat(Y).cuda()
need_rel=(need/(need.mean(1,keepdim=True)+1e-8))
p=np.percentile(need_rel.numpy(),[50,75,90,95,99])
print("need_rel percentiles  p50 %.2f  p75 %.2f  p90 %.2f  p95 %.2f  p99 %.2f"%tuple(p))
frag=frag.cuda(); need_rel=need_rel.cuda()
base=teacher.encoder.linear(feat); base_acc=(base.argmax(1)==y).float().mean().item()
print("raw teacher (no carve): acc %.2f  maxprob %.3f\n"%(base_acc*100,F.softmax(base,1).max(1).values.mean()))
for tau in [1.0,0.5]:
    print(f"--- tau={tau} ---")
    for beta in [0,0.05,0.1,0.15,0.2,0.3,0.5]:
        gate=torch.exp(-beta*need_rel); w=torch.exp(-tau*frag*gate)
        cl=teacher.encoder.linear(feat*w); pr=F.softmax(cl,1)
        acc=(cl.argmax(1)==y).float().mean().item()
        ent=(-(pr*(pr+1e-12).log()).sum(1)).mean().item()
        carved=(1-w).mean().item()   # avg feature suppression (0=no carve,1=all killed)
        print(f"  beta {beta:<4} acc {acc*100:5.2f}  maxprob {pr.max(1).values.mean():.3f}  entropy {ent:.3f}  avg_carve {carved:.3f}")
