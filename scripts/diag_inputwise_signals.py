"""Is there an input-wise signal ORTHOGONAL to confidence (what global temperature already exploits)?
If yes -> input-wise carving/softening can genuinely beat global temp. If everything collapses onto the
confidence axis (Spearman ~0.9) -> redundant, skip. Teacher-only, light (runs alongside training)."""
import sys, torch
import torch.nn.functional as F
sys.path.insert(0, "/mnt/d/research/FAT")
from CIFAR10.models.resnet import ResNet18
from converter import Converter
import torchvision, torchvision.transforms as T
import numpy as np
from scipy.stats import spearmanr

mean=(0.5070751592371323,0.48654887331495095,0.4409178433670343); std=(0.2673342858792401,0.2564384629170883,0.27615047132568404)
EPS=8/255; N,BS=1000,100
teacher=Converter(ResNet18(num_classes=100),mean,std).cuda()
teacher.load_state_dict(torch.load("/mnt/d/research/FAT/CIFAR100/checkpoint/clean/clean_last.pkl")); teacher.eval()
ds=torchvision.datasets.CIFAR100("/mnt/d/research/FAT/data/CIFAR100",train=False,download=False,transform=T.ToTensor())
loader=torch.utils.data.DataLoader(torch.utils.data.Subset(ds,range(N)),batch_size=BS)

conf=[]; ent=[]; margin=[]; vuln=[]          # per-sample
dim_ball_var=0; dim_data=[]; nb=0             # per-dim
for x,y in loader:
    x,y=x.cuda(),y.cuda()
    with torch.no_grad():
        feat,z=teacher.encoder.extract_feature(x) if hasattr(teacher,'encoder') else (None,teacher(x))
        z=teacher(x)
        p=F.softmax(z,1)
        conf+=p.max(1).values.tolist()
        ent+=(-(p*(p+1e-12).log()).sum(1)).tolist()
        top2=z.topk(2,1).values; margin+=(top2[:,0]-top2[:,1]).tolist()
        # per-dim feature (raw 512-d) via feat=True path
        f0,_=teacher(x,feat=True) if False else (None,None)
    # feature + per-dim ball variance (adversarial 2-step, random-start avg over K)
    K=6; feats=[]
    for _ in range(K):
        xa=(x+ (torch.rand_like(x)*2-1)*EPS).clamp(0,1).detach()
        for _ in range(2):
            xa.requires_grad_(True)
            g=torch.autograd.grad(F.cross_entropy(teacher(xa),y),xa)[0]
            xa=(xa.detach()+(EPS/2)*g.sign()).clamp(x-EPS,x+EPS).clamp(0,1)
        with torch.no_grad():
            fa,_=teacher(xa,feat=True)     # raw feature (pre-norm)
        feats.append(fa)
    with torch.no_grad():
        fclean,zc=teacher(x,feat=True)
        fstack=torch.stack(feats,0)                 # K x B x D
        # per-sample vulnerability = mean logit shift under attack
        za=teacher(xa); vuln+=(F.softmax(zc,1)*((F.softmax(zc,1)+1e-12).log()-(F.softmax(za,1)+1e-12).log())).sum(1).tolist()
        # per-dim ball variance (variance across the K adversarial samples, averaged over batch)
        dim_ball_var = dim_ball_var + fstack.var(0).mean(0)   # D
        dim_data.append(fclean)                               # for dataset variance
    nb+=1

conf=np.array(conf); ent=np.array(ent); margin=np.array(margin); vuln=np.array(vuln)
print("="*60); print("PER-SAMPLE: orthogonal to confidence? (Spearman |rho|)")
print("="*60)
for name,s in [("entropy",ent),("margin",margin),("vulnerability",vuln)]:
    r=abs(spearmanr(conf,s).correlation)
    print(f"  corr(confidence, {name:13s}) = {r:.3f}   {'<-- REDUNDANT (~confidence)' if r>0.8 else '<-- possibly orthogonal' if r<0.5 else ''}")
print(f"  corr(entropy, vulnerability)      = {abs(spearmanr(ent,vuln).correlation):.3f}")

dim_ball_var=(dim_ball_var/nb).cpu().numpy()
data_var=torch.cat(dim_data,0).var(0).cpu().numpy()
print(); print("="*60); print("PER-DIM (512 feature dims): is variance-carve a NEW signal?")
print("="*60)
print(f"  ball-variance dispersion (std/mean over dims) = {dim_ball_var.std()/dim_ball_var.mean():.3f}   {'(flat=no lever)' if dim_ball_var.std()/dim_ball_var.mean()<0.3 else '(has spread)'}")
print(f"  corr(ball-variance, dataset-variance) = {abs(spearmanr(dim_ball_var,data_var).correlation):.3f}  {'<-- ball-var ~ just high-activation dims' if abs(spearmanr(dim_ball_var,data_var).correlation)>0.8 else ''}")
