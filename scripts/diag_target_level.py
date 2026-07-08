"""Target-level redundancy test (the RIGHT question): when you per-sample soften with each signal, do the
resulting TARGETS actually differ? (signal rank-correlation was trivial.) Build softened targets and compare
by KL at the distribution level. T_s(x) = T0*exp(gamma*zscore_batch(signal)); global g = softmax(z/T0)."""
import sys, torch
import torch.nn.functional as F
sys.path.insert(0,"/mnt/d/research/FAT")
from CIFAR10.models.resnet import ResNet18
from converter import Converter
import torchvision, torchvision.transforms as T
import numpy as np

mean=(0.5070751592371323,0.48654887331495095,0.4409178433670343); std=(0.2673342858792401,0.2564384629170883,0.27615047132568404)
EPS=8/255; N,BS=1000,100; T0=16.0; GAMMA=0.3
teacher=Converter(ResNet18(num_classes=100),mean,std).cuda()
teacher.load_state_dict(torch.load("/mnt/d/research/FAT/CIFAR100/checkpoint/clean/clean_last.pkl")); teacher.eval()
ds=torchvision.datasets.CIFAR100("/mnt/d/research/FAT/data/CIFAR100",train=False,download=False,transform=T.ToTensor())
loader=torch.utils.data.DataLoader(torch.utils.data.Subset(ds,range(N)),batch_size=BS)

def zc(s): return ((s-s.mean())/(s.std()+1e-6)).clamp(-2,2)
def temp_target(z,s): return F.softmax(z/(T0*torch.exp(GAMMA*zc(s))).reshape(-1,1),1)
def KL(p,q): return (p*((p+1e-12).log()-(q+1e-12).log())).sum(1)   # per-sample

acc={k:[] for k in ["g|ent","g|mar","g|vul","ent|mar","ent|vul","mar|vul"]}
for x,y in loader:
    x,y=x.cuda(),y.cuda()
    with torch.no_grad():
        z=teacher(x); p=F.softmax(z,1)
        conf=p.max(1).values                       # confidence signal
        ent=-(p*(p+1e-12).log()).sum(1)            # entropy
        t2=z.topk(2,1).values; mar=t2[:,0]-t2[:,1] # margin
    # vulnerability (2-step teacher PGD, KL clean||adv)
    xa=(x+(torch.rand_like(x)*2-1)*EPS).clamp(0,1).detach()
    for _ in range(2):
        xa.requires_grad_(True)
        g=torch.autograd.grad(F.cross_entropy(teacher(xa),y),xa)[0]
        xa=torch.min(torch.max(xa.detach()+(EPS/2)*g.sign(),x-EPS),x+EPS).clamp(0,1)
    with torch.no_grad():
        pa=F.softmax(teacher(xa),1); vul=(p*((p+1e-12).log()-(pa+1e-12).log())).sum(1)
        g_tgt=F.softmax(z/T0,1)
        te=temp_target(z,ent); tm=temp_target(z,mar); tv=temp_target(z,vul)
        acc["g|ent"]+=KL(g_tgt,te).tolist(); acc["g|mar"]+=KL(g_tgt,tm).tolist(); acc["g|vul"]+=KL(g_tgt,tv).tolist()
        acc["ent|mar"]+=KL(te,tm).tolist();  acc["ent|vul"]+=KL(te,tv).tolist(); acc["mar|vul"]+=KL(tm,tv).tolist()

print(f"T0={T0}  gamma={GAMMA}  (per-sample mean KL between softened TARGETS)")
print("="*58)
print("  vs GLOBAL target (얼마나 per-sample 소프트닝이 타겟을 움직이나):")
print(f"    global vs entropy-softened      = {np.mean(acc['g|ent']):.4f}")
print(f"    global vs margin-softened       = {np.mean(acc['g|mar']):.4f}")
print(f"    global vs vulnerability-softened= {np.mean(acc['g|vul']):.4f}")
print("  타겟끼리 (신호가 정말 다른 타겟을 만드나):")
print(f"    entropy-tgt vs margin-tgt       = {np.mean(acc['ent|mar']):.4f}   {'<- 사실상 동일' if np.mean(acc['ent|mar'])<0.02 else ''}")
print(f"    entropy-tgt vs vulnerability-tgt= {np.mean(acc['ent|vul']):.4f}   {'<- 다른 타겟!' if np.mean(acc['ent|vul'])>0.05 else ''}")
print(f"    margin-tgt  vs vulnerability-tgt= {np.mean(acc['mar|vul']):.4f}")
