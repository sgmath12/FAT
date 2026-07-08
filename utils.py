from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchattacks
import os,pdb,copy,math
from Externals.autoattack import AutoAttack
from Externals.robustbench.eval import benchmark
from Externals.robustbench.data import load_cifar10, load_cifar100, load_imagenet
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from converter import Converter
from torchvision.models import vit_b_16
import argparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def inner_loss_only_return(model,
                teacher_logits,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=6.0):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False,reduce=False)
    model.eval()

    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(teacher_logits, dim=1))
            loss_kl = torch.sum(loss_kl)
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    return x_adv


class TauNet(nn.Module):
    """Interpretable LEARNED per-sample teacher temperature (user's idea, 2026-07-04).

    Outputs a raw, UNCONSTRAINED per-sample score r(s) from 3 interpretable logit statistics only:
        norm = ||teacher_logits||_2, margin = top1 - top2, entropy = H(softmax(teacher_logits)).
    No raw features go in -- only these 3 scalars -- so r(x) stays inspectable/plottable per sample.

    r is turned into tau(x) = config.tau * exp(r_centered) in the training loop (see
    train_temperature_taunet), with the SAME batch log-centering trick train_temperature_tadapt
    uses: r_centered = clamp(r, -c, c) - batch_mean(clamp(r, -c, c)). This is NOT optional --
    an earlier version had this net directly emit a sigmoid-bounded tau in [1,20] with no batch
    anchor, and it collapsed to the clamp ceiling within 1 epoch: nn.KLDivLoss's target-entropy
    term (sum q*log(q), normally a constant when the teacher/tau is fixed) is NOT constant when
    tau is learned, so the optimizer could trivially shrink the loss by flattening EVERY sample's
    target (raising tau for the whole batch) with zero relation to actual distillation quality.
    Batch log-centering pins geomean(tau(x))==config.tau every step, closing that loophole --
    only RELATIVE per-sample softening/sharpening can still move the loss.

    Final layer zero-init (weight=0, bias=0) -> r==0 for every sample at step 0 -> tau(x)==config.tau
    exactly, matching the fixed-temperature baseline before any learning happens.

    Optional capacity knobs (default = the original minimal net, so old configs are unaffected):
    - use_bn: BatchNorm1d(3, affine=False) on the raw stats before the MLP. norm/margin/entropy live
      on very different scales (~10-20 / variable / 0-4.6 nats); without this the first layer's
      random init sees a lopsided input and has to learn the rescaling itself. Safe with a FROZEN
      teacher (norm/margin/entropy's population distribution never shifts during training).
    - hidden / depth: width and number of hidden layers (depth=1 == original single hidden layer).
    """
    def __init__(self, hidden=32, depth=1, use_bn=False):
        super().__init__()
        self.bn = nn.BatchNorm1d(3, affine=False) if use_bn else nn.Identity()
        layers = [nn.Linear(3, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, s):
        return self.net(self.bn(s))


class DeltaNet(nn.Module):
    """Learned per-sample VECTOR edit of the teacher target (user's idea, 2026-07-06): the last live
    axis after every per-sample SCALAR failed. A scalar on teacher logits is a temperature
    reparametrization and is absorbed by softmax (proven inert at T=16, where multiplicative changes
    barely move the near-uniform target -- diag_target_level.py); a VECTOR delta can move probability
    mass BETWEEN classes -- the target-DIRECTION axis that only hand-crafted swap ever touched.

    Consumes ONLY the teacher's logits (pre-divided by config.tau, so O(1) scale) -- no features, no
    labels. Emits a raw delta in R^C; the training loop (train_temperature_deltanet_bilevel)
    constrains it to direction-only: mean-centered (softmax shift invariance), per-sample norm-capped
    (config.delta_r), and ENTROPY-MATCHED back to the fixed-tau baseline target so overall
    sharpness -- the bilevel estimator's confirmed cheat axis (see the globaltau collapse) -- is
    structurally unlearnable.

    Final layer zero-init -> delta==0 for every sample at step 0 -> target == the plain temperature
    baseline exactly (same fair-start convention as TauNet/T_head/p_head).
    """
    def __init__(self, num_classes, hidden=128, depth=1, use_bn=False):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_classes, affine=False) if use_bn else nn.Identity()
        layers = [nn.Linear(num_classes, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, num_classes)]
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z):
        return self.net(self.bn(z))


def load_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", default="clean.yaml")
    parser.add_argument("--alpha", default=-1, type = float)
    parser.add_argument("--lamda", default=-1, type = float)
    parser.add_argument("--beta", default=-1, type = float)
    parser.add_argument("--gamma", default=-1, type = float)
    parser.add_argument("--eta", default=-1, type = float)
    parser.add_argument("--tau", default=-1, type = float)
    parser.add_argument("--pct", default=-1, type = float)
    parser.add_argument("--epochs", default=-1, type = int)
    parser.add_argument("--temperature", default=-1, type = int)
    parser.add_argument("--feat_scale", default=-1, type = float)
    parser.add_argument("--delta_r", default=-1, type = float)
    parser.add_argument("--delta_meta_lr", default=-1, type = float)
    parser.add_argument("--tau_meta_lr", default=-1, type = float)
    parser.add_argument("--bilevel_start", default=-1, type = int)
    parser.add_argument("--bilevel_end", default=-1, type = int)
    parser.add_argument("--seed", default=0, type = int)
    parser.add_argument("--lr", default=-1, type = float)
    parser.add_argument("--batch_size", default=-1, type = float)
    parser.add_argument("--tags", default = " ",type = str)
    parser.add_argument("--optim", default = "None",type = str)
    parser.add_argument("--checkpoint", default="-1",type = str)
    parser.add_argument('--schedule', nargs='+', default=None, help='<Required> Set flag', required=False)
    parser.add_argument('--arch', default = "None", required=False)
    parser.add_argument('--wandb_name', default = "default", required=False)
    parser.add_argument('--dataset', default = "CIFAR10", required=False)

    args = parser.parse_args()
    return args

def load_config(args):
    path = Path(os.path.realpath(__file__))
    path = str(path.parent.absolute())
    root = path + "/config/" + args.dataset + "/"  + args.config_name
    with open(root) as file:
        config = yaml.safe_load(file)
    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    def convert(s):
        try:
            return float(s)
        except ValueError:
            
            return float(num) / float(denom)

    config = dotdict(config)
    config.lamda = args.lamda if args.lamda != -1 else config.lamda
    config.alpha = args.alpha if args.alpha != -1 else config.alpha
    config.beta = args.beta if args.beta != -1 else config.beta
    config.gamma = args.gamma if args.gamma != -1 else config.gamma
    config.eta = args.eta if args.eta != -1 else config.eta
    config.tau = args.tau if args.tau != -1 else config.tau
    config.temperature = args.temperature if args.temperature != -1 else config.temperature
    config.feat_scale = args.feat_scale if args.feat_scale != -1 else getattr(config, "feat_scale", 1.0)
    config.delta_r = args.delta_r if args.delta_r != -1 else getattr(config, "delta_r", None)
    config.delta_meta_lr = args.delta_meta_lr if args.delta_meta_lr != -1 else getattr(config, "delta_meta_lr", None)
    config.tau_meta_lr = args.tau_meta_lr if args.tau_meta_lr != -1 else getattr(config, "tau_meta_lr", None)
    config.bilevel_start = args.bilevel_start if args.bilevel_start != -1 else getattr(config, "bilevel_start", None)
    config.bilevel_end = args.bilevel_end if args.bilevel_end != -1 else getattr(config, "bilevel_end", None)
    config.seed = args.seed
    config.epochs = args.epochs if args.epochs != -1 else config.epochs
    config.lr = args.lr if args.lr != -1 else config.lr
    config.pct = args.pct if args.pct != -1 else config.pct
    config.batch_size = int(args.batch_size) if args.batch_size != -1 else int(config.batch_size)
    config.config_name = args.config_name
    config.optim = args.optim if args.optim != "None" else config.optim
    config.schedule = args.schedule if args.schedule != None else config.schedule
    config.checkpoint = args.checkpoint if args.checkpoint != "-1" else config.checkpoint
    config.arch = args.arch if args.arch != "None" else config.arch
    config.wandb_name = args.wandb_name


    

    num, denom = config.eps.split('/')
    config.eps = float(num)/float(denom)
    num1,num2,num3 = config.step_size.split('/')
    config.step_size = (float(num1)/float(num2))/float(num3)

    return config

def get_model(config):


    if config.dataset == 'CIFAR10':
        mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

        from CIFAR10.models.resnet import ResNet18
        from CIFAR10.models.resnet_z import ResNet18_z
        model = ResNet18()
        model_reform = ResNet18_z()

        # student & teacher feature-normalization are INDEPENDENT toggles.
        # Fall back to config.reformation when the explicit flags are absent (old configs unchanged).
        student_norm = getattr(config, "student_norm", None)
        if student_norm is None: student_norm = config.reformation
        teacher_norm = getattr(config, "teacher_norm", None)
        if teacher_norm is None: teacher_norm = config.reformation
        base_teacher = model_reform if teacher_norm else model
        base_student = copy.deepcopy(model_reform if student_norm else model)

        if config.convert is False:
            # CURE-style: no input normalization, model operates on raw [0,1]
            teacher_model = base_teacher
            student_model = base_student
        else:
            teacher_model = Converter(base_teacher, mean, std)
            student_model = Converter(base_student, mean, std)

    if config.dataset == 'CIFAR100':
        # CIFAR-100 stats; reuse the CIFAR10/ ResNet defs with num_classes=100
        mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

        from CIFAR10.models.resnet import ResNet18
        from CIFAR10.models.resnet_z import ResNet18_z
        model = ResNet18(num_classes=100)
        if bool(getattr(config, "block_norm", False)):
            from CIFAR10.models.resnet_zbn import ResNet18_zbn
            model_reform = ResNet18_zbn(num_classes=100, scale=(getattr(config, "feat_scale", 1.0) or 1.0), block_norm=True)
        elif bool(getattr(config, "p_adapt", False)):
            from CIFAR10.models.resnet_zp import ResNet18_zp
            model_reform = ResNet18_zp(num_classes=100, scale=(getattr(config, "feat_scale", 1.0) or 1.0))
        elif bool(getattr(config, "p_global", False)):
            from CIFAR10.models.resnet_zp import ResNet18_zpg
            model_reform = ResNet18_zpg(num_classes=100, scale=(getattr(config, "feat_scale", 1.0) or 1.0))
        elif bool(getattr(config, "cos_head", False)):
            # fully directional student: feature AND classifier weights normalized, logits = s*cos.
            # See resnet_zcos.py for the fair-start/learnable-s rationale.
            from CIFAR10.models.resnet_zcos import ResNet18_zcos
            model_reform = ResNet18_zcos(num_classes=100, scale=(getattr(config, "feat_scale", 1.0) or 1.0))
        else:
            model_reform = ResNet18_z(num_classes=100, scale=(getattr(config, "feat_scale", 1.0) or 1.0))

        # student & teacher feature-normalization are INDEPENDENT toggles.
        # Fall back to config.reformation when the explicit flags are absent (old configs unchanged).
        student_norm = getattr(config, "student_norm", None)
        if student_norm is None: student_norm = config.reformation
        teacher_norm = getattr(config, "teacher_norm", None)
        if teacher_norm is None: teacher_norm = config.reformation
        base_teacher = model_reform if teacher_norm else model
        base_student = copy.deepcopy(model_reform if student_norm else model)

        if config.convert is False:
            teacher_model = base_teacher
            student_model = base_student
        else:
            teacher_model = Converter(base_teacher, mean, std)
            student_model = Converter(base_student, mean, std)


    if config.load:
        path = Path(os.path.realpath(__file__))
        path = str(path.parent.absolute())
        try  :
            checkpoint = os.path.join(path, config.checkpoint)
            checkpoint = torch.load(checkpoint)
            teacher_model.load_state_dict(checkpoint)
        except:
            print ("There is no natural model! ")
            exit()
        
    config.finetune_checkpoint = config.checkpoint
    if config.finetune : 
        path = Path(os.path.realpath(__file__))
        path = str(path.parent.absolute())
        try  :
            checkpoint = os.path.join(path, config.finetune_checkpoint)
            checkpoint = torch.load(checkpoint)
            try:
                student_model.load_state_dict(checkpoint)
            except RuntimeError:
                # students with extra heads (e.g. p_adapt's p_head) load the clean backbone with strict=False
                r = student_model.load_state_dict(checkpoint, strict=False)
                print ("finetune loaded strict=False; missing keys:", r.missing_keys)
        except:
            print ("There is no natural model! ")

    if bool(getattr(config, "t_adapt", False)):
        # learned per-sample teacher temperature T(x): tiny zero-init head on the STUDENT,
        # consuming the (detached) TEACHER's raw 512-d feature. See train_temperature_tadapt.
        student_model.T_head = nn.Linear(512, 1)
        nn.init.zeros_(student_model.T_head.weight)
        nn.init.zeros_(student_model.T_head.bias)

    if bool(getattr(config, "tau_adapt", False)):
        # interpretable learned per-sample teacher temperature tau(x): TauNet lives on the STUDENT
        # so it shares the optimizer; consumes only 3 logit statistics (norm/margin/entropy), never
        # a raw feature. See train_temperature_taunet.
        student_model.tau_net = TauNet(
            hidden=int(getattr(config, "tau_hidden", None) or 32),
            depth=int(getattr(config, "tau_depth", None) or 1),
            use_bn=bool(getattr(config, "tau_bn", False)),
        )

    if bool(getattr(config, "tau_global_bilevel", False)):
        # ONE single learnable GLOBAL scalar temperature (no per-sample structure at all, no MLP) --
        # user's idea, 2026-07-05: isolate whether bilevel can find a better GLOBAL constant than the
        # hand-picked config.tau=16, separate from the per-sample-structure question. log-parametrized
        # so tau=exp(log_tau) stays positive for any real-valued update; init at log(config.tau) so
        # tau(x)==config.tau exactly at step 0 (matches every other variant's fair-start convention).
        # See train_temperature_bilevel_globaltau.
        student_model.log_tau = nn.Parameter(torch.tensor(float(np.log(config.tau))))

    if bool(getattr(config, "tau_classwise", False)):
        # ONE learnable temperature PER CLASS (logit coordinate), shared by every sample -- user's
        # idea, 2026-07-07: the last open cell of the {per-sample, per-class} x {scalar, direction}
        # matrix. Unlike the per-sample scalar (temperature reparametrization, proven inert), a
        # per-class divisor CAN reorder classes; unlike the additive per-class tilt the uncentered
        # DeltaNet pilot converged to (bought nothing), it acts proportionally to logit magnitude.
        # zero-init -> tau_c == config.tau exactly at step 0 (fair-start convention); the training
        # loop log-centers so only RELATIVE per-class structure is learnable (geomean pinned to
        # config.tau -- the global-sharpness cheat that collapsed globaltau is unreachable).
        # See train_temperature_tauclass_bilevel.
        _nc = 100 if config.dataset == "CIFAR100" else 10
        student_model.log_tau_c = nn.Parameter(torch.zeros(_nc))

    if bool(getattr(config, "delta_adapt", False)):
        # learned per-sample VECTOR edit of the teacher target: DeltaNet lives on the STUDENT (shares
        # the optimizer via model.parameters(), same as tau_net); consumes only teacher_logits/tau.
        # See train_temperature_deltanet_bilevel.
        _nc = 100 if config.dataset == "CIFAR100" else 10
        student_model.delta_net = DeltaNet(
            _nc,
            hidden=int(getattr(config, "delta_hidden", None) or 128),
            depth=int(getattr(config, "delta_depth", None) or 1),
            use_bn=bool(getattr(config, "delta_bn", False)),
        )

    student_model = student_model.cuda()
    teacher_model = teacher_model.cuda()

    return teacher_model, student_model




def clamp(input, min=None, max=None):
    ndim = input.ndimension()
    if min is None:
        pass
    elif isinstance(min, (float, int)):
        input = torch.clamp(input, min=min)
    elif isinstance(min, torch.Tensor):
        if min.ndimension() == ndim - 1 and min.shape == input.shape[1:]:
            input = torch.max(input, min.view(1, *min.shape))
        else:
            assert min.shape == input.shape
            input = torch.max(input, min)
    else:
        raise ValueError("min can only be None | float | torch.Tensor")

    if max is None:
        pass
    elif isinstance(max, (float, int)):
        input = torch.clamp(input, max=max)
    elif isinstance(max, torch.Tensor):
        if max.ndimension() == ndim - 1 and max.shape == input.shape[1:]:
            input = torch.min(input, max.view(1, *max.shape))
        else:
            assert max.shape == input.shape
            input = torch.min(input, max)
    else:
        raise ValueError("max can only be None | float | torch.Tensor")
    return input

def CW_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()

    loss_value = -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind))
    return loss_value.mean()

def cw_Linf_attack(model, X, y, epsilon, alpha, attack_iters, restarts):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    # y_true = np.eye(10)[y.cuda().data.cpu().numpy()]
    # y_true = torch.from_numpy(y_true).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        # for i in range(len(epsilon)):
        #     delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta += torch.FloatTensor(*delta.shape).uniform_(-epsilon, epsilon).cuda()
        delta.data = clamp(delta, 0 - X, 1 - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)

            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = CW_loss(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, 0 - X[index[0], :, :, :], 1 - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X + delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return X + max_delta

def evaluate(model,loader, config):
    model.eval()
    eps = config.eps
    clean_acc = 0
    fgsm_acc = aa_acc = pgd_acc = pgd_l1_acc = pgd10_acc = pgd50_acc = cw_acc = 0
    total_samples = 0
    fgsm_attack = torchattacks.FGSM(model,eps = eps)
    
    pgd_attack = torchattacks.PGD(model,eps = eps,steps = 20, alpha = 2/255, random_start = True)
    pgd10_attack = torchattacks.PGD(model,eps = eps,steps = 10, alpha = 2/255, random_start = True)
    pgd50_attack = torchattacks.PGD(model,eps = eps,steps = 50, alpha = 2/255, random_start = True)
    pgdl2_attack = torchattacks.PGDL2(model, eps=128/255, alpha=0.1, steps=20, random_start=True)


    for i,(x,y) in enumerate(loader):
        total_samples += x.shape[0]
        x,y = x.to(device), y.to(device)
        
        x_fgsm = fgsm_attack(x,y)
        x_pgd = pgd_attack(x,y)
        x_pgd10 = pgd10_attack(x,y)
        x_pgd50 = pgd50_attack(x,y)
        # x_cw = cw_attack(x,y)
        x_cw = cw_Linf_attack(model,x,y,eps,2/255,20,1)
        # x_pgd_l1, z_l1_acc = PGD_L1_Attack(model,x,y)
        # x_pgd_l1 = PGD_L1_Attack(model,x,y)
        # x_pgd_l2 = pgdl2_attack(x,y)
  
        z_clean = model(x)
        z_fgsm = model(x_fgsm)
        z_pgd = model(x_pgd)
        z_pgd10 = model(x_pgd10)
        z_pgd50 = model(x_pgd50)
        z_cw = model(x_cw)
        # z_pgd_l1 = model(x_pgd_l1)
        # z_pgd_l2 = model(x_pgd_l2)


  
        z_clean_out = z_clean.argmax(dim = 1)
        z_fgsm_out = z_fgsm.argmax(dim = 1)
        z_pgd_out = z_pgd.argmax(dim = 1)
        z_pgd10_out = z_pgd10.argmax(dim = 1)
        z_pgd50_out = z_pgd50.argmax(dim = 1)
        z_cw_out = z_cw.argmax(dim = 1)
        # z_pgd_l1_out = z_pgd_l1.argmax(dim = 1)
        # z_pgd_l2_out = z_pgd_l2.argmax(dim = 1)

        clean_acc += (z_clean_out ==y).sum().detach().cpu()
        fgsm_acc += (z_fgsm_out == y).sum().detach().cpu()
        pgd_acc += (z_pgd_out == y).sum().detach().cpu()
        pgd10_acc += (z_pgd10_out == y).sum().detach().cpu()
        pgd50_acc += (z_pgd50_out == y).sum().detach().cpu()
        cw_acc += (z_cw_out == y).sum().detach().cpu()
        # pgd_l1_acc += (z_pgd_l1_out == y).sum().detach()
        # pgd_l2_acc += (z_pgd_l2_out == y).sum().detach()

        
    clean_acc = clean_acc/total_samples * 100.0
    fgsm_acc = fgsm_acc/total_samples * 100.0
    pgd_acc = pgd_acc/total_samples * 100.0
    pgd10_acc = pgd10_acc/total_samples * 100.0
    pgd50_acc = pgd50_acc/total_samples * 100.0
    cw_acc = cw_acc/total_samples * 100.0

    return clean_acc.item(), fgsm_acc.item(), pgd_acc.item(), pgd10_acc.item(), pgd50_acc.item(), cw_acc.item()

def evaluate_final_aa(model, loader, args,  n_samples = 10000, full = False, eps = 8/255.0):
    
    adversary = AutoAttack(model, norm='Linf', eps=args.eps, version='standard')
    # adversary = AutoAttack(model, norm='Linf', eps=8/255.0, version='standard')
    if not full :
        adversary.attacks_to_run = ['apgd-ce','apgd-t']
    else : 
        None

    l = [x for (x, y) in loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in loader]
    y_test = torch.cat(l, 0)    
    adv_complete, robust_acc = adversary.run_standard_evaluation(x_test[:], y_test[:], bs=args.batch_size)
    return robust_acc * 100

def evaluate_pgd(model,loader, config = None):
    eps = config.eps
    model.eval()
    clean_acc = 0
    fgsm_acc = aa_acc = pgd_acc = pgd_l1_acc = pgd_l2_acc =  0
    total_samples = 0
    pgd10_attack = torchattacks.PGD(model, eps= eps, alpha=2/225, steps=10, random_start=True)

    for i,(x,y) in enumerate(loader):
        total_samples += x.shape[0]
        x,y = x.cuda(), y.cuda()
        x_pgd10 = pgd10_attack(x,y)
        # x_pgd_l1, z_l1_acc = PGD_L1_Attack(model,x,y)
    
        # x_pgdl2 = pgdl2_attack(x,y)

        z_pgd = model(x_pgd10)
        # z_pgd_l2 = model(x_pgdl2)
        z = model(x)

        z_pgd_out = z_pgd.argmax(dim = 1)
        # z_pgd_l2_out = z_pgd_l2.argmax(dim = 1)
        z_out = z.argmax(dim = 1)

        pgd_acc += (z_pgd_out == y).sum()
        # pgd_l1_acc += z_l1_acc
        # pgd_l2_acc += (z_pgd_l2_out == y).sum()
        clean_acc += (z_out == y).sum()

    clean_acc = clean_acc/total_samples * 100.0
    pgd_acc = pgd_acc/total_samples * 100.0
    # pgd_l1_acc = pgd_l1_acc/total_samples * 100
    # pgd_l2_acc = pgd_l2_acc/total_samples * 100

    return  clean_acc.detach().cpu().item(), pgd_acc.detach().cpu().item()
 
def evaluate_clean(model,loader, config = None):
    model.eval()
    clean_acc = 0
    total_samples = 0

    for i,(x,y) in enumerate(loader):
        total_samples += x.shape[0]
        x,y = x.to(device), y.to(device)
        z_clean = model(x)
        z_clean_out = z_clean.argmax(dim = 1)
        clean_acc += (z_clean_out ==y).sum().detach()

    clean_acc = clean_acc/total_samples  * 100.0

    return clean_acc.detach().cpu().item(),0
