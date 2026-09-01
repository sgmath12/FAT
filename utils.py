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


def inner_featdir_cons_return(model,
                phi_t_hat,
                x_natural,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                Q=None,
                w_cons=1.0,
                clean_logits=None):
    """Direction + CONSISTENCY hybrid attack (user, 2026-07-15): maximize
        ||Q^T(Phi_hat_s(xa) - Phi_hat_t)||^2 + w_cons * KL(model(xa) || model(x).detach())
    Rationale = the project's own matched-adversary principle (dirattack crash): the lamda
    consistency term was inert on k350 because the dir-only adversary never stressed it
    (measured backbone g_cons ~4e-6). TRADES-style second term makes x_adv challenge BOTH
    outer terms. clean_logits must be precomputed (no_grad) by the caller."""
    criterion_kl = nn.KLDivLoss(size_average=False, reduce=False)
    model.eval()
    p_clean = F.softmax(clean_logits, dim=1).detach()

    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            feat_s, logits_a = model(x_adv, feat=True)
            fs_hat = F.normalize(feat_s, dim=1)
            d = fs_hat - phi_t_hat
            loss_dir = (d @ Q).pow(2).sum() if Q is not None else d.pow(2).sum()
            loss_cons = criterion_kl(F.log_softmax(logits_a, dim=1), p_clean).sum()
            loss_atk = loss_dir + w_cons * loss_cons
        grad = torch.autograd.grad(loss_atk, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    optimizer.zero_grad()
    return x_adv


def inner_featdir_teacher_at_adv(model, teacher, x_natural, optimizer,
                                 step_size=0.003, epsilon=0.031, perturb_steps=10,
                                 Q=None, raw_student=False, raw_teacher=True):
    """Ablation counterpart of `inner_featdir_only_return` (2026-09-01): the teacher target is
    RE-READ at the perturbed point every step, so the objective is ||Phi_s(x') - Phi_t(x')|| rather
    than ||Phi_s(x') - Phi_t(x)||.

    This exists to test the paper's central structural claim directly.  Section 3.3 argues that the
    teacher's own instability cannot be inherited because Phi_t is evaluated at the clean point and
    nowhere else; the only way to check that is to move the read point and change nothing else.  With
    this attack (and the matching outer target in train_feat_direction) the method becomes our own
    version of what AdaAD does in logit space, and the teacher's measured 63.8 degrees of rotation
    under eps=8/255 enters the target.  The teacher branch is deliberately NOT detached, so the
    attack gradient flows through both networks, as it does in AdaAD."""
    model.eval()
    teacher.eval()
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            feat_s, _ = model(x_adv, feat=True)
            feat_t, _ = teacher(x_adv, feat=True)
            fs_ = feat_s if raw_student else F.normalize(feat_s, dim=1)
            ft_ = feat_t if raw_teacher else F.normalize(feat_t, dim=1)
            d = fs_ - ft_
            loss_dir = (d @ Q).pow(2).sum() if Q is not None else d.pow(2).sum()
        grad = torch.autograd.grad(loss_dir, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()
    optimizer.zero_grad()
    return x_adv.detach()


def inner_featdir_only_return(model,
                phi_t_hat,
                x_natural,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                Q=None,
                raw_student=False):
    """Direction-space inner maximization (train_feat_direction, 2026-07-13): perturb x to
    maximize || Phi_hat_s(x_adv) - Phi_hat_t(x) ||^2 = 2 - 2 cos, i.e. rotate the student's
    feature direction away from the teacher's. Neither head is touched -- the attack sees the
    same purely directional objective the backbone trains on (mirrors inner_loss_only_return,
    which pairs the KL attack with the KL loss)."""
    model.eval()

    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            feat_s, _ = model(x_adv, feat=True)
            # raw_student (2026-08-01): leave the student feature UNNORMALIZED, so the student must
            # match the teacher's unit-direction guidance with its own raw feature (magnitude
            # included). The teacher side stays normalized either way -- normalization is applied
            # only when the teacher builds its guidance.
            fs_hat = feat_s if raw_student else F.normalize(feat_s, dim=1)
            d = fs_hat - phi_t_hat
            # Q (512 x k, orthonormal): SUBSPACE-projected direction attack -- the adversary can
            # only score by rotating the feature within span(Q) (exists-and-unique cells, 2026-07-13).
            loss_dir = (d @ Q).pow(2).sum() if Q is not None else d.pow(2).sum()
        grad = torch.autograd.grad(loss_dir, [x_adv])[0]
        # epsilon / step_size may be per-sample tensors of shape [N,1,1,1] (angular-budget eps,
        # 2026-08-04). Broadcasting makes the scalar and per-sample cases identical code.
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    return x_adv


_AWP_EPS = 1e-20

def _awp_diff_in_weights(model, proxy):
    """Wu et al. NeurIPS'20 AWP port. Norm-scaled weight diff (>=2D tensors only, i.e. conv/linear
    weight matrices, not BN/bias) so the perturbation magnitude is relative per-layer."""
    diff = {}
    for (k_old, w_old), (k_new, w_new) in zip(model.state_dict().items(), proxy.state_dict().items()):
        if len(w_old.size()) <= 1 or 'weight' not in k_old:
            continue
        d = w_new - w_old
        diff[k_old] = w_old.norm() / (d.norm() + _AWP_EPS) * d
    return diff

def _awp_add_into_weights(model, diff, coeff=1.0):
    names = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names:
                param.add_(coeff * diff[name])

class AdvWeightPerturb:
    """AWP (Wu, Xia, Wang, NeurIPS 2020): finds a norm-ball weight perturbation that MAXIMIZES
    the (already input-adversarial) training loss, seeking flat minima -- standard AA/robust-
    generalization booster, orthogonal to WA. `loss_fn(proxy_model) -> scalar` lets callers reuse
    their own training loss (dir_loss + head KL for featdir, KL for the KL pipeline) instead of
    the original paper's plain CE, matching this project's own matched-adversary discipline.
    Usage per step: diff = awp.calc_awp(loss_fn); awp.perturb(diff); <backward+opt.step()>; awp.restore(diff)."""
    def __init__(self, model, gamma=5e-3, proxy_lr=0.01):
        self.model = model
        self.proxy = copy.deepcopy(model)
        self.proxy_optim = optim.SGD(self.proxy.parameters(), lr=proxy_lr)
        self.gamma = gamma

    def calc_awp(self, loss_fn):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()
        loss = -loss_fn(self.proxy)          # ascent: proxy_optim minimizes -loss = maximizes loss
        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()
        return _awp_diff_in_weights(self.model, self.proxy)

    def perturb(self, diff):
        _awp_add_into_weights(self.model, diff, coeff=self.gamma)

    def restore(self, diff):
        _awp_add_into_weights(self.model, diff, coeff=-self.gamma)


def _awp_bn_disable(model):
    """Freeze BN running-stat updates (momentum->0) for one forward. AdvWeightPerturbSAM's
    per-step recipe does two forwards (ascent-gradient pass, then the perturbed-weight pass
    whose gradient the optimizer actually uses); only the first should count towards
    running_mean/var, matching ADR's util/bypass_bn.py."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m._awp_backup_momentum = m.momentum
            m.momentum = 0

def _awp_bn_enable(model):
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and hasattr(m, '_awp_backup_momentum'):
            m.momentum = m._awp_backup_momentum
            del m._awp_backup_momentum


class AdvWeightPerturbSAM:
    """ADR-repo-style AWP port (their src/util/awp.py) -- an ALTERNATIVE to AdvWeightPerturb
    above, same reference (Wu et al. NeurIPS'20) but different mechanics. This is structurally
    a SAM (Foret et al. 2020) wrapper, not the original AWP paper's proxy-model construction:
    - no separate proxy network -- perturbs the model's OWN parameters in place, using the
      gradient just backprop'd at the CURRENT (unperturbed) weights, scaled per-parameter by
      rho*||p||/||grad|| (same norm-ratio idea as the proxy version's diff scaling, just
      computed directly from one gradient instead of a second network's ascent step).
    - perturbs ALL parameters with a grad (weight + bias + BN affine) -- the proxy version
      above restricts to >=2D conv/linear weight tensors only.
    - one hyperparameter (rho) instead of two (proxy_lr for the ascent step, gamma for how much
      of the diff to apply).
    - the perturbation is a transient "look-ahead": the caller must restore weights to w BEFORE
      optimizer.step(), so the persistent update always comes from the base optimizer at the
      true w, using the gradient measured at w+e(w) (matches ADR's second_step order: restore,
      then base_optimizer.step()) -- the proxy version's perturbation is instead what
      optimizer.step() is applied AT (perturb -> forward/backward/step -> restore).

    Usage per training step (mirrors ADR's advTrainer.py first_step/second_step, adapted to
    reuse this codebase's own already-computed x_pgd instead of a closure):
        optimizer.zero_grad(); loss1 = loss_fn(); loss1.backward()
        awp.first_step()
        _awp_bn_disable(model)          # first forward already updated BN running stats
        optimizer.zero_grad(); loss2 = loss_fn(); loss2.backward()
        awp.restore()
        _awp_bn_enable(model)
        optimizer.step()
    """
    def __init__(self, model, rho=0.005):
        self.model = model
        self.rho = rho
        self._backup = {}

    @torch.no_grad()
    def first_step(self):
        for p in self.model.parameters():
            if p.grad is None:
                continue
            self._backup[p] = p.data.clone()
            grad_norm = p.grad.norm()
            scale = self.rho * p.data.norm() / (grad_norm + _AWP_EPS)
            p.add_(p.grad * scale)

    @torch.no_grad()
    def restore(self):
        for p, old in self._backup.items():
            p.data.copy_(old)
        self._backup = {}


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
    parser.add_argument("--kappa", default=-1, type = float)
    parser.add_argument("--awp_gamma", default=-1, type = float)
    parser.add_argument("--awp_warmup", default=-1, type = int)
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
    config.kappa = args.kappa if args.kappa != -1 else config.kappa
    config.awp_gamma = args.awp_gamma if args.awp_gamma != -1 else getattr(config, "awp_gamma", 0.0)
    config.awp_warmup = args.awp_warmup if args.awp_warmup != -1 else getattr(config, "awp_warmup", 0)
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

        if str(getattr(config, "arch", "ResNet18")) == "WideResNet":
            # WideResNet-34-10 (depth=34, widen=10). model = plain net (clean teacher / teacher_norm
            # base); model_reform = WideResNet_z, the L2-normalized-feature student for featdir
            # (student_norm=True). Same backbone keys, so a clean WideResNet checkpoint loads into
            # either with strict=True.
            from CIFAR10.models.WideResNet import WideResNet
            from CIFAR10.models.WideResNet_z import WideResNet_z
            model = WideResNet(depth=34, num_classes=10, widen_factor=10)
            model_reform = WideResNet_z(depth=34, num_classes=10, widen_factor=10,
                                        scale=(getattr(config, "feat_scale", 1.0) or 1.0))
        else:
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
        # teacher_cos_head (2026-08-01): build the TEACHER from ResNet18_zcos independently of the
        # student's architecture, so a cosine-classifier-trained clean checkpoint can guide a plain
        # normalized student. Without it cos_head sets model_reform for BOTH sides at once.
        if bool(getattr(config, "teacher_cos_head", False)):
            from CIFAR10.models.resnet_zcos import ResNet18_zcos as _RZC
            base_teacher = _RZC(num_classes=(100 if config.dataset == 'CIFAR100' else 10),
                                scale=(getattr(config, "feat_scale", 1.0) or 1.0))
        else:
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

        # WideResNet-34-10 on CIFAR-100 (2026-09-01).  Until today this branch DID NOT EXIST: the
        # CIFAR-10 side had it but the CIFAR-100 side fell straight through to ResNet18, so
        # `arch: WideResNet` on CIFAR-100 silently built an 11.2M-parameter ResNet-18 instead of the
        # 46.2M WideResNet and reported it under the WRN name.  Any CIFAR-100 WRN result produced
        # before this date is a ResNet-18 result.  Same construction as the CIFAR-10 branch, so a
        # clean WideResNet checkpoint loads into either variant with strict=True.
        if str(getattr(config, "arch", "ResNet18")) == "WideResNet":
            from CIFAR10.models.WideResNet import WideResNet
            from CIFAR10.models.WideResNet_z import WideResNet_z
            model = WideResNet(depth=34, num_classes=100, widen_factor=10)
            model_reform = WideResNet_z(depth=34, num_classes=100, widen_factor=10,
                                        scale=(getattr(config, "feat_scale", 1.0) or 1.0))
        else:
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
            elif bool(getattr(config, "gain_head", False)):
                # gain-only student head: w_s,c = exp(log_g_c) * w_t,c, direction frozen at the
                # teacher head (finetune load), 100 learnable gains. See resnet_zgain.py.
                from CIFAR10.models.resnet_zgain import ResNet18_zgain
                model_reform = ResNet18_zgain(num_classes=100, scale=(getattr(config, "feat_scale", 1.0) or 1.0))
            else:
                model_reform = ResNet18_z(num_classes=100, scale=(getattr(config, "feat_scale", 1.0) or 1.0))

        # student & teacher feature-normalization are INDEPENDENT toggles.
        # Fall back to config.reformation when the explicit flags are absent (old configs unchanged).
        student_norm = getattr(config, "student_norm", None)
        if student_norm is None: student_norm = config.reformation
        teacher_norm = getattr(config, "teacher_norm", None)
        if teacher_norm is None: teacher_norm = config.reformation
        # teacher_cos_head (2026-08-01): build the TEACHER from ResNet18_zcos independently of the
        # student's architecture, so a cosine-classifier-trained clean checkpoint can guide a plain
        # normalized student. Without it cos_head sets model_reform for BOTH sides at once.
        if bool(getattr(config, "teacher_cos_head", False)):
            from CIFAR10.models.resnet_zcos import ResNet18_zcos as _RZC
            base_teacher = _RZC(num_classes=(100 if config.dataset == 'CIFAR100' else 10),
                                scale=(getattr(config, "feat_scale", 1.0) or 1.0))
        else:
            base_teacher = model_reform if teacher_norm else model
        base_student = copy.deepcopy(model_reform if student_norm else model)
        # feat_scale on the RAW student ONLY (2026-08-21).  ResNet18_z takes its scale at
        # construction, but the plain student shares its object with the teacher whenever
        # teacher_norm is False -- scaling at construction scales z_t too and cancels out, which is
        # exactly how the first attempt at this silently did nothing (warm-start clean stayed 77.4
        # at every scale, and the raw KD cell diverged just as before).  Set it after the deepcopy
        # so only the student carries it.  Default 1.0 leaves every earlier run bit-identical.
        if not student_norm:
            base_student.scale = float(getattr(config, "feat_scale", 1.0) or 1.0)

        if config.convert is False:
            teacher_model = base_teacher
            student_model = base_student
        else:
            teacher_model = Converter(base_teacher, mean, std)
            student_model = Converter(base_student, mean, std)

    if config.dataset == 'TinyImageNet':
        # Tiny-ImageNet-200: 64x64, 200 classes. Stats and, crucially, the ARCHITECTURE POLICY are
        # taken from ADR (src/model/create_model.py) so the comparison is like-for-like: ADR does
        # NOT swap in an ImageNet-style 7x7/stride-2 stem for this dataset. It keeps the very same
        # CIFAR backbone (3x3 stem, stride 1, no maxpool) and absorbs the larger input purely by
        # switching the final pooling to adaptive -- the last feature map is 8x8 here instead of
        # CIFAR's 4x4, and adaptive pooling collapses either to 1x1. Our models now use
        # F.adaptive_avg_pool2d(_, 1) for exactly this reason; on CIFAR that is bit-equivalent to
        # the fixed avg_pool2d they replaced, so no CIFAR result is affected.
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2302, 0.2265, 0.2262)

        if str(getattr(config, "arch", "ResNet18")) == "WideResNet":
            from CIFAR10.models.WideResNet import WideResNet
            from CIFAR10.models.WideResNet_z import WideResNet_z
            model = WideResNet(depth=34, num_classes=200, widen_factor=10)
            model_reform = WideResNet_z(depth=34, num_classes=200, widen_factor=10,
                                        scale=(getattr(config, "feat_scale", 1.0) or 1.0))
        else:
            from CIFAR10.models.resnet import ResNet18
            from CIFAR10.models.resnet_z import ResNet18_z
            model = ResNet18(num_classes=200)
            model_reform = ResNet18_z(num_classes=200, scale=(getattr(config, "feat_scale", 1.0) or 1.0))

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
    # AA is per-sample, so bs only chunks the work. On deep nets under WSL2 the eval is
    # kernel-launch bound (GPU util 100% but memory util 2%, power 21% of cap), so a larger
    # bs cuts launches per sample. Defaults to batch_size = unchanged behaviour.
    aa_bs = getattr(args, "aa_batch_size", None) or args.batch_size
    adv_complete, robust_acc = adversary.run_standard_evaluation(x_test[:], y_test[:], bs=aa_bs)
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
