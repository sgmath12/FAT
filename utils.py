from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchattacks
import os,pdb,copy
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

        base_teacher = model_reform if config.reformation else model
        base_student = copy.deepcopy(base_teacher)

        if config.convert is False:
            # CURE-style: no input normalization, model operates on raw [0,1]
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
            student_model.load_state_dict(checkpoint)
        except:
            print ("There is no natural model! ")
    
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
