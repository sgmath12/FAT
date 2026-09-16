import argparse
import logging
import sys
import time
import math
import datetime
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard.writer import SummaryWriter
from attacks import AttackerPolymer
from utils import *

from networks.vgg import VGG16
from networks.mobilenetv2 import MobileNetV2 as MobileV2
from networks.wideresnet import WideResNet
from networks.preactresnet import PreActResNet18
from networks.resnet import ResNet18
import copy


class AdvWeightPerturb:
    """AWP proxy form, copied from FAT utils.py (AdvWeightPerturb, _awp_diff_in_weights,
    _awp_add_into_weights): one ascent step on a proxy copy, norm-scaled per >=2D weight tensor,
    added with coefficient gamma before the training step and removed after it."""
    def __init__(self, model, gamma=5e-3, proxy_lr=0.01):
        self.model, self.gamma = model, gamma
        self.proxy = copy.deepcopy(model)
        self.proxy_optim = torch.optim.SGD(self.proxy.parameters(), lr=proxy_lr)

    def calc_awp(self, loss_fn):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()
        loss = -loss_fn(self.proxy)
        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()
        diff = {}
        for (k, w_old), (_, w_new) in zip(self.model.state_dict().items(), self.proxy.state_dict().items()):
            if len(w_old.size()) <= 1 or 'weight' not in k:
                continue
            d = w_new - w_old
            diff[k] = w_old.norm() / (d.norm() + 1e-20) * d
        return diff

    def _add(self, diff, coeff):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in diff:
                    param.add_(coeff * diff[name])

    def perturb(self, diff):
        self._add(diff, self.gamma)

    def restore(self, diff):
        self._add(diff, -self.gamma)

parser = argparse.ArgumentParser()

# general
parser.add_argument('--fname', default='cifar', type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--val', action='store_true')  # use validation-based early stopping

# evaluation
parser.add_argument('--eval', action='store_true')  # evaluation mode
parser.add_argument('--eval-last-only', action='store_true')
parser.add_argument('--eval-best-only', action='store_true')
parser.add_argument('--eval-online', action='store_true')
parser.add_argument('--eval-train-robust', action='store_true')

# model
parser.add_argument('--model', default='PreActResNet18',
                    choices=['PreActResNet18', 'ResNet18', 'WideResNet', 'VGG16', 'MobileNet'])
parser.add_argument('--width-factor', default=10, type=int)  # for WRN
parser.add_argument('--resume', default=0, type=int)  # resume from this epoch
parser.add_argument('--load-folder', default=None,
                    type=str)  # can specify a folder to load checkpoints; if not specified, load from default folder
parser.add_argument('--save-path', type=str, default='exps')
parser.add_argument('--chkpt-iters', default=10, type=int)  # checkpoint save frequency

# WA model
parser.add_argument('--decay-rate', default=0.999, type=float)
parser.add_argument('--warmup-epochs', default=105, type=int)

# dataset
parser.add_argument('--data-dir', default='cifar-data', type=str)
parser.add_argument('--num-classes', default=10, type=int)  # set to 100 for CIFAR 100

# data augmentation (CutMix)
parser.add_argument('--cutmix', action='store_true')
parser.add_argument('--cutmix-alpha', type=float, default=1.0)
parser.add_argument('--cutmix-beta', type=float, default=1.0)

# training
parser.add_argument('--l2', default=0, type=float)
parser.add_argument('--l1', default=0, type=float)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=200, type=int)

# learning rate
parser.add_argument('--lr-schedule', default='piecewise', choices=['piecewise', 'linear', 'cosine', 'constant'])
parser.add_argument('--lr-max', default=0.1, type=float)
parser.add_argument('--lr-factor', type=float, default=1.5)  # decay factor for piecewise schedule
parser.add_argument('--stage1', type=int, default=100)
parser.add_argument('--stage2', type=int, default=150)

# attacker
parser.add_argument('--attack', default='pgd', type=str, choices=['pgd', 'none'])
parser.add_argument('--eval-attack', default='pgd', type=str, choices=['pgd', 'none'])
parser.add_argument('--epsilon', default=8, type=int)
parser.add_argument('--attack-iters', default=10, type=int)
parser.add_argument('--restarts', default=1, type=int)
parser.add_argument('--pgd-alpha', default=2, type=float)
parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])

# stronger attacker for ReBAT++
parser.add_argument('--stronger-attack', action='store_true')
parser.add_argument('--stronger-epsilon', default=10, type=int)
parser.add_argument('--stronger-attack-iters', default=12, type=int)
parser.add_argument('--stronger-eval', action='store_true')  # also use stronger attack during evaluation

# AWP (Wu et al., NeurIPS 2020), proxy form, added 2026-09-15 to give RPAT++ the same stack as every
# other baseline: its own WA stays as it is, and AWP is added with the gamma, proxy lr and warmup
# fraction (10% of epochs) used across the FAT experiments.  gamma 0 (default) = upstream behaviour.
parser.add_argument('--awp-gamma', type=float, default=0.0)
parser.add_argument('--awp-warmup', type=int, default=0)
parser.add_argument('--awp-proxy-lr', type=float, default=0.01)

# Natural warm start (2026-09-16), so RPAT++ gets the same three things every other baseline in the
# paper's stack table gets: its own recipe, initialization at our naturally trained network, and the
# stack.  The checkpoint is a FAT state dict: its keys carry an `encoder.` prefix, and the network is
# wrapped there in a Converter that normalizes with the CIFAR-100/10 channel statistics below, which
# are NOT the ones this script hardcodes (it uses CIFAR-10's on both datasets).  Passing a checkpoint
# therefore also switches the normalization to the one the loaded weights were trained with; measured
# 77.25% clean on a CIFAR-100 sample with these statistics against 36.30% with the script's own.
parser.add_argument('--natural-init', type=str, default='')

# BoAT regularization
parser.add_argument('--use-reg-schedule',
                    action='store_true')  # if set to False, by default it stays constant as args.beta
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--beta-factor', type=float, default=1.5)  # multiply factor in piecewise schedule
parser.add_argument('--reg-schedule', default='dependent', choices=['piecewise', 'dependent'])

args = parser.parse_args()


def main():
    # ------------------ basic settings ------------------
    args.fname = args.save_path + '/' + args.fname  # collect all the experiments
    if not os.path.exists(args.fname):
        os.makedirs(args.fname)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.fname, 'eval.log' if args.eval else 'output.log')),
            logging.StreamHandler()
        ])

    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    transforms = [Crop(32, 32), FlipLR()]
    if args.val:  # please use --val for validation-based early stopping
        try:
            dataset = torch.load("cifar10_validation_split.pth") if args.num_classes == 10 else torch.load(
                "cifar100_validation_split.pth")
        except:
            print("Couldn't find a dataset with a validation split, did you run "
                  "generate_validation.py?")
            return
        val_set = list(zip(transpose(dataset['val']['data'] / 255.), dataset['val']['labels']))
        val_batches = Batches(val_set, args.batch_size, shuffle=False, num_workers=2)
    else:
        dataset = cifar(args.data_dir, num_classes=args.num_classes)
    train_set = list(zip(transpose(pad(dataset['train']['data'], 4) / 255.),
                         dataset['train']['labels']))
    train_set_x = Transform(train_set, transforms)
    train_batches = Batches(train_set_x, args.batch_size, shuffle=True, set_random_choices=True, num_workers=2)

    test_set = list(zip(transpose(dataset['test']['data'] / 255.), dataset['test']['labels']))
    test_batches = Batches(test_set, args.batch_size, shuffle=False, num_workers=2)

    # ------------------ attacker ------------------
    epsilon = (args.epsilon / 255.)
    pgd_alpha = (args.pgd_alpha / 255.)

    Attackers = AttackerPolymer(epsilon, args.attack_iters, pgd_alpha, args.num_classes, device)

    # ------------------ model ------------------
    if args.model == 'PreActResNet18':
        model = PreActResNet18(num_classes=args.num_classes)
    elif args.model == 'ResNet18':  # the post-activation ResNet-18 used by our own experiments
        model = ResNet18(num_classes=args.num_classes)
    elif args.model == 'WideResNet':
        model = WideResNet(34, 10, widen_factor=args.width_factor, dropRate=0.0)
    elif args.model == 'VGG16':
        model = VGG16(n_classes=args.num_classes)
    elif args.model == 'MobileNet':
        model = MobileV2(num_classes=args.num_classes)
    else:
        raise ValueError("Unknown model")

    if args.natural_init:
        import utils as _rpat_utils
        sd = torch.load(args.natural_init, map_location='cpu')
        sd = {k[len('encoder.'):] if k.startswith('encoder.') else k: v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        fat_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        fat_std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        _rpat_utils.mu = torch.tensor(fat_mean).view(3, 1, 1).to(_rpat_utils.mu)
        _rpat_utils.std = torch.tensor(fat_std).view(3, 1, 1).to(_rpat_utils.std)
        globals()['mu'] = _rpat_utils.mu
        globals()['std'] = _rpat_utils.std
        logger.info(f'natural init from {args.natural_init} '
                    f'(missing {list(missing)}, unexpected {list(unexpected)}); '
                    f'normalization switched to the teacher\'s {fat_mean} / {fat_std}')

    model.train()
    model.to(device)

    from copy import deepcopy
    model_wa = deepcopy(model)

    # ------------------ optimizer ------------------
    if args.l2:
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if 'bn' not in name and 'bias' not in name:
                decay.append(param)
            else:
                no_decay.append(param)
        params = [{'params': decay, 'weight_decay': args.l2},
                  {'params': no_decay, 'weight_decay': 0}]
    else:
        params = model.parameters()

    opt = torch.optim.SGD(params, lr=args.lr_max, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    # ------------------ learning rate decay schedule ------------------
    if args.lr_schedule == 'constant':
        lr_schedule = lambda t: args.lr_max
    elif args.lr_schedule == 'piecewise':
        def lr_schedule(t):
            if t < args.stage1:
                return args.lr_max
            elif t < args.stage2:
                return args.lr_max / args.lr_factor
            else:
                return args.lr_max / args.lr_factor ** 2
    elif args.lr_schedule == 'linear':
        lr_schedule = lambda t: np.interp([t], [0, args.epochs // 3, args.epochs * 2 // 3, args.epochs],
                                          [args.lr_max, args.lr_max, args.lr_max / 10, args.lr_max / 100])[0]
    elif args.lr_schedule == 'cosine':
        def lr_schedule(t):
            return args.lr_max * 0.5 * (1 + np.cos(t / args.epochs * np.pi))
    else:
        raise NotImplementedError("Unknown LR decay schedule!")

    # ------------------ BoAT regularization strength schedule ------------------
    # will only be used when args.use_reg_schedule=True; by default it stays constant as args.beta
    if args.reg_schedule == 'piecewise':
        def reg_schedule(t):
            if t < args.stage2:  # WA and BoAT regularization start after the first LR decay, usually at epoch 105
                return args.beta
            else:
                return args.beta * args.beta_factor
    elif args.reg_schedule == 'dependent':
        def reg_schedule(t):
            rate = lr_schedule(t)
            return (args.lr_max / rate - 1) / 2
    else:
        raise NotImplementedError("Unknown regularization schedule!")

    # ------------------ preparation for training ------------------
    best_test_robust_acc = 0
    best_test_robust_acc_wa = 0
    start_epoch = 0
    epochs = args.epochs

    # resume from checkpoints
    if args.resume:
        if args.load_folder is None:
            args.load_folder = args.fname
        start_epoch = args.resume
        logger.info(f'Resuming at epoch {start_epoch}')

        # load optimizer and online model weights
        model.load_state_dict(torch.load(os.path.join(args.load_folder, f'model_{start_epoch - 1}.pth')))
        opt.load_state_dict(torch.load(os.path.join(args.load_folder, f'opt_{start_epoch - 1}.pth')))

        # load WA model weights
        try:  # after args.warmup_epochs, the WA model is different from the online model, so WA model weights and online model weights are saved into two different files
            model_wa.load_state_dict(torch.load(os.path.join(args.load_folder, f'wa_model_{start_epoch - 1}.pth')))
        except:  # before args.warmup_epochs, the WA model is the same as the online model, so we only save one copy of the weights and load the online model weights for the WA model
            model_wa.load_state_dict(torch.load(os.path.join(args.load_folder, f'model_{start_epoch - 1}.pth')))

    # ------------------ start training ------------------
    if not args.eval:
        log_dir = os.path.join(args.fname, 'tblog', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        writer = SummaryWriter(log_dir=log_dir)  # Tensorboard
        logger.info(
            'Epoch \t Train Time \t Test Time \t LR \t \t Train Loss \t Train Acc \t Train Robust Loss \t Train Robust Acc \t Test Loss \t Test Acc \t Test Robust Loss \t Test Robust Acc')

    for epoch in range(start_epoch, epochs):
        if args.eval:  # in evaluation mode, just skip the training loop
            break
        model.train()
        train_loss = 0
        train_acc = 0
        train_robust_loss = 0
        train_reg_loss = 0
        train_robust_acc = 0
        train_n = 0
        decay_rate = args.decay_rate if epoch >= args.warmup_epochs else 0.  # for WA
        beta = args.beta if epoch >= args.warmup_epochs else 0.  # force deactivating BoAT regularization before WA starts
        start_time = time.time()
        for i, batch in enumerate(train_batches):
            if args.eval:
                break
            X, y = batch['input'], batch['target']
            if args.cutmix:
                X, y_a, y_b, lam = cutmix_data(X, y, args.cutmix_alpha, args.cutmix_beta)
                X, y_a, y_b = map(Variable, (X, y_a, y_b))

            lr = lr_schedule(epoch + (i + 1) / len(train_batches))
            opt.param_groups[0].update(lr=lr)

            if args.attack == 'pgd':
                if not args.stronger_attack or epoch < args.stage1:  # ReBAT[strong]
                    if args.cutmix:
                        delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts,
                                           args.norm, mixup=True, y_a=y_a, y_b=y_b, lam=lam)
                    else:
                        delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts,
                                           args.norm)
                else:  # ReBAT (without stronger attack)
                    if args.cutmix:
                        delta = attack_pgd(model, X, y, args.stronger_epsilon / 255, pgd_alpha,
                                           args.stronger_attack_iters, args.restarts,
                                           args.norm, mixup=True, y_a=y_a, y_b=y_b, lam=lam)
                    else:
                        delta = attack_pgd(model, X, y, args.stronger_epsilon / 255, pgd_alpha,
                                           args.stronger_attack_iters, args.restarts,
                                           args.norm)
                delta = delta.detach()
            elif args.attack == 'none':  # standard training
                delta = torch.zeros_like(X)

            # The RPAT/BoAT targets come from the frozen WA model and do not depend on the trained
            # weights, so they are computed once; the loss below is then a function of the network
            # alone, which is what AWP needs to evaluate it on its proxy copy.
            interpolation_rate = 0.5
            x_adv_in = normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit))
            if beta > 0:
                with torch.no_grad():
                    benign_output = model_wa(
                        normalize(torch.clamp(X, min=lower_limit, max=upper_limit)))
                    interpolation_output = model_wa(
                        normalize(torch.clamp(X + interpolation_rate * delta[:X.size(0)], min=lower_limit, max=upper_limit)))
                ra_target = F.softmax(interpolation_output - (1 - interpolation_rate) * benign_output, dim=1)
                if args.use_reg_schedule:
                    beta = reg_schedule(epoch + (i + 1) / len(train_batches))

            def _rpat_loss(net):
                out = net(x_adv_in)
                if args.cutmix:
                    l = mixup_criterion(criterion, out, y_a, y_b, lam)
                else:
                    l = criterion(out, y)
                ### Robust Perception Adversarial Training ###
                if beta > 0:  # apply with BoAT loss
                    ra_loss = F.kl_div(F.log_softmax(interpolation_rate * out, dim=1), ra_target,
                                       reduction='batchmean')
                    if ra_loss < 1e10:
                        l = l + ra_loss * beta
                return l, out

            use_awp = args.awp_gamma > 0 and epoch >= args.awp_warmup
            if use_awp:
                if not hasattr(main, '_awp'):
                    main._awp = AdvWeightPerturb(model, gamma=args.awp_gamma, proxy_lr=args.awp_proxy_lr)
                awp_diff = main._awp.calc_awp(lambda net: _rpat_loss(net)[0])
                main._awp.perturb(awp_diff)

            robust_loss, robust_output = _rpat_loss(model)

            reg_loss = torch.tensor(0.).cuda()

            if args.l1:
                for name, param in model.named_parameters():
                    if 'bn' not in name and 'bias' not in name:
                        robust_loss += args.l1 * param.abs().sum()

            opt.zero_grad()
            robust_loss.backward()
            opt.step()
            if use_awp:
                main._awp.restore(awp_diff)

            output = model(normalize(X))
            if args.cutmix:
                loss = mixup_criterion(criterion, output, y_a, y_b, lam)
            else:
                loss = criterion(output, y)

            moving_average(model_wa, model, decay_rate, update_bn=True)

            train_robust_loss += robust_loss.item() * y.size(0)
            train_robust_acc += (robust_output.max(1)[1] == y).sum().item()
            train_reg_loss += reg_loss.item() * y.size(0)
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)

        train_time = time.time()

        # evaluate one model, can be the online/WA model
        def val(model, prefix=''):
            model.eval()
            test_loss = 0
            test_acc = 0
            test_robust_loss = 0
            test_robust_acc = 0
            test_n = 0
            true_y, pred_y, pred_y_rob = [], [], []
            batches = val_batches if args.val else test_batches

            for i, batch in enumerate(batches):
                X, y = batch['input'], batch['target']

                if args.eval_attack == 'none':
                    delta = torch.zeros_like(X)
                else:
                    if args.stronger_attack and args.stronger_eval:
                        delta = attack_pgd(model, X, y, args.stronger_epsilon / 255, pgd_alpha,
                                           args.stronger_attack_iters, args.restarts,
                                           args.norm, early_stop=args.eval)
                    else:
                        delta = attack_pgd(model, X, y, 8.0 / 255.0, 2.0 / 255, 10, args.restarts, args.norm,
                                           early_stop=args.eval)
                delta = delta.detach()

                output = model(normalize(X))
                loss = criterion(output, y)

                if args.eval_attack == 'none':
                    robust_output = output
                    robust_loss = loss
                else:
                    robust_output = model(
                        normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
                    robust_loss = criterion(robust_output, y)

                true_y.append(y.cpu())
                pred_y.append(output.argmax(1).cpu())
                pred_y_rob.append(robust_output.argmax(1).cpu())

                test_robust_loss += robust_loss.item() * y.size(0)
                test_robust_acc += (robust_output.max(1)[1] == y).sum().item()
                test_loss += loss.item() * y.size(0)
                test_acc += (output.max(1)[1] == y).sum().item()
                test_n += y.size(0)

            test_time = time.time()

            writer.add_scalar(f'{prefix}val/acc', test_acc / test_n, epoch)
            writer.add_scalar(f'{prefix}val/loss', test_loss / test_n, epoch)
            writer.add_scalar(f'{prefix}val/robust_loss', test_robust_loss / test_n, epoch)
            writer.add_scalar(f'{prefix}val/robust_acc', test_robust_acc / test_n, epoch)
            writer.add_scalar(f'{prefix}train/acc', train_acc / train_n, epoch)
            logger.info(
                '[%s] %d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f \t \t %.4f \t %.4f \t %.4f \t \t %.4f',
                prefix, epoch, train_time - start_time, test_time - train_time, lr,
                               train_loss / train_n, train_acc / train_n, train_robust_loss / train_n,
                               train_robust_acc / train_n,
                               test_loss / test_n, test_acc / test_n, test_robust_loss / test_n,
                               test_robust_acc / test_n)

            # save checkpoint
            if (epoch + 1) % args.chkpt_iters == 0 or epoch + 1 == epochs:
                torch.save(model.state_dict(), os.path.join(args.fname, f'{prefix}model_{epoch}.pth'))
                torch.save(opt.state_dict(), os.path.join(args.fname, f'{prefix}opt_{epoch}.pth'))

            return test_acc / test_n, test_loss / test_n, test_robust_acc / test_n, test_robust_loss / test_n

        # evaluate on training data (for WA model)
        def val_train(model, prefix=''):
            model.eval()
            test_loss = 0
            test_acc = 0
            test_robust_loss = 0
            test_robust_acc = 0
            test_n = 0
            true_y, pred_y, pred_y_rob = [], [], []

            for i, batch in enumerate(train_batches):
                X, y = batch['input'], batch['target']

                if args.eval_attack == 'none':
                    delta = torch.zeros_like(X)
                else:
                    delta = attack_pgd(model, X, y, 8.0 / 255.0, 2.0 / 255, 10, args.restarts, args.norm,
                                       early_stop=args.eval)
                delta = delta.detach()

                output = model(normalize(X))
                loss = criterion(output, y)

                if args.eval_attack == 'none':
                    robust_output = output
                    robust_loss = loss
                else:
                    robust_output = model(
                        normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
                    robust_loss = criterion(robust_output, y)

                true_y.append(y.cpu())
                pred_y.append(output.argmax(1).cpu())
                pred_y_rob.append(robust_output.argmax(1).cpu())

                test_robust_loss += robust_loss.item() * y.size(0)
                test_robust_acc += (robust_output.max(1)[1] == y).sum().item()
                test_loss += loss.item() * y.size(0)
                test_acc += (output.max(1)[1] == y).sum().item()
                test_n += y.size(0)

            test_time = time.time()

            true_y = torch.cat(true_y)
            pred_y = torch.cat(pred_y)
            pred_y_rob = torch.cat(pred_y_rob)
            try:
                record_stats(true_y, pred_y, pred_y_rob, writer, epoch, prefix=f'{prefix}val')
            except:
                pass

            writer.add_scalar(f'{prefix}train/acc', test_acc / test_n, epoch)
            writer.add_scalar(f'{prefix}train/loss', test_loss / test_n, epoch)
            writer.add_scalar(f'{prefix}train/robust_loss', test_robust_loss / test_n, epoch)
            writer.add_scalar(f'{prefix}train/robust_acc', test_robust_acc / test_n, epoch)
            logger.info(
                '[%s] %d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f \t \t %.4f \t %.4f \t %.4f \t \t %.4f',
                prefix, epoch, train_time - start_time, test_time - train_time, lr,
                               train_loss / train_n, train_acc / train_n, train_robust_loss / train_n,
                               train_robust_acc / train_n,
                               test_loss / test_n, test_acc / test_n, test_robust_loss / test_n,
                               test_robust_acc / test_n)

            # save checkpoint
            if (epoch + 1) % args.chkpt_iters == 0 or epoch + 1 == epochs:
                torch.save(model.state_dict(), os.path.join(args.fname, f'{prefix}model_{epoch}.pth'))
                torch.save(opt.state_dict(), os.path.join(args.fname, f'{prefix}opt_{epoch}.pth'))

            return test_acc / test_n, test_loss / test_n, test_robust_acc / test_n, test_robust_loss / test_n

        writer.add_scalar('train/loss', train_loss / train_n, epoch)
        writer.add_scalar('train/robust_loss', train_robust_loss / train_n, epoch)
        writer.add_scalar('train/reg_loss', train_reg_loss / train_n, epoch)
        writer.add_scalar('train/robust_acc', train_robust_acc / train_n, epoch)

        test_acc, test_loss, test_robust_acc, test_robust_loss = val(model)
        if epoch >= args.warmup_epochs:
            test_acc_wa, test_loss_wa, test_robust_acc_wa, test_robust_loss_wa = val(model_wa, prefix='wa_')
            if args.eval_train_robust:
                val_train(model_wa, prefix='wa_')
        else:
            test_acc_wa, test_loss_wa, test_robust_acc_wa, test_robust_loss_wa = test_acc, test_loss, test_robust_acc, test_robust_loss

        # save best
        if test_robust_acc > best_test_robust_acc:
            print(f"update best online model! Current online best: {best_test_robust_acc} -> {test_robust_acc}")
            torch.save(model.state_dict(), os.path.join(args.fname, f'model_best.pth'))
            best_test_robust_acc = test_robust_acc
        # save best
        if test_robust_acc_wa > best_test_robust_acc_wa:
            print(f"update best WA model! Current WA best: {best_test_robust_acc_wa} -> {test_robust_acc_wa}")
            torch.save(model_wa.state_dict(), os.path.join(args.fname, f'wa_model_best.pth'))
            best_test_robust_acc_wa = test_robust_acc_wa

    # ------------------ final evaluation ------------------
    print("Evaluating best and last...")
    logger.info(' \t '.join(['Mode'] + [name for name, _ in EVAL_METRICS]))
    res_fmt = '%s \t ' + '%.4f \t ' * (len(EVAL_METRICS) - 1) + '%.4f'

    # last
    if not args.eval_best_only:
        print("Now evaluating last...")
        if args.eval_online:
            model_wa.load_state_dict(torch.load(os.path.join(args.fname, f'model_{args.epochs - 1}.pth')))
        else:
            model_wa.load_state_dict(torch.load(os.path.join(args.fname, f'wa_model_{args.epochs - 1}.pth')))
        res_list = attack_all(model_wa, test_batches, Attackers)
        logger.info(res_fmt, '[last wa]', *res_list)

    # best
    if not args.eval_last_only:
        print("Now evaluating best...")
        if args.eval_online:
            model_wa.load_state_dict(torch.load(os.path.join(args.fname, f'model_best.pth')))
        else:
            model_wa.load_state_dict(torch.load(os.path.join(args.fname, f'wa_model_best.pth')))
        res_list = attack_all(model_wa, test_batches, Attackers)
        logger.info(res_fmt, '[best wa]', *res_list)


# ------------------ evaluation functions ------------------
# borrowed from SEAT - https://arxiv.org/abs/2203.09678
class AverageMeter(object):
    name = 'No name'

    def __init__(self, name='No name'):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = 0
        self.mean = 0
        self.num = 0
        self.now = 0

    def update(self, mean_var, count=1):
        if math.isnan(mean_var):
            mean_var = 1e6
            print('Avgmeter getting Nan!')
        self.now = mean_var
        self.num += count

        self.sum += mean_var * count
        self.mean = float(self.sum) / self.num


class NormInputModel(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, X):
        return self.model(normalize(X))


# reported metrics, in table order: (column name, key returned by Attackers.run_all)
EVAL_METRICS = [('clean', 'NAT'), ('PGD10', 'PGD_10'), ('PGD20', 'PGD_20'), ('PGD50', 'PGD_50'),
                ('CW', 'CW'), ('AA', 'AA')]


def attack_all(model, test_loader, Attackers):
    model = NormInputModel(model)
    model.eval()

    meters = {name: AverageMeter() for name, _ in EVAL_METRICS}
    from tqdm import tqdm
    from collections import OrderedDict
    pbar = tqdm(test_loader)
    pbar.set_description('Attacking all')

    for batch_idx, batch in enumerate(pbar):
        pbar_dic = OrderedDict()
        inputs, targets = batch['input'], batch['target']

        acc_dict = Attackers.run_all(model, inputs, targets)

        for name, key in EVAL_METRICS:
            meters[name].update(acc_dict[key][0].item(), inputs.size(0))
            pbar_dic[name] = '{:.2f}'.format(meters[name].mean)
        pbar.set_postfix(pbar_dic)

    return [meters[name].mean for name, _ in EVAL_METRICS]


if __name__ == "__main__":
    main()
