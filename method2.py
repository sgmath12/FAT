"""Champion recipe (2026-07-31): feat_direction + EMA-consistency + WA + AWP + strong train_eps.


Best record: CIFAR100 ResNet18, 100ep, lamda=0, eta=512, train_eps=8/255*1.1, freeze_lr_epoch=65
    -> 60.67 clean / 28.42 AA (NRR 38.69)


This module isolates the clean version of `train_feat_direction_ema_cons` from methods.py:
dead branches stripped, defaults locked to the champion config, comments pruned to essentials.
helpers (AdvWeightPerturb, inner_featdir_only_return, _log_gain_stats) are imported from methods.
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from tqdm import tqdm


from methods import (
    AdvWeightPerturb,
    AdvWeightPerturbSAM,
    inner_featdir_only_return,
    _log_gain_stats,
    _awp_bn_disable,
    _awp_bn_enable,
)


def train_feat_direction_ema_cons(model, train_loader, optimizer, origin_model,
                                  epoch, config, scheduler, exp_avg):
    """Champion AT step:
        1. inner PGD on feat_direction target -> x_adv
        2. outer loss = dir_loss (Q-projected feature diff) + head KL to teacher + EMA consistency
        3. optional AWP (proxy or SAM) wraps the outer step
        4. weight averaging (EMA shadow) maintained for stable consistency target
        5. LR freeze at `freeze_lr_epoch` (constant LR for the tail)
    """
    model.train()
    origin_model.eval()

    annealing = (epoch / config.epochs) ** 2
    decay = annealing * (1 - config.kappa) + config.kappa
    criterion_kl = nn.KLDivLoss(reduction='none')
    enc = model.encoder if hasattr(model, "encoder") else model
    beta = config.beta if config.beta is not None else 1.0
    scale = float(getattr(config, "feat_scale", 1.0) or 1.0)
    eps_train = float(getattr(config, "train_eps", 0.0) or 0.0) or config.eps
    eps_train = eps_train * config.beta

    # AWP (Wu et al. NeurIPS'20): off until awp_warmup, then proxy (default) or SAM.
    awp_style = str(getattr(config, "awp_style", "proxy") or "proxy")
    awp_gamma = float(getattr(config, "awp_gamma", 0.0) or 0.0)
    use_awp = awp_gamma > 0 and epoch >= int(getattr(config, "awp_warmup", 0) or 0)
    if use_awp:
        attr = "_awp_sam" if awp_style == "sam" else "_awp"
        if not hasattr(train_feat_direction_ema_cons, attr):
            cls = AdvWeightPerturbSAM if awp_style == "sam" else AdvWeightPerturb
            setattr(train_feat_direction_ema_cons, attr, cls(model, rho=awp_gamma if awp_style == "sam" else None,
                                                             gamma=None if awp_style == "sam" else awp_gamma))
        awp = getattr(train_feat_direction_ema_cons, attr)
        if awp_style == "sam":
            awp.rho = awp_gamma
        else:
            awp.gamma = awp_gamma

    # Q: fixed random subspace (or teacher span) for projecting feature diffs.
    Q = None
    span_mode = getattr(config, "featdir_span", None)
    if span_mode:
        if not hasattr(train_feat_direction_ema_cons, "_Q_cache"):
            train_feat_direction_ema_cons._Q_cache = {}
        t_enc = origin_model.encoder if hasattr(origin_model, "encoder") else origin_model
        Wt = t_enc.linear.weight.detach()
        k = int(config.eta) if (span_mode != "teacher" and getattr(config, "eta", None)) else Wt.shape[0]
        ck = (span_mode, k)
        if ck not in train_feat_direction_ema_cons._Q_cache:
            if span_mode == "teacher":
                base = Wt.t()
            else:
                g = torch.Generator().manual_seed(int(getattr(config, "featdir_span_seed", 0) or 0))
                base = torch.randn(Wt.shape[1], k, generator=g)
            Qm, _ = torch.linalg.qr(base.double().cpu())
            train_feat_direction_ema_cons._Q_cache[ck] = Qm.float().cuda()
            logging.info({"featdir_span": span_mode, "span_k": k})
        Q = train_feat_direction_ema_cons._Q_cache[ck]

    _wa_start = int(getattr(config, "wa_start", 0) or 0)
    ema_ready = bool(getattr(config, "weight_avg", False)) and epoch >= _wa_start

    dir_sum, cos_sum, n_sum = 0.0, 0.0, 0
    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        N, C, H, W = x.shape
        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()

        with torch.no_grad():
            phi_t, teacher_logits = origin_model(x, feat=True)
            phi_t_hat = F.normalize(phi_t, dim=1).detach()
            target = (teacher_logits / config.tau).detach()

        x_pgd = inner_featdir_only_return(
            model, phi_t_hat, x, optimizer, config.step_size, eps_train,
            perturb_steps=config.steps, Q=Q)

        if ema_ready:
            with torch.no_grad():
                ema_logits = functional_call(model, exp_avg, (x,))
            target_logits_src = ema_logits
        else:
            target_logits_src = None

        def _step_loss():
            feat_s, plus_logits = model(x_pgd, feat=True)
            fs_hat = F.normalize(feat_s, dim=1)
            d_feat = fs_hat - phi_t_hat
            if Q is not None:
                dir_loss_ = (d_feat @ Q).pow(2).sum(dim=1)
            else:
                dir_loss_ = d_feat.pow(2).sum(dim=1)
            loss_ = dir_loss_.mean()

            head_logits = enc.head_from_feat(scale * fs_hat.detach())
            kl_loss = criterion_kl(F.log_softmax(head_logits, dim=1), F.softmax(target, dim=1))
            loss_ = loss_ + beta * (1.0 / N) * (kl_loss.sum(dim=1)).sum()

            student_logits = model(x)  # clean forward: updates student BN running stats
            if config.lamda is not None and config.lamda > 0:
                if ema_ready:
                    target_logits = target_logits_src
                else:
                    target_logits = student_logits.detach() if bool(getattr(config, "featdir_cons_detach", False)) else student_logits
                consistency_loss = criterion_kl(
                    F.log_softmax(plus_logits, dim=1), F.softmax(target_logits, dim=1))
                loss_ = loss_ + annealing * config.lamda * (consistency_loss).mean()

            return loss_, dir_loss_, fs_hat, plus_logits

        # Outer step: AWP wraps the loss if enabled, else plain forward-backward.
        if use_awp and awp_style == "sam":
            optimizer.zero_grad()
            loss1, _, _, _ = _step_loss()
            loss1.backward()
            awp.first_step()
            _awp_bn_disable(model)
            optimizer.zero_grad()
            loss, dir_loss, fs_hat, plus_logits = _step_loss()
            loss.backward()
            awp.restore()
            _awp_bn_enable(model)
            optimizer.step()
        elif use_awp:
            def _awp_loss_fn(pm, _x_pgd=x_pgd, _phi_t_hat=phi_t_hat, _target=target):
                p_enc = pm.encoder if hasattr(pm, "encoder") else pm
                fs_, _ = pm(_x_pgd, feat=True)
                fh_ = F.normalize(fs_, dim=1)
                if Q is not None:
                    dl_ = ((fh_ - _phi_t_hat) @ Q).pow(2).sum(dim=1)
                else:
                    dl_ = (fh_ - _phi_t_hat).pow(2).sum(dim=1)
                l_ = dl_.mean()
                hl_ = p_enc.head_from_feat(scale * fh_.detach())
                kl_ = criterion_kl(F.log_softmax(hl_, dim=1), F.softmax(_target, dim=1))
                return l_ + beta * (1.0 / N) * kl_.sum(dim=1).sum()
            awp_diff = awp.calc_awp(_awp_loss_fn)
            awp.perturb(awp_diff)
            loss, dir_loss, fs_hat, plus_logits = _step_loss()
            loss.backward()
            optimizer.step()
            awp.restore(awp_diff)
        else:
            loss, dir_loss, fs_hat, plus_logits = _step_loss()
            loss.backward()
            optimizer.step()

        # LR schedule: OneCycleLR per-batch; freeze (constant) once freeze_lr_epoch is reached.
        if config.cyclic:
            _freeze = getattr(config, "freeze_lr_epoch", None)
            if _freeze is None or epoch < _freeze:
                scheduler.step()
            elif epoch == _freeze:
                _cur_lr = optimizer.param_groups[0]["lr"]
                logging.info({"freeze_lr_epoch": _freeze, "frozen_lr": _cur_lr})

        # Weight averaging: EMA shadow maintained every step from wa_start onward.
        if config.weight_avg and epoch >= _wa_start:
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 - decay) * value + decay * exp_avg[key]

        with torch.no_grad():
            dir_sum += dir_loss.sum().item()
            cos_sum += (fs_hat * phi_t_hat).sum(dim=1).sum().item()
            n_sum += N

    logging.info({"dir_loss_adv": round(dir_sum / max(n_sum, 1), 4),
                  "cos_adv": round(cos_sum / max(n_sum, 1), 4), "epoch": epoch})
    _log_gain_stats(model, epoch)
