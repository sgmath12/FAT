from asyncio.unix_events import BaseChildWatcher
from cmath import tanh
import torch.nn as nn
from torch.distributions import Beta
from tqdm import tqdm
from utils import *
from utils import _awp_bn_disable, _awp_bn_enable
import pdb
import torch.nn.functional as F

import numpy as np
from torch.autograd import Variable

import torch
import torch.nn.functional as F
import torchattacks
import logging
import copy
import math
import os
import dataset as _dataset_mod
from torch.func import functional_call
import matplotlib
matplotlib.use('Agg') # 서버 환경(GUI 없는 환경)에서 X11 화면 출력 에러 방지
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'stix' # 수식 폰트도 Times와 어울리는 STIX로 설정


def train_analyze_norm_vs_gradient(model,train_loader,optimizer, origin_model,epoch, config, scheduler, exp_avg = None, target_class = None):
    model.eval() 
    origin_model.eval()
    
    pgd_attack = torchattacks.PGD(model, eps=config.eps, steps=20, alpha=2/255, random_start=True)
    
    all_norms = []
    all_feature_distances = []
    all_grad_norms = [] 

    tau = getattr(config, 'tau', 1.0)

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.cuda(), y.cuda()

        with torch.no_grad():
            feat_teacher, output_teacher = origin_model(x, feat=True)
            norms = torch.norm(feat_teacher, p=2, dim=1) 
            
        x_pgd = pgd_attack(x, y)
        
        model.zero_grad()
        feat_adv, adv_output = model(x_pgd, feat=True)
        feat_adv.retain_grad() 
        
        loss_kl = F.kl_div(
            F.log_softmax(adv_output / tau, dim=1),
            F.softmax(output_teacher / tau, dim=1),
            reduction='batchmean'
        ) * (tau ** 2)
        
        loss_kl.backward()
        
        distances = torch.norm(feat_adv.detach() - feat_teacher.detach(), p=2, dim=1)
        grad_norms = torch.norm(feat_adv.grad.detach(), p=2, dim=1)
            
        all_norms.extend(norms.cpu().numpy())
        all_feature_distances.extend(distances.cpu().numpy())
        all_grad_norms.extend(grad_norms.cpu().numpy())

    all_norms = np.array(all_norms)
    all_feature_distances = np.array(all_feature_distances)
    all_grad_norms = np.array(all_grad_norms)

    # ================= 시각화 파트 =================
    # 파레토 그래프처럼 깔끔한 흰 배경에 연한 그리드
    sns.set_style("whitegrid", {'axes.grid': True, 'grid.linestyle': '--', 'grid.color': '#e0e0e0'})
    num_bins = 5
    
    fig, ax1 = plt.subplots(figsize=(6, 4.5)) # 캡션 공간을 위해 세로를 살짝 줄임
    
    bins = np.percentile(all_norms, np.linspace(0, 100, num_bins + 1))
    bin_indices = np.digitize(all_norms, bins) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)
    
    bin_labels = [f"Q{i+1}" for i in range(num_bins)]
    avg_dist = [all_feature_distances[bin_indices == i].mean() for i in range(num_bins)]
    avg_grad = [all_grad_norms[bin_indices == i].mean() for i in range(num_bins)]

    # 👉 [핵심 수정 1] 상대적 스케일 (Relative Scale) 계산
    # Q1의 그래디언트 노름을 1.0으로 기준 잡고 비율로 변환
    base_grad = avg_grad[0]
    rel_grad = [g / base_grad for g in avg_grad]

    # 👉 [핵심 수정 2] 세련된 색상과 테두리 (파레토 그래프 스타일)
    # 밋밋한 파란색 대신, 세련된 청록색(crest) 그라데이션 적용 + 검은색 얇은 테두리
    bar_colors = sns.color_palette("crest", num_bins)
    sns.barplot(
        x=bin_labels, y=avg_dist, 
        palette=bar_colors, ax=ax1, 
        edgecolor='black', linewidth=0.7 # 테두리를 넣으면 훨씬 선명하고 예뻐보입니다.
    )
    
    ax1.set_xlabel("Norm Quantiles ($||\Phi_t(x)||_2$)", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Avg. Feature Deviation", fontsize=13, color="#1f77b4", fontweight='bold')
    ax1.tick_params(axis='y', colors="#1f77b4")

    # 👉 [핵심 수정 3] 꺾은선 그래프 스타일 변경
    ax2 = ax1.twinx() 
    # 파레토 그래프의 'Ours'와 동일한 강렬한 빨간색 + 별 마커 + 마커 테두리 적용
    ax2.plot(
        bin_labels, rel_grad, 
        color="#d62728", marker="*", markersize=14, 
        linewidth=2.5, linestyle="--", 
        markeredgecolor='black', markeredgewidth=0.8
    )
    # Y축 라벨을 'Relative'로 변경
    ax2.set_ylabel("Relative KD Gradient Norm", fontsize=13, color="#d62728", fontweight='bold')
    ax2.tick_params(axis='y', colors="#d62728")
    
    # 불필요한 테두리 및 타이틀 제거 (잔여물 방지)
    sns.despine(right=False, top=True) 
    
    plt.tight_layout()
    
    save_path = "kd_gradient_vs_norm_relative.pdf"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Analysis complete. Plot saved to {save_path}")
    return all_norms, all_feature_distances, rel_grad

# def train_analyze_norm_vs_robustness(model,train_loader,optimizer, origin_model,epoch, config, scheduler, exp_avg = None, target_class = None):
#     model.eval() 
    
#     # PGD 공격 설정
#     pgd_attack = torchattacks.PGD(model, eps=config.eps, steps=20, alpha=2/255, random_start=True)
    
#     all_norms = []
#     all_adv_confidences = [] # 잘못된 정답에 대한 확신도
#     all_feature_distances = [] # 피처 왜곡 거리

#     for batch_idx, (x, y) in enumerate(train_loader):
#         x, y = x.cuda(), y.cuda()

#         # 1. Clean 이미지 피처 추출 및 노름 계산
#         with torch.no_grad():
#             feat_clean, _ = model(x, feat=True)
#             norms = torch.norm(feat_clean, p=2, dim=1) 
            
#         # 2. PGD 공격 수행
#         x_pgd = pgd_attack(x, y)
        
#         # 3. 적대적 이미지 분석
#         with torch.no_grad():
#             feat_adv, adv_output = model(x_pgd, feat=True)
            
#             # (대안 1) Confidence: 적대적 예제에 대한 Max Softmax Probability
#             # 이미 공격에 성공했다고 가정하므로, 모델이 내뱉는 가장 높은 확률값을 취합니다.
#             adv_probs = F.softmax(adv_output, dim=1)
#             confidences, _ = torch.max(adv_probs, dim=1)
            
#             # (대안 2) Feature Distance: Clean 피처와 Adv 피처 사이의 유클리드 거리
#             distances = torch.norm(feat_adv - feat_clean, p=2, dim=1)
            
#         all_norms.extend(norms.cpu().numpy())
#         all_adv_confidences.extend(confidences.cpu().numpy())
#         all_feature_distances.extend(distances.cpu().numpy())

#     all_norms = np.array(all_norms)
#     all_adv_confidences = np.array(all_adv_confidences)
#     all_feature_distances = np.array(all_feature_distances)

#     # ================= 시각화 (2개의 그래프 생성) =================
# # ================= 4. 본문용 컴팩트 시각화 (Deciles) =================
#     sns.set_style("whitegrid")
#     num_bins = 5
#     # wrapfigure용 사이즈 (가로 5, 세로 4.5)
#     plt.figure(figsize=(5, 4.5)) 
    
#     # 10개 구간(Deciles) 경계값 계산
#     bins = np.percentile(all_norms, np.linspace(0, 100, num_bins + 1))
#     bin_indices = np.digitize(all_norms, bins) - 1
#     bin_indices = np.clip(bin_indices, 0, num_bins - 1)
    
#     # Q1 ~ Q10 라벨 생성
#     bin_labels = [f"Q{i+1}" for i in range(num_bins)]
#     avg_dist = [all_feature_distances[bin_indices == i].mean() for i in range(num_bins)]

#     # 막대 그래프 그리기 (Blues_d: 끝으로 갈수록 진해지는 색상)
#     ax = sns.barplot(x=bin_labels, y=avg_dist, palette="Blues_d")
    
#     # 폰트 크기 최적화 (논문 삽입 시 작아지는 것을 대비)
#     # plt.title("Feature Instability vs. Norm", fontsize=14, fontweight='bold', pad=15)
#     plt.xlabel("Norm Deciles ($||\Phi_t(x)||_2$)", fontsize=12)
#     plt.ylabel("Avg. Feature Deviation", fontsize=12)
    
#     # X축 라벨이 10개면 겹칠 수 있으므로 폰트를 줄이거나 각도 조절
#     plt.xticks(fontsize=10)
#     plt.yticks(fontsize=10)
    
#     # 상단/우측 테두리 제거로 깔끔하게 정리
#     sns.despine()

#     plt.tight_layout()
#     # 고해상도 저장
#     save_path = "feature_deviation_deciles.pdf"
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()
    
#     print(f"Analysis complete. Plot saved to {save_path}")
#     return all_norms, all_feature_distances


def train_clean(model,train_loader,optimizer, origin_model,epoch, config, scheduler, exp_avg = None):
    XENT_loss = nn.CrossEntropyLoss()
    model.train()
    for batch_idx, (x,y) in enumerate(train_loader):
        optimizer.zero_grad()
        x,y = x.cuda(), y.cuda()
        output = model(x)
        
        loss = XENT_loss(output,y)

        loss.backward()
        optimizer.step()
        if scheduler !=None:
            scheduler.step()

    return




def train_clean_mixup(model,train_loader,optimizer, origin_model,epoch, config, scheduler, exp_avg = None):
    """Clean training with input mixup (Zhang et al. 2018), to produce a higher-entropy natural
    teacher. lam ~ Beta(a,a); mix x and y within the batch; loss = lam*CE(out,y_a)+(1-lam)*CE(out,y_b).
    a = config.mixup_alpha (default 1.0 = standard CIFAR mixup). Everything else = train_clean."""
    XENT_loss = nn.CrossEntropyLoss()
    a = float(getattr(config, "mixup_alpha", 1.0) or 1.0)
    beta_dist = Beta(torch.tensor(a), torch.tensor(a))
    model.train()
    for batch_idx, (x,y) in enumerate(train_loader):
        optimizer.zero_grad()
        x,y = x.cuda(), y.cuda()
        lam = beta_dist.sample().item()
        index = torch.randperm(x.size(0), device=x.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        output = model(mixed_x)
        loss = lam * XENT_loss(output, y) + (1 - lam) * XENT_loss(output, y[index])

        loss.backward()
        optimizer.step()
        if scheduler != None:
            scheduler.step()

    return


def train_clean_and_plot(model,train_loader,optimizer, origin_model,epoch, config, scheduler, exp_avg = None):
    XENT_loss = nn.CrossEntropyLoss()
    model.train()
    for batch_idx, (x,y) in enumerate(train_loader):
        optimizer.zero_grad()
        x,y = x.cuda(), y.cuda()
        feat, output = model(x, feat = True)
        
        loss = XENT_loss(output,y)

        loss.backward()
        optimizer.step()
        if scheduler !=None:
            scheduler.step()

    return



def train_FAD(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """Fragility-Aware Distillation: temperature scales with teacher's FGSM sensitivity."""
    model.train()
    origin_model.eval()

    annealing = (epoch / config.epochs) ** 2
    decay = annealing * (1 - config.kappa) + config.kappa
    criterion_kl = nn.KLDivLoss(reduction="none")

    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()

        # teacher fragility via FGSM
        x_fgsm = x.detach().clone().requires_grad_(True)
        _, logits_fgsm = origin_model(x_fgsm, feat=True)
        ce_loss = F.cross_entropy(logits_fgsm, y, reduction="sum")
        grad = torch.autograd.grad(ce_loss, x_fgsm)[0]
        x_adv_teacher = (x + config.eps * grad.sign()).clamp(0.0, 1.0).detach()

        # adaptive temperature from teacher feature shift
        with torch.no_grad():
            z_clean, teacher_logits = origin_model(x, feat=True)
            z_adv, _ = origin_model(x_adv_teacher, feat=True)
            fragility = (z_clean - z_adv).abs().sum(dim=1)
            temperature = (config.tau * fragility + config.alpha).unsqueeze(1)
            teacher_logits_smooth = teacher_logits / temperature

        # student adversarial examples + losses
        x_adv_student = inner_loss_only_return(
            model=model, teacher_logits=teacher_logits_smooth, x_natural=x, y=y,
            optimizer=optimizer, step_size=config.step_size, epsilon=config.eps,
            perturb_steps=config.steps,
        )
        adv_logits = model(x_adv_student)
        clean_logits = model(x)

        kd_loss = criterion_kl(
            F.log_softmax(adv_logits, dim=1), F.softmax(teacher_logits_smooth.detach(), dim=1)
        ).sum(dim=1).mean()
        consistency_loss = criterion_kl(
            F.log_softmax(adv_logits, dim=1), F.softmax(clean_logits, dim=1)
        ).mean()
        loss = kd_loss + config.lamda * consistency_loss

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if config.weight_avg:
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 - decay) * value + decay * exp_avg[key]


def train_FAD_FS(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """FAD + Feature Suppression: channels the teacher finds FGSM-fragile are scaled down
    (via forward_with_score) before the student's own linear head, instead of only
    reweighting via a scalar temperature."""
    model.train()
    origin_model.eval()

    annealing = (epoch / config.epochs) ** 2
    decay = annealing * (1 - config.kappa) + config.kappa
    beta = getattr(config, "beta", 1.0)
    criterion_kl = nn.KLDivLoss(reduction="none")

    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()

        # teacher fragility via FGSM
        x_fgsm = x.detach().clone().requires_grad_(True)
        _, logits_fgsm = origin_model(x_fgsm, feat=True)
        ce_loss = F.cross_entropy(logits_fgsm, y, reduction="sum")
        grad = torch.autograd.grad(ce_loss, x_fgsm)[0]
        x_adv_teacher = (x + config.eps * grad.sign()).clamp(0.0, 1.0).detach()

        # per-channel fragility -> suppression score (fragile channels -> score near 0)
        with torch.no_grad():
            z_clean, teacher_logits = origin_model(x, feat=True)
            z_adv, _ = origin_model(x_adv_teacher, feat=True)
            channel_fragility = (z_clean - z_adv).abs()
            score = torch.exp(-beta * channel_fragility)

            fragility = channel_fragility.sum(dim=1)
            temperature = (config.tau * fragility + config.alpha).unsqueeze(1)
            teacher_logits_smooth = teacher_logits / temperature

        # student adversarial examples + suppressed-feature losses
        x_adv_student = inner_loss_only_return(
            model=model, teacher_logits=teacher_logits_smooth, x_natural=x, y=y,
            optimizer=optimizer, step_size=config.step_size, epsilon=config.eps,
            perturb_steps=config.steps,
        )
        _, adv_logits = model.forward_with_score(x_adv_student, score)
        _, clean_logits = model.forward_with_score(x, score)

        kd_loss = criterion_kl(
            F.log_softmax(adv_logits, dim=1), F.softmax(teacher_logits_smooth.detach(), dim=1)
        ).sum(dim=1).mean()
        consistency_loss = criterion_kl(
            F.log_softmax(adv_logits, dim=1), F.softmax(clean_logits, dim=1)
        ).mean()
        loss = kd_loss + config.lamda * consistency_loss

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if config.weight_avg:
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 - decay) * value + decay * exp_avg[key]


def train_DPFAT_adaptive_constant(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    model.train()

    origin_model.eval()
    annealing =  (epoch/config.epochs)**2
    decay = annealing * (1 - config.kappa) + config.kappa
    criterion_kl = nn.KLDivLoss(reduction='none')

    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        N, C, H, W = x.shape

        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()
        teacher_feat, teacher_logits = origin_model(x, feat = True)
        teacher_logits = teacher_logits/((config.tau * teacher_feat.norm(dim = 1).reshape([-1,1])) + config.alpha)
        x_pgd = inner_loss_only_return(model, teacher_logits, x, y, optimizer, config.step_size, config.eps, perturb_steps=config.steps)
        plus_logits = model(x_pgd)

        kl_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(teacher_logits.detach(), dim=1))
        loss = (1.0 / N) * (kl_loss.sum(dim = 1)).sum()

        student_logits = model(x)
        consistency_loss  = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(student_logits, dim=1))        
        loss +=  config.lamda * (consistency_loss).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()
             
        if config.weight_avg == True:
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 -decay) * value + decay * exp_avg[key]



def train_DPFAT_adaptive(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    model.train()

    origin_model.eval()
    annealing =  (epoch/config.epochs)**2
    decay = annealing * (1 - config.kappa) + config.kappa
    criterion_kl = nn.KLDivLoss(reduction='none')

    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        N, C, H, W = x.shape

        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()
        teacher_feat, teacher_logits = origin_model(x, feat = True)
        teacher_logits = teacher_logits/((config.tau * teacher_feat.norm(dim = 1).reshape([-1,1])) + config.alpha)
        x_pgd = inner_loss_only_return(model, teacher_logits, x, y, optimizer, config.step_size, config.eps, perturb_steps=config.steps)
        plus_logits = model(x_pgd)

        kl_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(teacher_logits.detach(), dim=1))
        loss = (1.0 / N) * (kl_loss.sum(dim = 1)).sum()

        student_logits = model(x)
        consistency_loss  = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(student_logits, dim=1))        
        loss += annealing * config.lamda * (consistency_loss).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()
             
        if config.weight_avg == True:
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 -decay) * value + decay * exp_avg[key]


def rectify_swap(logits, y):
    """Teacher rectification (PARAMETER-FREE): make the true class the top logit.

    For samples the teacher already gets right (argmax==y) this is a no-op. For wrong samples it
    SWAPS logit[y] with logit[argmax] -> true class becomes top while the soft distribution's shape
    (confidence magnitude / entropy) is preserved. Uses the label, so any baseline it is compared
    against must apply the same rectification (fairness). Apply BEFORE the global temperature /T.
    """
    out = logits.clone()
    idx = torch.arange(logits.size(0), device=logits.device)
    amax = logits.argmax(dim=1)
    y_val = out[idx, y].clone()
    out[idx, y] = out[idx, amax]
    out[idx, amax] = y_val
    return out


def rectify_soft(logits, y, margin):
    """Gentle teacher rectification: LIFT the true-class logit to (current max + margin) ONLY where the
    teacher is wrong (argmax != y); leave every other logit untouched. Unlike hard rectify_swap (which
    demotes the wrong top class to y's low value, erasing the confusion structure), this keeps the whole
    teacher distribution and just floats y to the top by `margin` -> the previously-top wrong class stays
    #2 (dark-knowledge / confusion structure preserved). margin in logit space, applied BEFORE /T."""
    out = logits.clone()
    idx = torch.arange(logits.size(0), device=logits.device)
    cur_max = out.max(dim=1).values
    need = out[idx, y] < cur_max                                   # only where y is not already top
    out[idx, y] = torch.where(need, cur_max + margin, out[idx, y])
    return out


def train_temperature_swapsoft(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """Global temperature + SOFT swap-rectification (rectify_soft): float true class to top by `margin`
    while preserving the teacher's confusion structure. Refinement of train_temperature_swap (hard swap).
    knobs: tau = global temperature, beta = margin (logit space)."""
    model.train()
    origin_model.eval()
    annealing = (epoch / config.epochs) ** 2
    decay = annealing * (1 - config.kappa) + config.kappa
    criterion_kl = nn.KLDivLoss(reduction='none')
    margin = getattr(config, "beta", 1.0) or 1.0

    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        N, C, H, W = x.shape
        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()

        with torch.no_grad():
            _, teacher_logits = origin_model(x, feat=True)
            target = (rectify_soft(teacher_logits, y, margin) / config.tau).detach()   # soft rectify THEN soften

        x_pgd = inner_loss_only_return(model, target, x, y, optimizer, config.step_size, config.eps, perturb_steps=config.steps)
        plus_logits = model(x_pgd)
        kl_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(target, dim=1))
        loss = (1.0 / N) * (kl_loss.sum(dim=1)).sum()

        student_logits = model(x)
        if config.lamda is not None and config.lamda > 0:
            consistency_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(student_logits, dim=1))
            loss += annealing * config.lamda * (consistency_loss).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()

        if config.weight_avg == True:
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 - decay) * value + decay * exp_avg[key]


def train_temperature_swap(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """Baseline (global temperature) + teacher swap-rectification. Fair swap-baseline for the carve method.

    Same as train_temperature (student normalizes, teacher target = teacher_logits / tau) but the raw
    teacher logits are first rectify_swap'd so the true class is the top logit. tau = global temperature.
    """
    model.train()
    origin_model.eval()
    annealing = (epoch / config.epochs) ** 2
    decay = annealing * (1 - config.kappa) + config.kappa
    criterion_kl = nn.KLDivLoss(reduction='none')

    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        N, C, H, W = x.shape
        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()

        with torch.no_grad():
            _, teacher_logits = origin_model(x, feat=True)
            target = (rectify_swap(teacher_logits, y) / config.tau).detach()   # rectify THEN soften

        x_pgd = inner_loss_only_return(model, target, x, y, optimizer, config.step_size, config.eps, perturb_steps=config.steps)
        plus_logits = model(x_pgd)
        kl_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(target, dim=1))
        loss = (1.0 / N) * (kl_loss.sum(dim=1)).sum()

        student_logits = model(x)
        if config.lamda is not None and config.lamda > 0:
            consistency_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(student_logits, dim=1))
            loss += annealing * config.lamda * (consistency_loss).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()

        if config.weight_avg == True:
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 - decay) * value + decay * exp_avg[key]


def train_temperature_tadapt(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """train_temperature_swap + a LEARNED per-sample teacher temperature T(x) (teacher-side analog of padapt).

    T(x) = tau * exp(r_centered),  r_centered = clamp(T_head(Phi_teacher.detach()), -c, c) - batch_mean(clamp(...))
    T_head (model.T_head, zero-init, set up in get_model when config.t_adapt) lives on the STUDENT so it
    shares the optimizer; its input is the FROZEN teacher's raw feature (detached), so gradients only ever
    update T_head, never the teacher. Zero-init -> r=0 everywhere -> T(x)=tau exactly at step 0, matching
    the fixed swap baseline. The clamp bounds T(x) to [tau/2, 2*tau] (blocks the kappa(x) near-zero-temp
    divergence failure seen before). Centering r by its own batch mean BEFORE the exp pins the batch's
    GEOMETRIC-MEAN temperature at tau on every single step: the KD loss can only reshuffle which samples
    get a harder/softer target, it can never lower the whole batch's softness to trivially minimize itself
    (the "T->inf flattens every target toward uniform" collapse mode a free-running learned T would have).
    The adversarial search uses a DETACHED copy of the target (the attack doesn't need to move T_head);
    only the outer KD loss backprops into T_head. Logs per-epoch T(x) mean/std/p5/p50/p95 (same observable
    padapt used for p(x)) to see whether it collapses back to flat (T(x)==tau, reconfirms uniqueness) or
    finds real per-sample structure.
    """
    model.train()
    origin_model.eval()
    annealing = (epoch / config.epochs) ** 2
    decay = annealing * (1 - config.kappa) + config.kappa
    criterion_kl = nn.KLDivLoss(reduction='none')
    clamp_range = getattr(config, "t_clamp", None) or 0.6931471805599453  # ln(2) -> T(x) in [tau/2, 2*tau]

    T_all = []
    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        N, C, H, W = x.shape
        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()

        with torch.no_grad():
            teacher_feat, teacher_logits = origin_model(x, feat=True)
            teacher_logits = rectify_swap(teacher_logits, y)

        r = model.T_head(teacher_feat.detach())              # [N,1], zero-init -> r=0
        r = torch.clamp(r, -clamp_range, clamp_range)
        r = r - r.mean()                                      # batch-center in log-space: pins geomean(T)=tau
        T_x = config.tau * torch.exp(r)                       # [N,1]
        target = teacher_logits / T_x                         # differentiable wrt T_head only

        x_pgd = inner_loss_only_return(model, target.detach(), x, y, optimizer, config.step_size, config.eps, perturb_steps=config.steps)
        plus_logits = model(x_pgd)
        kl_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(target, dim=1))
        loss = (1.0 / N) * (kl_loss.sum(dim=1)).sum()

        student_logits = model(x)
        if config.lamda is not None and config.lamda > 0:
            consistency_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(student_logits, dim=1))
            loss += annealing * config.lamda * (consistency_loss).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()

        if config.weight_avg == True:
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 - decay) * value + decay * exp_avg[key]

        T_all.append(T_x.detach().cpu())

    if T_all:
        T_cat = torch.cat(T_all).flatten()
        q = torch.quantile(T_cat, torch.tensor([0.05, 0.5, 0.95]))
        logging.info({"T_mean": round(T_cat.mean().item(), 4), "T_std": round(T_cat.std().item(), 4),
                      "T_p5": round(q[0].item(), 4), "T_p50": round(q[1].item(), 4),
                      "T_p95": round(q[2].item(), 4), "epoch": epoch})


def train_temperature_taunet(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """Interpretable LEARNED per-sample teacher temperature (user's idea, priority #3, 2026-07-04).

    tau(x) comes from a tiny MLP (model.tau_net, see utils.TauNet) fed ONLY 3 interpretable logit
    statistics -- no raw feature vector at all (unlike train_temperature_tadapt, which reads the
    teacher's raw 512-d feature through a linear head):
        norm = ||teacher_logits||_2, margin = top1 - top2, entropy = H(softmax(teacher_logits))
    tau(x) = config.tau * exp(r_centered), r_centered = clamp(tau_net(stats), -c, c) - batch_mean(clamp(...)).
    SAME anti-collapse mechanism as train_temperature_tadapt: nn.KLDivLoss's target-entropy term
    (sum q*log(q)) is NOT constant once tau is learned, so without batch centering the optimizer can
    trivially shrink the loss by flattening every sample's target (raising tau for the whole batch)
    with zero relation to distillation quality -- confirmed empirically (an uncentered version
    collapsed to its clamp ceiling within 1 epoch). Centering pins geomean(tau(x))==config.tau every
    step, so only RELATIVE per-sample softening/sharpening can move the loss.
    NO extra loss term: tau_net is trained purely by the ordinary KD loss's gradient (same mechanism
    as padapt/t_adapt). Final layer zero-init -> tau(x)==config.tau for every sample at step 0 --
    exact match to the fixed-temperature baseline before any learning happens.
    Logs per-epoch tau(x) mean/std/p5/p50/p95 (same observable as padapt/t_adapt) to see whether tau
    collapses back to a constant (reconfirms the session's uniqueness finding) or finds real
    per-sample structure.
    """
    model.train()
    origin_model.eval()
    annealing = (epoch / config.epochs) ** 2
    decay = annealing * (1 - config.kappa) + config.kappa
    criterion_kl = nn.KLDivLoss(reduction='none')
    clamp_range = getattr(config, "t_clamp", None) or 0.6931471805599453  # ln(2) -> tau in [tau/2, 2*tau]

    tau_all = []
    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        N, C, H, W = x.shape
        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()

        with torch.no_grad():
            _, teacher_logits = origin_model(x, feat=True)
            norm = teacher_logits.norm(dim=1, keepdim=True)
            top2 = teacher_logits.topk(2, dim=1).values
            margin = (top2[:, 0] - top2[:, 1]).unsqueeze(1)
            probs = F.softmax(teacher_logits, dim=1)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1, keepdim=True)
            stats = torch.cat([norm, margin, entropy], dim=1)

        r = model.tau_net(stats)                      # [N,1], differentiable wrt tau_net only
        r = torch.clamp(r, -clamp_range, clamp_range)
        r = r - r.mean()                              # batch-center in log-space: pins geomean(tau)=config.tau
        tau_x = config.tau * torch.exp(r)
        target = teacher_logits / tau_x

        x_pgd = inner_loss_only_return(model, target.detach(), x, y, optimizer, config.step_size, config.eps, perturb_steps=config.steps)
        plus_logits = model(x_pgd)
        kl_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(target, dim=1))
        loss = (1.0 / N) * (kl_loss.sum(dim=1)).sum()

        student_logits = model(x)   # clean forward: updates student BN running stats
        if config.lamda is not None and config.lamda > 0:
            consistency_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(student_logits, dim=1))
            loss += annealing * config.lamda * (consistency_loss).mean()

        loss.backward()
        if epoch == 0 and batch_idx == 0:
            g = model.tau_net.net[-1].weight.grad
            logging.info({"tau_net_first_batch_grad_abs_mean": g.abs().mean().item() if g is not None else None})
        optimizer.step()
        scheduler.step()

        if config.weight_avg == True:
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 - decay) * value + decay * exp_avg[key]

        tau_all.append(tau_x.detach().cpu())

    if tau_all:
        tau_cat = torch.cat(tau_all).flatten()
        q = torch.quantile(tau_cat, torch.tensor([0.05, 0.5, 0.95]))
        logging.info({"tau_mean": round(tau_cat.mean().item(), 4), "tau_std": round(tau_cat.std().item(), 4),
                      "tau_p5": round(q[0].item(), 4), "tau_p50": round(q[1].item(), 4),
                      "tau_p95": round(q[2].item(), 4), "epoch": epoch})


def _pgd_attack_true_label(model, x_natural, y, step_size, epsilon, perturb_steps):
    """Plain Madry-style PGD maximizing CE against the TRUE label (not a KL vs. a teacher target) --
    used only for the bilevel meta-batch attack, since the meta objective must use ground truth to
    avoid re-opening the tau-can-game-the-ruler loophole (see train_temperature_taunet_bilevel)."""
    was_training = model.training
    model.eval()
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_ce = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss_ce, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    if was_training:
        model.train()
    return x_adv.detach()


def _entropy_match(u, H_target, beta_lo=0.05, beta_hi=20.0, n_bisect=12, n_newton=2):
    """Per-sample inverse temperature beta s.t. H(softmax(beta*u)) == H_target; returns beta[:,None]*u.

    Used by train_temperature_deltanet_bilevel to pin every EDITED target's entropy to the plain
    tau-baseline target's entropy, so the delta edit can only MOVE probability mass between classes
    (direction), never sharpen/soften the target overall. Sharpening is the bilevel estimator's
    confirmed cheat axis: a sharper target inflates the inner-loss gradient, so the one-step proxy
    takes a bigger step and looks better to L_after regardless of transferability -- the globaltau
    run collapsed through exactly that hole, and _normalized_proxy_step's attempt to fix it at the
    proxy level backfired (see the REVERTED note in train_temperature_taunet_bilevel). This closes
    the hole at the PARAMETRIZATION level instead: sharpness is not in delta's reachable set at all.

    H(beta) = logZ - beta*E_p[u] is strictly decreasing in beta (dH/dbeta = -beta*Var_p(u) < 0), so:
    bracketing bisection under no_grad, then n_newton DIFFERENTIABLE Newton steps from the bracketed
    point -- gradients w.r.t. u flow through the Newton refinement, so delta_net's gradient SEES the
    constraint (a detached solve would let the optimizer chase sharpness directions that the forward
    pass silently cancels, wasting the norm budget on a dead axis).

    u: [N,C] edited logits. H_target: [N] target entropies (must be attainable, i.e. < log C, which
    holds since it comes from an actual softmax). At delta==0, u equals the baseline target and the
    solver returns beta==1 exactly (Newton fixed point), preserving the zero-init fair start.
    """
    def _H_and_var(beta):
        p = F.softmax(beta.unsqueeze(1) * u, dim=1)
        logp = torch.log(p.clamp_min(1e-12))
        H = -(p * logp).sum(dim=1)
        Eu = (p * u).sum(dim=1)
        var = (p * u * u).sum(dim=1) - Eu * Eu
        return H, var

    N = u.shape[0]
    with torch.no_grad():
        lo = u.new_full((N,), beta_lo)
        hi = u.new_full((N,), beta_hi)
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            H_mid, _ = _H_and_var(mid)
            too_soft = H_mid > H_target        # H decreasing in beta: still too soft -> raise beta
            lo = torch.where(too_soft, mid, lo)
            hi = torch.where(too_soft, hi, mid)
        beta = 0.5 * (lo + hi)
    for _ in range(n_newton):
        H, var = _H_and_var(beta)
        dH = -(beta * var).clamp_min(1e-8)     # dH/dbeta, strictly negative; floor for stability
        beta = (beta - (H - H_target) / dH).clamp(beta_lo, beta_hi)
    return beta.unsqueeze(1) * u


def _normalized_proxy_step(backbone_named, grads, lr_now):
    """Build the differentiable proxy backbone update with a FIXED overall step norm (user's idea,
    2026-07-05): without this, a sharper tau produces a bigger KL(target||student) and hence a bigger
    proxy_grads magnitude, so the proxy step is simply LARGER -- L_after's one-step lookahead then
    confounds "sharper tau took a bigger step" with "sharper tau is genuinely better", since a bigger
    step usually looks like more progress early in training regardless of direction. Normalizing the
    WHOLE gradient vector to a fixed unit norm before scaling by lr_now removes this confound: tau can
    only influence the proxy update's DIRECTION now, not its magnitude. The norm itself is DETACHED so
    backprop into tau flows only through the direction, not through "how tau affects the norm".
    """
    valid = [g for g in grads if g is not None]
    total_norm = torch.sqrt(sum((g ** 2).sum() for g in valid) + 1e-12).detach()
    new_params = {}
    for (n, p), g in zip(backbone_named, grads):
        if g is None:
            new_params[n] = p
        else:
            new_params[n] = p - lr_now * (g / total_norm)
    return new_params


def train_temperature_taunet_bilevel(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """Bilevel/meta-learning TauNet (user's idea, 2026-07-04) -- fixes the DEEPER collapse mode of
    train_temperature_taunet: even after batch log-centering blocks the "inflate everyone" cheat,
    tau_net's gradient still only reflects "does nudging tau(x) make the (frozen) x_adv easier for the
    CURRENT student", which can reward curve-fitting to the student's CURRENT idiosyncratic weaknesses
    rather than a genuinely transferable per-sample signal. Fix (Ren et al. 2018 "Learning to Reweight
    Examples" / Meta-Weight-Net, adapted from sample-weight to temperature):

      1. A DIFFERENTIABLE PROXY step on the backbone: proxy_theta = theta - lr*grad(inner_robust_KD_loss,
         create_graph=True, retain_graph=True). This is ONLY used to build a differentiable path from
         tau_net into "which direction would this batch push the backbone" -- it is a plain-SGD finite
         step (standard in bilevel/meta-learning, e.g. Ren et al. 2018, regardless of what the REAL/outer
         optimizer is) and does NOT have to match AdamW's real dynamics. The REAL backbone update happens
         SEPARATELY, later, via a normal inner_loss.backward() + optimizer.step() (full AdamW) reusing
         the SAME retained graph -- FOUND EMPIRICALLY (2026-07-05) that routing the REAL update through
         this raw-SGD proxy instead starves the backbone: AdamW's adaptive per-parameter normalization
         produces a much larger effective step than raw SGD at the same nominal OneCycleLR value, so a
         raw-SGD-only backbone barely moves (confirmed: epoch 0-5 eval showed clean/rob accuracy
         IDENTICAL to the untouched natural teacher, ~74/0 -- the backbone was essentially frozen).
      2. L_after = CE(functional_call(model, proxy_theta, x_meta_adv), y_meta) on a HELD-OUT meta batch
         (CIFAR100's val split, config.val=True, 45000/5000, never seen by train_loader) using the TRUE
         label and a FRESH plain PGD attack (_pgd_attack_true_label, current real weights, not part of
         the meta graph) -- deliberately NOT the tau-parameterized KD/soft target, or tau could game
         this ruler exactly like the original entropy-collapse bug. Per user's explicit decision: robust
         (adversarial) meta-loss, not clean-only -- the main loss is already about adversarial training,
         so the meta-objective stays consistent with that instead of switching to clean for convenience.
      3. tau_grads = autograd.grad(L_after, tau_net.parameters()) -- ONLY tau_net gets this gradient.
         tau_net is stepped with its OWN plain SGD optimizer (config.tau_meta_lr, default 100), NOT
         the shared AdamW -- FOUND EMPIRICALLY (2026-07-05): routing tau_net's tiny (~1e-5) meta-gradient
         through AdamW causes runaway growth, because AdamW normalizes step size by the gradient's OWN
         magnitude, so even a consistently-tiny-but-same-signed gradient gets a nearly full-LR-sized
         step every batch regardless of how small it actually is. Within ~15-20 meta-phase batches this
         inflated tau_net's raw output far past the log-centering clamp range, saturating EVERY sample
         to the same clamp boundary -> after batch-centering that's identically 0 for everyone -> total
         collapse (confirmed via scratchpad/diag_bilevel_collapse.py: AdamW showed weight-norm growing
         every single step with no equilibrium; plain SGD at lr=100 keeps weight-norm essentially flat
         while still letting real, bounded per-sample dispersion develop gradually).
      4. Efficiency (user's idea): this expensive path (2nd-order + meta batch + functional_call) only
         runs for epoch in [config.bilevel_start, config.bilevel_end) -- a MID-training window, not
         epoch 0 (student is undertrained then, so "what helps the student" is a moving/unstable
         target). Outside the window tau_net gets ZERO gradient (forward-only, wrapped in
         torch.no_grad()) -- before the window it's still zero-init (tau(x)==config.tau exactly,
         matching the plain baseline); after the window it's frozen at whatever the meta phase found.

    Requires config.val=True (dataset.py carves a genuine held-out 5000-image split, never touched by
    train_loader) and config.tau_adapt=True (attaches model.tau_net via get_model, same as the other
    taunet variants). Reuses the SAME batch log-centering reparam as train_temperature_taunet for
    tau(x) itself, so the original entropy-collapse fix stays in place regardless of the meta objective
    layered on top.
    """
    model.train()
    origin_model.eval()
    annealing = (epoch / config.epochs) ** 2
    decay = annealing * (1 - config.kappa) + config.kappa
    criterion_kl = nn.KLDivLoss(reduction='none')
    clamp_range = getattr(config, "t_clamp", None) or 0.6931471805599453
    _tc = getattr(config, "tau_center", None)
    tau_center = bool(_tc) if _tc is not None else True  # False -> let the batch's global tau level move too
    _bs = getattr(config, "bilevel_start", None)
    _be = getattr(config, "bilevel_end", None)
    bilevel_start = int(_bs) if _bs is not None else 10
    bilevel_end = int(_be) if _be is not None else 20
    in_meta_phase = bilevel_start <= epoch < bilevel_end

    if in_meta_phase and not hasattr(train_temperature_taunet_bilevel, "_meta_loader"):
        _, meta_loader, _ = getattr(_dataset_mod, config.dataset)(
            root=os.path.join(config.data_root, config.dataset), download=False,
            batch_size=config.batch_size, val=True, config=config)
        train_temperature_taunet_bilevel._meta_loader = meta_loader
        train_temperature_taunet_bilevel._meta_iter = iter(meta_loader)
        _tml = getattr(config, "tau_meta_lr", None)
        tau_meta_lr = float(_tml) if _tml is not None else 100.0
        train_temperature_taunet_bilevel._tau_optimizer = torch.optim.SGD(model.tau_net.parameters(), lr=tau_meta_lr)

    if epoch == bilevel_end:
        model.tau_net.requires_grad_(False)

    tau_all = []
    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        N, C, H, W = x.shape
        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()

        # teacher input-gradient norm at the clean point (2026-07-05: replaces `margin`, which was
        # redundant with entropy, corr~0.99 -- see dpfat-signal-ablation memory). gradnorm alone was the
        # best teacher-only signal found this session (H=41.97 @ gamma=0.3, beats baseline/taunet_v2).
        x_req = x.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            _, z_req = origin_model(x_req, feat=True)
            ce = F.cross_entropy(z_req, y)
        g_in = torch.autograd.grad(ce, x_req)[0]
        gradnorm = g_in.flatten(1).norm(dim=1, keepdim=True)

        with torch.no_grad():
            _, teacher_logits = origin_model(x, feat=True)
            norm = teacher_logits.norm(dim=1, keepdim=True)
            probs = F.softmax(teacher_logits, dim=1)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1, keepdim=True)
            stats = torch.cat([norm, entropy, gradnorm], dim=1)

        with torch.no_grad() if not in_meta_phase else torch.enable_grad():
            r = model.tau_net(stats)
            r = torch.clamp(r, -clamp_range, clamp_range)
            if tau_center:
                r = r - r.mean()   # pins geomean(tau)==config.tau every batch -- see docstring for why
                                    # this is skippable under bilevel (config.tau_center=False): the
                                    # entropy-reward-hacking loophole this blocks is specific to training
                                    # tau_net off the shared (gameable) KD loss; bilevel's L_after is a
                                    # genuine held-out CE against true labels with no such free-lunch
                                    # direction, so the global level is free to move if that's what helps.
            tau_x = config.tau * torch.exp(r)
        target = teacher_logits / tau_x

        x_pgd = inner_loss_only_return(model, target.detach(), x, y, optimizer, config.step_size, config.eps, perturb_steps=config.steps)

        if in_meta_phase:
            plus_logits = model(x_pgd)
            kl_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(target, dim=1))
            inner_loss = (1.0 / N) * (kl_loss.sum(dim=1)).sum()
            with torch.no_grad():
                model(x)   # clean forward: updates student BN running stats, matches every other variant

            backbone_named = [(n, p) for n, p in model.named_parameters() if not n.startswith("tau_net.")]
            backbone_params = [p for _, p in backbone_named]
            # DIFFERENTIABLE PROXY step (retain_graph=True: inner_loss's graph is reused below for the
            # REAL update too). This proxy is a plain-SGD finite step used ONLY to build a differentiable
            # path from tau_net into "how would the backbone move" -- it does NOT have to exactly match
            # AdamW's real dynamics (standard in bilevel/meta-learning: Ren et al. 2018 also uses a plain
            # SGD virtual step regardless of what the real/outer optimizer is). The REAL backbone update
            # happens further below via a normal .backward() + optimizer.step() (full AdamW), so training
            # dynamics match every other (working) variant -- found empirically (2026-07-05) that routing
            # the REAL update through this same raw-SGD proxy starves the backbone: AdamW's adaptive
            # normalization produces a much larger effective step than raw SGD at the same nominal
            # OneCycleLR value, so a raw-SGD-only backbone barely moves (confirmed: epoch 0-5 eval showed
            # clean_acc/rob_acc identical to the untouched natural teacher, ~74/0).
            proxy_grads = torch.autograd.grad(inner_loss, backbone_params, create_graph=True, retain_graph=True, allow_unused=True)
            lr_now = optimizer.param_groups[0]['lr']
            # REVERTED (2026-07-06): normalizing the proxy step's magnitude was meant to remove the
            # "sharper tau -> bigger inner_loss gradient -> bigger one-step proxy update -> looks better
            # to L_after" confound, but empirically it made things WORSE (H~31 vs 41.82, tau_std blew up
            # to ~6 vs ~0.73) -- normalization apparently also removed an unintentional damping effect the
            # raw gradient magnitude provided, letting the underlying "sharper is better" greedy bias run
            # much more freely (same failure mode as the global-scalar collapse, just weaker before this
            # change). Confirmed via an ISOLATED (no GPU contention) rerun reproducing the bad result, so
            # this is a genuine regression from the normalization, not noise. Back to the plain (raw,
            # un-normalized) proxy step that produced the good 41.82 (3-step) / 42.36 (10-step) results.
            proxy_backbone_params = {
                n: (p - lr_now * g if g is not None else p)
                for (n, p), g in zip(backbone_named, proxy_grads)
            }

            try:
                x_m, y_m = next(train_temperature_taunet_bilevel._meta_iter)
            except StopIteration:
                train_temperature_taunet_bilevel._meta_iter = iter(train_temperature_taunet_bilevel._meta_loader)
                x_m, y_m = next(train_temperature_taunet_bilevel._meta_iter)
            x_m, y_m = x_m.cuda(), y_m.cuda()
            x_m_adv = _pgd_attack_true_label(model, x_m, y_m, config.step_size, config.eps, perturb_steps=config.steps)

            # meta_loss variants (user's idea, 2026-07-05): "adv" (default, original) only rewards true
            # held-out ROBUST accuracy; "cleanadv" also rewards clean accuracy (so tau can't buy robustness
            # by trashing clean performance); "featdiff" rewards clean/adv FEATURE consistency directly
            # (a TRADES-flavored invariance signal) instead of classification accuracy at all; "advfeat"
            # combines CE(adv) with the feature-consistency term -- both "does the student still get it
            # right under attack" AND "did the attack move the feature representation" matter.
            meta_loss_kind = getattr(config, "bilevel_meta_loss", None) or "adv"
            was_training = model.training
            model.eval()
            if meta_loss_kind == "featdiff":
                feat_clean, _ = functional_call(model, proxy_backbone_params, (x_m,), kwargs={"feat": True})
                feat_adv, _ = functional_call(model, proxy_backbone_params, (x_m_adv,), kwargs={"feat": True})
                L_after = F.mse_loss(feat_adv, feat_clean)
            elif meta_loss_kind == "advfeat":
                feat_clean, _ = functional_call(model, proxy_backbone_params, (x_m,), kwargs={"feat": True})
                feat_adv, meta_logits_adv = functional_call(model, proxy_backbone_params, (x_m_adv,), kwargs={"feat": True})
                L_after = F.cross_entropy(meta_logits_adv, y_m) + F.mse_loss(feat_adv, feat_clean)
            else:
                meta_logits_adv = functional_call(model, proxy_backbone_params, (x_m_adv,))
                if meta_loss_kind == "cleanadv":
                    meta_logits_clean = functional_call(model, proxy_backbone_params, (x_m,))
                    L_after = F.cross_entropy(meta_logits_adv, y_m) + F.cross_entropy(meta_logits_clean, y_m)
                else:
                    L_after = F.cross_entropy(meta_logits_adv, y_m)
            if was_training:
                model.train()

            tau_net_params = list(model.tau_net.parameters())
            tau_grads = torch.autograd.grad(L_after, tau_net_params, retain_graph=True, allow_unused=True)

            # REAL backbone update: normal backward + full AdamW (matches every other variant's dynamics).
            # inner_loss's graph is still alive (retain_graph=True was used above) -- reuse it here instead
            # of recomputing the forward pass. MUST happen before tau_optimizer.step() below: that call
            # modifies tau_net's parameters in-place, and tau_net is a LEAF inside inner_loss's own graph
            # (via target = teacher_logits/tau_x) -- mutating it before this backward() would corrupt/
            # invalidate the retained graph.
            inner_loss.backward()

            # tau_net moves via the L_after meta-gradient (own plain SGD optimizer), NOT AdamW using
            # inner_loss's gradient -- overwrite whatever inner_loss.backward() just put in tau_net's .grad.
            tau_optimizer = train_temperature_taunet_bilevel._tau_optimizer
            tau_optimizer.zero_grad()
            for p, g in zip(tau_net_params, tau_grads):
                p.grad = g
            tau_optimizer.step()

            # backbone-only AdamW step: tau_net's .grad must be cleared first so the shared optimizer
            # doesn't ALSO move tau_net (it already got its real update above).
            for p in tau_net_params:
                p.grad = None
            optimizer.step()
            scheduler.step()
        else:
            plus_logits = model(x_pgd)
            kl_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(target, dim=1))
            loss = (1.0 / N) * (kl_loss.sum(dim=1)).sum()

            student_logits = model(x)
            if config.lamda is not None and config.lamda > 0:
                consistency_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(student_logits, dim=1))
                loss += annealing * config.lamda * (consistency_loss).mean()

            loss.backward()
            optimizer.step()
            scheduler.step()

        if config.weight_avg == True:
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 - decay) * value + decay * exp_avg[key]

        tau_all.append(tau_x.detach().cpu())

    if tau_all:
        tau_cat = torch.cat(tau_all).flatten()
        q = torch.quantile(tau_cat, torch.tensor([0.05, 0.5, 0.95]))
        logging.info({"tau_mean": round(tau_cat.mean().item(), 4), "tau_std": round(tau_cat.std().item(), 4),
                      "tau_p5": round(q[0].item(), 4), "tau_p50": round(q[1].item(), 4),
                      "tau_p95": round(q[2].item(), 4), "epoch": epoch, "in_meta_phase": in_meta_phase})


def train_temperature_bilevel_globaltau(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """Bilevel-learned SINGLE GLOBAL temperature (user's idea, 2026-07-05): isolates the "can bilevel find
    a better GLOBAL constant than the hand-picked config.tau=16" question from the per-sample-structure
    question. No MLP, no per-sample stats at all -- just ONE scalar parameter (model.log_tau, see
    get_model's tau_global_bilevel branch), tau = exp(log_tau), IDENTICAL for every sample. Otherwise
    reuses the exact same bilevel machinery as train_temperature_taunet_bilevel: a differentiable proxy
    backbone step (used only to build the path from log_tau into "how would the backbone move"), a
    held-out meta batch with a TRUE-label adversarial attack (L_after = CE), log_tau's OWN plain SGD
    optimizer (config.tau_meta_lr, same AdamW-runaway-avoidance finding applies here too), and a REAL
    backbone update via normal backward+AdamW (matches every other variant's training dynamics).
    log_tau is clamped to a wide but finite safety range (tau in [2, 64]) purely to avoid numerical
    blowup -- NOT meant to bind; if it saturates there, that itself is informative (mirrors the earlier
    finding that a per-sample free-mean version collapsed to the tau/2 floor with a narrower [8,32]
    range, which was worse than the hand-picked 16, H=41.34).
    """
    model.train()
    origin_model.eval()
    annealing = (epoch / config.epochs) ** 2
    decay = annealing * (1 - config.kappa) + config.kappa
    criterion_kl = nn.KLDivLoss(reduction='none')
    LOG_TAU_MIN, LOG_TAU_MAX = 0.6931471805599453, 4.1588830833596715  # tau in [2, 64]
    _bs = getattr(config, "bilevel_start", None)
    _be = getattr(config, "bilevel_end", None)
    bilevel_start = int(_bs) if _bs is not None else 10
    bilevel_end = int(_be) if _be is not None else 20
    in_meta_phase = bilevel_start <= epoch < bilevel_end

    if in_meta_phase and not hasattr(train_temperature_bilevel_globaltau, "_meta_loader"):
        _, meta_loader, _ = getattr(_dataset_mod, config.dataset)(
            root=os.path.join(config.data_root, config.dataset), download=False,
            batch_size=config.batch_size, val=True, config=config)
        train_temperature_bilevel_globaltau._meta_loader = meta_loader
        train_temperature_bilevel_globaltau._meta_iter = iter(meta_loader)
        _tml = getattr(config, "tau_meta_lr", None)
        tau_meta_lr = float(_tml) if _tml is not None else 100.0
        train_temperature_bilevel_globaltau._tau_optimizer = torch.optim.SGD([model.log_tau], lr=tau_meta_lr)

    if epoch == bilevel_end:
        model.log_tau.requires_grad_(False)

    tau_all = []
    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        N, C, H, W = x.shape
        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()

        with torch.no_grad():
            _, teacher_logits = origin_model(x, feat=True)

        with torch.no_grad() if not in_meta_phase else torch.enable_grad():
            log_tau_c = torch.clamp(model.log_tau, LOG_TAU_MIN, LOG_TAU_MAX)
            tau_x = torch.exp(log_tau_c)   # scalar, identical for every sample -- broadcasts below
        target = teacher_logits / tau_x

        x_pgd = inner_loss_only_return(model, target.detach(), x, y, optimizer, config.step_size, config.eps, perturb_steps=config.steps)

        if in_meta_phase:
            plus_logits = model(x_pgd)
            kl_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(target, dim=1))
            inner_loss = (1.0 / N) * (kl_loss.sum(dim=1)).sum()
            with torch.no_grad():
                model(x)   # clean forward: updates student BN running stats, matches every other variant

            backbone_named = [(n, p) for n, p in model.named_parameters() if n != "log_tau"]
            backbone_params = [p for _, p in backbone_named]
            proxy_grads = torch.autograd.grad(inner_loss, backbone_params, create_graph=True, retain_graph=True, allow_unused=True)
            lr_now = optimizer.param_groups[0]['lr']
            # REVERTED (2026-07-06): see train_temperature_taunet_bilevel's comment -- normalizing the
            # proxy step made things worse, not better; back to the plain raw-gradient proxy step.
            proxy_backbone_params = {
                n: (p - lr_now * g if g is not None else p)
                for (n, p), g in zip(backbone_named, proxy_grads)
            }

            try:
                x_m, y_m = next(train_temperature_bilevel_globaltau._meta_iter)
            except StopIteration:
                train_temperature_bilevel_globaltau._meta_iter = iter(train_temperature_bilevel_globaltau._meta_loader)
                x_m, y_m = next(train_temperature_bilevel_globaltau._meta_iter)
            x_m, y_m = x_m.cuda(), y_m.cuda()
            x_m_adv = _pgd_attack_true_label(model, x_m, y_m, config.step_size, config.eps, perturb_steps=config.steps)

            was_training = model.training
            model.eval()
            meta_logits_adv = functional_call(model, proxy_backbone_params, (x_m_adv,))
            L_after = F.cross_entropy(meta_logits_adv, y_m)
            if was_training:
                model.train()

            tau_grads = torch.autograd.grad(L_after, [model.log_tau], retain_graph=True, allow_unused=True)

            inner_loss.backward()

            tau_optimizer = train_temperature_bilevel_globaltau._tau_optimizer
            tau_optimizer.zero_grad()
            model.log_tau.grad = tau_grads[0]
            tau_optimizer.step()

            model.log_tau.grad = None
            optimizer.step()
            scheduler.step()
        else:
            plus_logits = model(x_pgd)
            kl_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(target, dim=1))
            loss = (1.0 / N) * (kl_loss.sum(dim=1)).sum()

            student_logits = model(x)
            if config.lamda is not None and config.lamda > 0:
                consistency_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(student_logits, dim=1))
                loss += annealing * config.lamda * (consistency_loss).mean()

            loss.backward()
            optimizer.step()
            scheduler.step()

        if config.weight_avg == True:
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 - decay) * value + decay * exp_avg[key]

        tau_all.append(tau_x.detach().cpu().item())

    if tau_all:
        logging.info({"tau": round(sum(tau_all) / len(tau_all), 4), "epoch": epoch, "in_meta_phase": in_meta_phase})


def train_temperature_tauclass_bilevel(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """Bilevel-learned PER-CLASS temperature vector (user's idea, 2026-07-07): tau_c in R^C, one
    temperature per logit COORDINATE, shared by every sample. Fills the last open cell of the
    {per-sample, per-class} x {scalar, direction} matrix: the per-sample scalar is a temperature
    reparametrization (proven inert), the per-sample vector (DeltaNet) drove delta->0 under batch
    centering, and the uncentered DeltaNet pilot's answer WAS a per-class constant -- but ADDITIVE
    (prior tilt, bought nothing). This tests the multiplicative sibling: dividing each class's logit
    by its own constant CAN reorder classes (a direction edit, unlike the per-sample scalar) and
    acts proportionally to logit magnitude (unlike the additive tilt) -- e.g. selectively flattening
    classes the teacher is habitually overconfident about.

    Parametrization (model.log_tau_c, get_model's tau_classwise branch, zero-init in R^C):
        s = clamp(log_tau_c, +-t_clamp) - mean(clamp(log_tau_c, +-t_clamp))
        tau_c = config.tau * exp(s)          # geomean(tau_c) == config.tau ALWAYS
    The LOG-CENTERING is load-bearing: globaltau collapsed by moving the GLOBAL temperature level
    (the estimator's confirmed sharpness cheat: sharper target -> bigger inner gradient -> bigger
    one-step proxy update -> better-looking L_after). Centering removes the global level from the
    reachable set entirely, so only RELATIVE per-class structure is learnable; clamp keeps any
    single class within e^{+-t_clamp} (~[tau/2, 2*tau] at the default) of the shared level.

    Bilevel machinery copied verbatim from train_temperature_bilevel_globaltau (differentiable
    plain-SGD proxy step -> L_after = true-label adversarial CE on the held-out meta batch
    (config.val=True, 45000/5000 split) -> autograd.grad into log_tau_c only, stepped by its OWN
    plain SGD (config.tau_meta_lr, default 100 -- same AdamW-runaway-avoidance finding); REAL
    backbone update via normal backward+AdamW; meta phase only for epoch in [bilevel_start,
    bilevel_end), default 0-10 (the low-lr OneCycle warmup window per the taunet_bilevel v2
    finding), frozen after).

    Per-epoch logging: dispersion of the realized tau_c vector (std/min/max/p5/p50/p95). Structure
    emerging == std grows through the window; std ~ 0 at freeze == the per-class multiplicative
    axis is inert too (matrix cell closed -- a uniqueness result, not a failure)."""
    model.train()
    origin_model.eval()
    annealing = (epoch / config.epochs) ** 2
    decay = annealing * (1 - config.kappa) + config.kappa
    criterion_kl = nn.KLDivLoss(reduction='none')
    clamp_range = getattr(config, "t_clamp", None) or 0.6931471805599453
    _bs = getattr(config, "bilevel_start", None)
    _be = getattr(config, "bilevel_end", None)
    bilevel_start = int(_bs) if _bs is not None else 0
    bilevel_end = int(_be) if _be is not None else 10
    in_meta_phase = bilevel_start <= epoch < bilevel_end

    if in_meta_phase and not hasattr(train_temperature_tauclass_bilevel, "_meta_loader"):
        _, meta_loader, _ = getattr(_dataset_mod, config.dataset)(
            root=os.path.join(config.data_root, config.dataset), download=False,
            batch_size=config.batch_size, val=True, config=config)
        train_temperature_tauclass_bilevel._meta_loader = meta_loader
        train_temperature_tauclass_bilevel._meta_iter = iter(meta_loader)
        _tml = getattr(config, "tau_meta_lr", None)
        tau_meta_lr = float(_tml) if _tml is not None else 100.0
        train_temperature_tauclass_bilevel._tau_optimizer = torch.optim.SGD([model.log_tau_c], lr=tau_meta_lr)

    if epoch == bilevel_end:
        model.log_tau_c.requires_grad_(False)

    tau_c_last = None
    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        N, C, H, W = x.shape
        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()

        with torch.no_grad():
            _, teacher_logits = origin_model(x, feat=True)

        with torch.no_grad() if not in_meta_phase else torch.enable_grad():
            s = torch.clamp(model.log_tau_c, -clamp_range, clamp_range)
            s = s - s.mean()                     # geomean(tau_c)==config.tau: global level unreachable
            tau_c = config.tau * torch.exp(s)    # [C], broadcasts over the batch below
        target = teacher_logits / tau_c

        x_pgd = inner_loss_only_return(model, target.detach(), x, y, optimizer, config.step_size, config.eps, perturb_steps=config.steps)

        if in_meta_phase:
            plus_logits = model(x_pgd)
            kl_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(target, dim=1))
            inner_loss = (1.0 / N) * (kl_loss.sum(dim=1)).sum()
            with torch.no_grad():
                model(x)   # clean forward: updates student BN running stats, matches every other variant

            backbone_named = [(n, p) for n, p in model.named_parameters() if n != "log_tau_c"]
            backbone_params = [p for _, p in backbone_named]
            proxy_grads = torch.autograd.grad(inner_loss, backbone_params, create_graph=True, retain_graph=True, allow_unused=True)
            lr_now = optimizer.param_groups[0]['lr']
            # plain raw-gradient proxy step (normalized proxy REVERTED 2026-07-06, see taunet_bilevel;
            # the sharpness confound it targeted is closed here by the log-centering instead)
            proxy_backbone_params = {
                n: (p - lr_now * g if g is not None else p)
                for (n, p), g in zip(backbone_named, proxy_grads)
            }

            try:
                x_m, y_m = next(train_temperature_tauclass_bilevel._meta_iter)
            except StopIteration:
                train_temperature_tauclass_bilevel._meta_iter = iter(train_temperature_tauclass_bilevel._meta_loader)
                x_m, y_m = next(train_temperature_tauclass_bilevel._meta_iter)
            x_m, y_m = x_m.cuda(), y_m.cuda()
            x_m_adv = _pgd_attack_true_label(model, x_m, y_m, config.step_size, config.eps, perturb_steps=config.steps)

            was_training = model.training
            model.eval()
            meta_logits_adv = functional_call(model, proxy_backbone_params, (x_m_adv,))
            L_after = F.cross_entropy(meta_logits_adv, y_m)
            if was_training:
                model.train()

            tau_grads = torch.autograd.grad(L_after, [model.log_tau_c], retain_graph=True, allow_unused=True)

            # REAL backbone update: must run before tau_optimizer.step() -- log_tau_c is a leaf inside
            # inner_loss's retained graph (via target); mutating it in-place first would corrupt it.
            inner_loss.backward()

            tau_optimizer = train_temperature_tauclass_bilevel._tau_optimizer
            tau_optimizer.zero_grad()
            model.log_tau_c.grad = tau_grads[0]
            tau_optimizer.step()

            model.log_tau_c.grad = None
            optimizer.step()
            scheduler.step()
        else:
            plus_logits = model(x_pgd)
            kl_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(target, dim=1))
            loss = (1.0 / N) * (kl_loss.sum(dim=1)).sum()

            student_logits = model(x)
            if config.lamda is not None and config.lamda > 0:
                consistency_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(student_logits, dim=1))
                loss += annealing * config.lamda * (consistency_loss).mean()

            loss.backward()
            optimizer.step()
            scheduler.step()

        if config.weight_avg == True:
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 - decay) * value + decay * exp_avg[key]

        tau_c_last = tau_c.detach()

    if tau_c_last is not None:
        t = tau_c_last.float().cpu()
        q = torch.quantile(t, torch.tensor([0.05, 0.5, 0.95]))
        logging.info({"tauc_mean": round(t.mean().item(), 4), "tauc_std": round(t.std().item(), 4),
                      "tauc_min": round(t.min().item(), 4), "tauc_max": round(t.max().item(), 4),
                      "tauc_p5": round(q[0].item(), 4), "tauc_p50": round(q[1].item(), 4),
                      "tauc_p95": round(q[2].item(), 4), "epoch": epoch, "in_meta_phase": in_meta_phase})
        # full per-class vector (index == class id) + the extreme classes, one line per epoch --
        # cheap (~1KB) and lets the WHICH-classes question be answered from the log alone
        srt = torch.argsort(t)
        logging.info({"tauc_bot5_idx": srt[:5].tolist(), "tauc_top5_idx": srt[-5:].tolist(),
                      "tauc_vec": [round(v, 3) for v in t.tolist()], "epoch": epoch})


def train_temperature_deltanet_bilevel(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """Bilevel-learned per-sample VECTOR edit of the teacher target (user's idea, 2026-07-06):
    direction-only DeltaNet. Motivation: every per-sample SCALAR on the teacher's logits is a
    temperature reparametrization and provably inert at T=16 (softmax washout, diag_target_level.py);
    the one axis never explored by a LEARNED mechanism is target DIRECTION (moving probability mass
    between classes) -- hand-crafted swap was the only method that ever touched it. This learns that
    edit: delta = model.delta_net(teacher_logits/tau) (utils.DeltaNet, logits-only input, zero-init).

    Four structural constraints make the edit direction-only and collapse-proof:
      1. RESIDUAL + ZERO-INIT: target0 = z_t/tau; edited = target0 + delta; delta==0 at step 0 ->
         exact plain-baseline start (fair-start convention).
      2. MEAN-CENTERING: delta -= delta.mean(dim=1) -- a constant logit shift is softmax-invariant,
         so that direction is pure wasted capacity; remove it.
      3. PER-SAMPLE NORM CAP: ||delta|| <= config.delta_r (default 0.3; target0's own norm is O(1)
         since tau ~ ||z_t||) -- "slight edit" enforced by construction, not by hope.
      4. ENTROPY MATCHING (the load-bearing one): _entropy_match rescales the edited logits so every
         sample's target entropy equals its plain-baseline target's entropy. The bilevel estimator's
         confirmed cheat ("sharper target -> bigger inner gradient -> bigger one-step proxy step ->
         better-looking L_after"; killed globaltau) is thereby OUTSIDE delta's reachable set: delta
         can only choose WHERE the probability mass sits, never HOW concentrated it is. The solve is
         differentiable (Newton refinement) so delta_net's meta-gradient sees the constraint instead
         of chasing directions the forward pass cancels.

    Bilevel machinery is copied verbatim from train_temperature_taunet_bilevel (differentiable plain-
    SGD proxy step -> L_after = true-label adversarial CE on a held-out meta batch (config.val=True)
    -> autograd.grad into delta_net only; delta_net stepped by its OWN plain SGD (config.delta_meta_lr,
    fallback tau_meta_lr, default 100 -- same AdamW-runaway-avoidance finding); REAL backbone update
    via normal backward+AdamW; meta phase only for epoch in [bilevel_start, bilevel_end), frozen
    after). Window default 0-10 matching the taunet_bilevel v2 finding (mid-training windows sit on
    OneCycleLR's peak and wreck the backbone; halfwindow/fullwindow ablations confirmed worse).

    Per-epoch logging: realized ||delta|| stats + cap-saturation fraction, matched-beta stats (how
    hard entropy matching has to correct), and the swap-rediscovery probe: mean delta on the TRUE
    class for teacher-wrong vs teacher-right samples -- if the meta signal is real, delta should
    push mass toward the true class precisely where the teacher is wrong (labels used for LOGGING
    ONLY, never in the delta path).
    """
    model.train()
    origin_model.eval()
    annealing = (epoch / config.epochs) ** 2
    decay = annealing * (1 - config.kappa) + config.kappa
    criterion_kl = nn.KLDivLoss(reduction='none')
    delta_r = float(getattr(config, "delta_r", None) or 0.3)
    _bs = getattr(config, "bilevel_start", None)
    _be = getattr(config, "bilevel_end", None)
    bilevel_start = int(_bs) if _bs is not None else 0
    bilevel_end = int(_be) if _be is not None else 10
    in_meta_phase = bilevel_start <= epoch < bilevel_end

    if in_meta_phase and not hasattr(train_temperature_deltanet_bilevel, "_meta_loader"):
        _, meta_loader, _ = getattr(_dataset_mod, config.dataset)(
            root=os.path.join(config.data_root, config.dataset), download=False,
            batch_size=config.batch_size, val=True, config=config)
        train_temperature_deltanet_bilevel._meta_loader = meta_loader
        train_temperature_deltanet_bilevel._meta_iter = iter(meta_loader)
        _dml = getattr(config, "delta_meta_lr", None)
        if _dml is None:
            _dml = getattr(config, "tau_meta_lr", None)
        delta_meta_lr = float(_dml) if _dml is not None else 100.0
        train_temperature_deltanet_bilevel._delta_optimizer = torch.optim.SGD(model.delta_net.parameters(), lr=delta_meta_lr)

    if epoch == bilevel_end:
        model.delta_net.requires_grad_(False)

    dnorm_all, beta_all = [], []
    dtrue_wrong_sum, dtrue_right_sum, n_wrong, n_right = 0.0, 0.0, 0, 0
    sat_count, n_total = 0, 0
    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        N, C, H, W = x.shape
        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()

        with torch.no_grad():
            _, teacher_logits = origin_model(x, feat=True)
        target0 = teacher_logits / config.tau
        with torch.no_grad():
            p0 = F.softmax(target0, dim=1)
            H0 = -(p0 * torch.log(p0.clamp_min(1e-12))).sum(dim=1)

        with torch.no_grad() if not in_meta_phase else torch.enable_grad():
            d = model.delta_net(target0.detach())
            d = d - d.mean(dim=1, keepdim=True)                    # softmax shift invariance
            if bool(getattr(config, "delta_center_batch", False)):
                # PILOT FINDING (2026-07-06, diag_delta_direction.py): the unconstrained pilot
                # learned literally ONE global direction (alignment 1.0000, per-sample residual
                # 0.5% of ||delta||) -- a constant class-prior tilt added to every target -- and
                # tied the fair 45k baseline exactly (H 40.56 vs 40.55). Same escape route the
                # taunet entropy-hack used (a batch-uniform move), same medicine: subtract the
                # BATCH-MEAN delta vector so only RELATIVE per-sample structure is representable.
                # Win-either-way: signal survives -> genuinely per-sample; delta -> 0 -> the
                # vector/direction axis joins the scalar axes as provably-inert (uniqueness).
                d = d - d.mean(dim=0, keepdim=True)
            dn = d.norm(dim=1, keepdim=True)
            d = d * (delta_r / dn.clamp_min(1e-12)).clamp(max=1.0)  # ||delta|| <= delta_r
            edited = target0 + d
            target = _entropy_match(edited, H0)

        x_pgd = inner_loss_only_return(model, target.detach(), x, y, optimizer, config.step_size, config.eps, perturb_steps=config.steps)

        if in_meta_phase:
            plus_logits = model(x_pgd)
            kl_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(target, dim=1))
            inner_loss = (1.0 / N) * (kl_loss.sum(dim=1)).sum()
            with torch.no_grad():
                model(x)   # clean forward: updates student BN running stats, matches every other variant

            backbone_named = [(n, p) for n, p in model.named_parameters() if not n.startswith("delta_net.")]
            backbone_params = [p for _, p in backbone_named]
            # plain raw-gradient proxy step (see taunet_bilevel: normalized proxy REVERTED 2026-07-06;
            # here the sharpness confound it targeted is already closed by _entropy_match instead)
            proxy_grads = torch.autograd.grad(inner_loss, backbone_params, create_graph=True, retain_graph=True, allow_unused=True)
            lr_now = optimizer.param_groups[0]['lr']
            proxy_backbone_params = {
                n: (p - lr_now * g if g is not None else p)
                for (n, p), g in zip(backbone_named, proxy_grads)
            }

            try:
                x_m, y_m = next(train_temperature_deltanet_bilevel._meta_iter)
            except StopIteration:
                train_temperature_deltanet_bilevel._meta_iter = iter(train_temperature_deltanet_bilevel._meta_loader)
                x_m, y_m = next(train_temperature_deltanet_bilevel._meta_iter)
            x_m, y_m = x_m.cuda(), y_m.cuda()
            x_m_adv = _pgd_attack_true_label(model, x_m, y_m, config.step_size, config.eps, perturb_steps=config.steps)

            was_training = model.training
            model.eval()
            meta_logits_adv = functional_call(model, proxy_backbone_params, (x_m_adv,))
            L_after = F.cross_entropy(meta_logits_adv, y_m)
            if was_training:
                model.train()

            delta_net_params = list(model.delta_net.parameters())
            delta_grads = torch.autograd.grad(L_after, delta_net_params, retain_graph=True, allow_unused=True)

            # REAL backbone update: normal backward + full AdamW, reusing the retained graph. MUST run
            # before delta_optimizer.step() -- delta_net is a leaf inside inner_loss's graph (via
            # target), and mutating its params in-place first would corrupt the retained graph.
            inner_loss.backward()

            delta_optimizer = train_temperature_deltanet_bilevel._delta_optimizer
            delta_optimizer.zero_grad()
            for p, g in zip(delta_net_params, delta_grads):
                p.grad = g
            delta_optimizer.step()

            # backbone-only AdamW step: clear delta_net's .grad so the shared optimizer doesn't ALSO
            # move it (it already got its real meta update above).
            for p in delta_net_params:
                p.grad = None
            optimizer.step()
            scheduler.step()
        else:
            plus_logits = model(x_pgd)
            kl_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(target, dim=1))
            loss = (1.0 / N) * (kl_loss.sum(dim=1)).sum()

            student_logits = model(x)
            if config.lamda is not None and config.lamda > 0:
                consistency_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(student_logits, dim=1))
                loss += annealing * config.lamda * (consistency_loss).mean()

            loss.backward()
            optimizer.step()
            scheduler.step()

        if config.weight_avg == True:
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 - decay) * value + decay * exp_avg[key]

        with torch.no_grad():
            d_det = d.detach()
            dnorm_all.append(d_det.norm(dim=1).cpu())
            beta_all.append((target.detach().norm(dim=1) / edited.detach().norm(dim=1).clamp_min(1e-12)).cpu())
            sat_count += int((dn.detach().squeeze(1) >= delta_r).sum().item())
            n_total += N
            t_wrong = teacher_logits.argmax(dim=1) != y
            d_true = d_det.gather(1, y.unsqueeze(1)).squeeze(1)
            dtrue_wrong_sum += d_true[t_wrong].sum().item();  n_wrong += int(t_wrong.sum().item())
            dtrue_right_sum += d_true[~t_wrong].sum().item(); n_right += int((~t_wrong).sum().item())

    if dnorm_all:
        dn_cat = torch.cat(dnorm_all)
        beta_cat = torch.cat(beta_all)
        qd = torch.quantile(dn_cat, torch.tensor([0.05, 0.5, 0.95]))
        qb = torch.quantile(beta_cat, torch.tensor([0.05, 0.95]))
        logging.info({"dnorm_mean": round(dn_cat.mean().item(), 4), "dnorm_p5": round(qd[0].item(), 4),
                      "dnorm_p50": round(qd[1].item(), 4), "dnorm_p95": round(qd[2].item(), 4),
                      "sat_frac": round(sat_count / max(n_total, 1), 4),
                      "beta_mean": round(beta_cat.mean().item(), 4), "beta_p5": round(qb[0].item(), 4),
                      "beta_p95": round(qb[1].item(), 4),
                      "dtrue_wrong": round(dtrue_wrong_sum / max(n_wrong, 1), 5),
                      "dtrue_right": round(dtrue_right_sum / max(n_right, 1), 5),
                      "wrong_frac": round(n_wrong / max(n_total, 1), 4),
                      "epoch": epoch, "in_meta_phase": in_meta_phase})


def train_temperature_taunet_swap(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """train_temperature_taunet + rectify_swap (user's idea, 2026-07-04): stack the two independently-
    validated levers -- swap (parameter-free teacher rectification, best known result 42.17 H) and the
    interpretable learned per-sample temperature (taunet alone: 41.51 H, real non-degenerate per-sample
    structure but no win). stats (norm/margin/entropy) are computed from the RAW teacher_logits (same
    as train_temperature_taunet); swap is applied AFTER, only to build the actual target -- mirrors
    train_temperature_tadapt's convention (target = rectify_swap(teacher_logits, y) / T(x)).
    """
    model.train()
    origin_model.eval()
    annealing = (epoch / config.epochs) ** 2
    decay = annealing * (1 - config.kappa) + config.kappa
    criterion_kl = nn.KLDivLoss(reduction='none')
    clamp_range = getattr(config, "t_clamp", None) or 0.6931471805599453

    tau_all = []
    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        N, C, H, W = x.shape
        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()

        with torch.no_grad():
            _, teacher_logits = origin_model(x, feat=True)
            norm = teacher_logits.norm(dim=1, keepdim=True)
            top2 = teacher_logits.topk(2, dim=1).values
            margin = (top2[:, 0] - top2[:, 1]).unsqueeze(1)
            probs = F.softmax(teacher_logits, dim=1)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1, keepdim=True)
            stats = torch.cat([norm, margin, entropy], dim=1)
            swapped_logits = rectify_swap(teacher_logits, y)

        r = model.tau_net(stats)
        r = torch.clamp(r, -clamp_range, clamp_range)
        r = r - r.mean()
        tau_x = config.tau * torch.exp(r)
        target = swapped_logits / tau_x

        x_pgd = inner_loss_only_return(model, target.detach(), x, y, optimizer, config.step_size, config.eps, perturb_steps=config.steps)
        plus_logits = model(x_pgd)
        kl_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(target, dim=1))
        loss = (1.0 / N) * (kl_loss.sum(dim=1)).sum()

        student_logits = model(x)
        if config.lamda is not None and config.lamda > 0:
            consistency_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(student_logits, dim=1))
            loss += annealing * config.lamda * (consistency_loss).mean()

        loss.backward()
        if epoch == 0 and batch_idx == 0:
            g = model.tau_net.net[-1].weight.grad
            logging.info({"tau_net_first_batch_grad_abs_mean": g.abs().mean().item() if g is not None else None})
        optimizer.step()
        scheduler.step()

        if config.weight_avg == True:
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 - decay) * value + decay * exp_avg[key]

        tau_all.append(tau_x.detach().cpu())

    if tau_all:
        tau_cat = torch.cat(tau_all).flatten()
        q = torch.quantile(tau_cat, torch.tensor([0.05, 0.5, 0.95]))
        logging.info({"tau_mean": round(tau_cat.mean().item(), 4), "tau_std": round(tau_cat.std().item(), 4),
                      "tau_p5": round(q[0].item(), 4), "tau_p50": round(q[1].item(), 4),
                      "tau_p95": round(q[2].item(), 4), "epoch": epoch})


def train_temperature(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """Global-temperature-only distillation (clean, minimal).

    Student normalizes its own features (reformation=True / ResNet18_z) -- the confirmed lever.
    Teacher target = raw teacher logits / tau, where tau is a single GLOBAL temperature:
        tau = 1  ==  raw teacher logits (no softening) exactly.
    No carve, no feature normalization, no batch rescale -- tau is the only teacher knob.
    """
    model.train()
    origin_model.eval()
    annealing = (epoch / config.epochs) ** 2
    decay = annealing * (1 - config.kappa) + config.kappa
    criterion_kl = nn.KLDivLoss(reduction='none')

    # AWP (Wu et al. NeurIPS'20) -- optional, config.awp_gamma>0 and epoch>=awp_warmup. Off by
    # default: behavior identical to before this cell. Used as the fair baseline+WA+AWP control
    # for the featdir/k350+WA+AWP cell. config.awp_style selects the mechanics:
    #   "proxy" (default) = utils.AdvWeightPerturb, the original AWP paper's proxy-network port.
    #   "sam"             = utils.AdvWeightPerturbSAM, the ADR-repo port (SAM-style, no proxy
    #                        network, perturbs all params via the model's own gradient).
    # champion-stack knobs, ported from train_feat_direction (2026-08-02) so the no-feature-term
    # ablation can be compared like-for-like against featdir_champ200_100ep. All three are inert
    # when absent, so existing temperature configs are unchanged.
    _eps_train_tmp = float(getattr(config, "train_eps", 0.0) or 0.0) or config.eps
    _freeze_tmp = _resolve_epoch_arg(getattr(config, "freeze_lr_epoch", None), config.epochs)
    _wa_start_tmp = _resolve_epoch_arg(getattr(config, "wa_start", None), config.epochs) or 0

    awp_style = str(getattr(config, "awp_style", "proxy") or "proxy")
    awp_gamma = float(getattr(config, "awp_gamma", 0.0) or 0.0)
    use_awp = awp_gamma > 0 and epoch >= int(getattr(config, "awp_warmup", 0) or 0)
    if use_awp:
        if awp_style == "sam":
            if not hasattr(train_temperature, "_awp_sam"):
                train_temperature._awp_sam = AdvWeightPerturbSAM(model, rho=awp_gamma)
            awp = train_temperature._awp_sam
            awp.rho = awp_gamma
        else:
            if not hasattr(train_temperature, "_awp"):
                train_temperature._awp = AdvWeightPerturb(model, gamma=awp_gamma)
            awp = train_temperature._awp
            awp.gamma = awp_gamma

    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        N, C, H, W = x.shape
        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()

        with torch.no_grad():
            _, teacher_logits = origin_model(x, feat=True)      # raw teacher logits = linear(Phi)
            target = (teacher_logits / config.tau).detach()     # global temperature; tau=1 == raw teacher

        x_pgd = inner_loss_only_return(model, target, x, y, optimizer, config.step_size, _eps_train_tmp, perturb_steps=config.steps)

        def _step_loss():
            pl = model(x_pgd)
            kl = criterion_kl(F.log_softmax(pl, dim=1), F.softmax(target, dim=1))
            l = (1.0 / N) * (kl.sum(dim=1)).sum()
            sl = model(x)   # clean forward: updates student BN running stats (matches DPFAT_adaptive)
            if config.lamda is not None and config.lamda > 0:
                cons = criterion_kl(F.log_softmax(pl, dim=1), F.softmax(sl, dim=1))
                l = l + annealing * config.lamda * cons.mean()
            return l, pl

        if use_awp and awp_style == "sam":
            # ADR/SAM-style: ascent direction from the gradient at CLEAN weights; the gradient
            # the optimizer actually applies is measured at the perturbed weights, then weights
            # are restored to w before optimizer.step() (see AdvWeightPerturbSAM docstring).
            optimizer.zero_grad()
            loss1, _ = _step_loss()
            loss1.backward()
            awp.first_step()
            _awp_bn_disable(model)
            optimizer.zero_grad()
            loss, plus_logits = _step_loss()
            loss.backward()
            awp.restore()
            _awp_bn_enable(model)
            optimizer.step()
        elif use_awp:
            def _awp_loss_fn(pm, _x_pgd=x_pgd, _target=target):
                pl = pm(_x_pgd)
                return (1.0 / N) * criterion_kl(F.log_softmax(pl, dim=1), F.softmax(_target, dim=1)).sum(dim=1).sum()
            awp_diff = awp.calc_awp(_awp_loss_fn)
            awp.perturb(awp_diff)
            loss, plus_logits = _step_loss()
            loss.backward()
            optimizer.step()
            awp.restore(awp_diff)
        else:
            loss, plus_logits = _step_loss()
            loss.backward()
            optimizer.step()

        if _freeze_tmp is None or epoch < _freeze_tmp:
            scheduler.step()
        elif epoch == _freeze_tmp and batch_idx == 0:
            logging.info({"freeze_lr_epoch": _freeze_tmp, "frozen_lr": optimizer.param_groups[0]["lr"]})

        if config.weight_avg == True and epoch >= _wa_start_tmp:
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 - decay) * value + decay * exp_avg[key]


def train_temperature_tauclass_fixed(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """FIXED per-class temperature from precomputed teacher stats -- NO learning (user, 2026-07-07).

    Direct heuristic counterpart of train_temperature_tauclass_bilevel: instead of meta-learning
    tau_c (budget-starved in the 0-10 window; learned direction correlates with nothing,
    |r| < 0.14 vs gnorm/acc/margin/entropy), set it once from a per-class teacher statistic and
    train the plain 50k loop. Everything except the target divisor is train_temperature verbatim.

        tau_c = tau * exp(gamma * (log s_c - mean_c log s_c))

    geomean(tau_c) == tau for ANY gamma (same anchoring as tpnorm/bilevel: global sharpness is
    unreachable, only per-class SHAPE varies). s_c = config.tauclass_stat (default 'gnorm':
    per-class teacher grad-norm, 12.7x class spread, r(gnorm, class acc) = -0.92) read from the
    config.tauclass_stats npz (diag_perclass_teacher.npz). gamma > 0 -> hard/high-gnorm classes get
    SOFTER targets; gamma < 0 -> sharper. gamma = 0 == train_temperature exactly (50k H 41.77 seed0).
    """
    model.train()
    origin_model.eval()
    annealing = (epoch / config.epochs) ** 2
    decay = annealing * (1 - config.kappa) + config.kappa
    criterion_kl = nn.KLDivLoss(reduction='none')

    if not hasattr(train_temperature_tauclass_fixed, "_tau_c"):
        stats_path = getattr(config, "tauclass_stats", None) or "results/CIFAR100/diag_perclass_teacher.npz"
        stat_name = getattr(config, "tauclass_stat", None) or "gnorm"
        gamma = float(config.gamma)
        s = np.load(stats_path)[stat_name].astype(np.float64)
        u = np.log(s) - np.log(s).mean()
        # PLACEBO control (2026-07-08): permute the centered stat across classes -- same tau_c
        # dispersion, difficulty-alignment destroyed. If shuffled ~= gnorm-aligned, the per-class
        # signal is generic jitter, not the difficulty structure.
        shuffled = bool(getattr(config, "tauclass_shuffle", False))
        if shuffled:
            u = np.random.RandomState(int(getattr(config, "tauclass_shuffle_seed", 0) or 0)).permutation(u)
        tau_c = config.tau * np.exp(gamma * u)
        train_temperature_tauclass_fixed._tau_c = torch.tensor(tau_c, dtype=torch.float32).cuda()
        t = train_temperature_tauclass_fixed._tau_c
        logging.info({"tauc_fixed_stat": stat_name, "gamma": gamma, "tauc_shuffle": shuffled,
                      "tauc_mean": round(t.mean().item(), 4), "tauc_std": round(t.std().item(), 4),
                      "tauc_min": round(t.min().item(), 4), "tauc_max": round(t.max().item(), 4),
                      "tauc_vec": [round(v, 3) for v in t.tolist()]})
    tau_c = train_temperature_tauclass_fixed._tau_c

    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        N, C, H, W = x.shape
        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()

        with torch.no_grad():
            _, teacher_logits = origin_model(x, feat=True)
            target = (teacher_logits / tau_c).detach()   # [C] broadcasts over the batch

        x_pgd = inner_loss_only_return(model, target, x, y, optimizer, config.step_size, config.eps, perturb_steps=config.steps)
        plus_logits = model(x_pgd)
        kl_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(target, dim=1))
        loss = (1.0 / N) * (kl_loss.sum(dim=1)).sum()

        student_logits = model(x)   # clean forward: updates student BN running stats
        if config.lamda is not None and config.lamda > 0:
            consistency_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(student_logits, dim=1))
            loss += annealing * config.lamda * (consistency_loss).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()

        if config.weight_avg == True:
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 - decay) * value + decay * exp_avg[key]


def train_temperature_coshead(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """train_temperature verbatim, for the COSINE-HEAD student (config cos_head: True -> ResNet18_zcos):
    logits = s*cos(theta), feature AND classifier weights normalized, s a learnable global scalar.
    Closes the last magnitude channel (AT grows ||w_c|| 5x on the plain norm student). Only addition
    here: log s once per epoch -- if s reproduces the ~5x growth, the channel was purely global
    (cosine head should then TIE = directional-distillation pillar); if the tie breaks, per-class
    ||w_c|| structure mattered. Fair bar: 50k baseline tau16 seed0 H 41.77.
    """
    model.train()
    origin_model.eval()
    annealing = (epoch / config.epochs) ** 2
    decay = annealing * (1 - config.kappa) + config.kappa
    criterion_kl = nn.KLDivLoss(reduction='none')

    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        N, C, H, W = x.shape
        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()

        with torch.no_grad():
            _, teacher_logits = origin_model(x, feat=True)
            target = (teacher_logits / config.tau).detach()

        x_pgd = inner_loss_only_return(model, target, x, y, optimizer, config.step_size, config.eps, perturb_steps=config.steps)
        plus_logits = model(x_pgd)
        kl_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(target, dim=1))
        loss = (1.0 / N) * (kl_loss.sum(dim=1)).sum()

        student_logits = model(x)   # clean forward: updates student BN running stats
        if config.lamda is not None and config.lamda > 0:
            consistency_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(student_logits, dim=1))
            loss += annealing * config.lamda * (consistency_loss).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()

        if config.weight_avg == True:
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 - decay) * value + decay * exp_avg[key]

    log_s = model.encoder.log_s if hasattr(model, "encoder") else model.log_s
    logging.info({"cos_s": round(torch.exp(log_s.detach()).item(), 4), "epoch": epoch})


def train_temperature_costeacher(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """COSINE-TEACHER target (user, 2026-07-07 night): the teacher passes ONLY direction.

        target = s_t * (Phi_t_hat . w_t_hat)   -- both teacher feature AND classifier rows
        L2-normalized, bias dropped;  s_t = mean||Phi_t|| * mean||w_t,c|| / tau  (frozen at first
        batch) so tau=16 matches the raw z_t/tau target's sharpness exactly = fair start, and tau
        keeps its meaning as THE one global temperature (cos is bounded in [-1,1]; unscaled
        softmax(cos) is near-uniform, so a global scale is structurally necessary -- normalization
        does not kill temperature, it BECOMES it).

    vs the raw z_t/tau target this removes exactly: ||Phi_t|| variation (known-inert, iso tie),
    per-class ||w_t,c|| (1.25x spread; a per-class multiplicative rescale = the axis the tauclass
    sweep just closed), and the bias (std 0.02). Completes the cos 2x2 with temp_coshead:
    {student norm, student cos} x {teacher raw/tau, teacher cos} -- same isolation design that
    established student-norm as THE lever. Student arch is orthogonal: this method works for both
    the norm student and the cos_head student (logs cos_s when present). Fair bar: 50k H 41.77.
    """
    model.train()
    origin_model.eval()
    annealing = (epoch / config.epochs) ** 2
    decay = annealing * (1 - config.kappa) + config.kappa
    criterion_kl = nn.KLDivLoss(reduction='none')

    teacher_enc = origin_model.encoder if hasattr(origin_model, "encoder") else origin_model
    w_hat_t = F.normalize(teacher_enc.linear.weight.detach(), dim=1)

    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        N, C, H, W = x.shape
        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()

        with torch.no_grad():
            teacher_feat, _ = origin_model(x, feat=True)
            if not hasattr(train_temperature_costeacher, "_s_t"):
                mean_phi = teacher_feat.norm(dim=1).mean().item()
                mean_w = teacher_enc.linear.weight.detach().norm(dim=1).mean().item()
                train_temperature_costeacher._s_t = mean_phi * mean_w / config.tau
                logging.info({"cos_teacher_s_t": round(train_temperature_costeacher._s_t, 4),
                              "mean_phi_t": round(mean_phi, 4), "mean_w_t": round(mean_w, 4), "tau": config.tau})
            cos_t = F.linear(F.normalize(teacher_feat, dim=1), w_hat_t)
            target = (train_temperature_costeacher._s_t * cos_t).detach()

        x_pgd = inner_loss_only_return(model, target, x, y, optimizer, config.step_size, config.eps, perturb_steps=config.steps)
        plus_logits = model(x_pgd)
        kl_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(target, dim=1))
        loss = (1.0 / N) * (kl_loss.sum(dim=1)).sum()

        student_logits = model(x)   # clean forward: updates student BN running stats
        if config.lamda is not None and config.lamda > 0:
            consistency_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(student_logits, dim=1))
            loss += annealing * config.lamda * (consistency_loss).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()

        if config.weight_avg == True:
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 - decay) * value + decay * exp_avg[key]

    enc = model.encoder if hasattr(model, "encoder") else model
    if hasattr(enc, "log_s"):
        logging.info({"cos_s": round(torch.exp(enc.log_s.detach()).item(), 4), "epoch": epoch})


def _log_gain_stats(model, epoch):
    """Per-epoch gain-head telemetry: effective ||w_s,c|| = exp(log_g_c) * ||w_t,c||."""
    enc = model.encoder if hasattr(model, "encoder") else model
    if not hasattr(enc, "log_g"):
        return
    with torch.no_grad():
        g = enc.log_g.exp()
        eff = g * enc.linear.weight.norm(dim=1)
        logging.info({"g_mean": round(g.mean().item(), 4), "g_std": round(g.std().item(), 4),
                      "g_min": round(g.min().item(), 4), "g_max": round(g.max().item(), 4),
                      "wnorm_eff_mean": round(eff.mean().item(), 4),
                      "wnorm_eff_spread": round((eff.max() / eff.min()).item(), 4), "epoch": epoch})


def _featdir_target_head(Wt, config):
    """Optionally row-normalize the teacher head used to BUILD the featdir head-target.

    featdir_normhead_target: True deletes the teacher's per-class ||w_c|| structure (every row set
    to the same length) while preserving the mean row norm, so the target's overall sharpness --
    and hence its effective temperature -- is unchanged and the cell isolates the *relative*
    per-class weighting rather than confounding it with a global scale change."""
    if not bool(getattr(config, "featdir_normhead_target", False)):
        return Wt
    return F.normalize(Wt, dim=1) * Wt.norm(dim=1).mean()


def _resolve_epoch_arg(val, total_epochs):
    """Schedule-length-portable epoch arg (2026-08-01): a value in (0,1) is a FRACTION of the run,
    >=1 is an absolute epoch index, None/absent/negative -> None (feature off).

    The other server's champion config carried these as absolute ints tuned at one schedule length
    (wa_start 10 at 50ep = 0.2 of the run); replaying the same int at 100ep would land at 0.1 of
    the run instead. Writing 0.2 keeps the RELATIVE position -- which is what the decay schedule
    annealing=(epoch/epochs)^2 actually depends on -- fixed across schedule lengths."""
    if val is None:
        return None
    v = float(val)
    if v < 0:
        return None
    return int(round(v * total_epochs)) if 0 < v < 1 else int(v)


def train_temperature_dirattack(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """train_temperature (baseline KL outer loss) but the INNER attack is the direction-max
    adversary (inner_featdir_only_return) -- completes the {outer: KL/dir} x {inner: KL/dir} 2x2
    (user q, 2026-07-13 evening: "is baseline better because the ATTACK uses the classification
    loss?"). Known cells: KL/KL 41.77, dir/dir 39.93, dir/KL 39.79 (klattack, no recovery ->
    attack side is not the lever). THIS = KL/dir: stays ~41.7 -> the classification-aware
    advantage enters through the TRAINING gradient (span(W_s)-shaped backbone supervision), not
    the adversary; drops -> attack-side matters after all. Loop = train_temperature verbatim
    except the x_pgd line."""
    model.train()
    origin_model.eval()
    annealing = (epoch / config.epochs) ** 2
    decay = annealing * (1 - config.kappa) + config.kappa
    criterion_kl = nn.KLDivLoss(reduction='none')

    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        N, C, H, W = x.shape
        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()

        with torch.no_grad():
            phi_t, teacher_logits = origin_model(x, feat=True)
            phi_t_hat = F.normalize(phi_t, dim=1).detach()
            target = (teacher_logits / config.tau).detach()

        x_pgd = inner_featdir_only_return(model, phi_t_hat, x, optimizer, config.step_size, config.eps, perturb_steps=config.steps)
        plus_logits = model(x_pgd)
        kl_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(target, dim=1))
        loss = (1.0 / N) * (kl_loss.sum(dim=1)).sum()

        student_logits = model(x)   # clean forward: updates student BN running stats
        if config.lamda is not None and config.lamda > 0:
            consistency_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(student_logits, dim=1))
            loss += annealing * config.lamda * (consistency_loss).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()

        if config.weight_avg == True:
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 - decay) * value + decay * exp_avg[key]


def train_temperature_gainhead(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """train_temperature with the GAIN-ONLY student head (gain_head: True -> ResNet18_zgain):
    w_s,c = exp(log_g_c) * w_t,c, direction FROZEN at the teacher head, 100 learnable gains
    (+ free bias). The {direction frozen, ||w_c|| free} cell of the head-side 2x2 -- diagonal
    partner of coshead {direction free, ||w_c|| frozen}, which loses -1.4~-2.7. Fair bars:
    baseline (1) 41.77 3-step / 42.18 10-step, coshead (3) 40.01 (tau16 3-step). Diagnostic basis
    + registered prediction (TIE or small loss): scripts/diag_head_rotation.py, memory 2026-07-13.
    Loop = train_temperature verbatim + per-epoch gain stats."""
    model.train()
    origin_model.eval()
    annealing = (epoch / config.epochs) ** 2
    decay = annealing * (1 - config.kappa) + config.kappa
    criterion_kl = nn.KLDivLoss(reduction='none')

    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        N, C, H, W = x.shape
        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()

        with torch.no_grad():
            _, teacher_logits = origin_model(x, feat=True)
            target = (teacher_logits / config.tau).detach()

        x_pgd = inner_loss_only_return(model, target, x, y, optimizer, config.step_size, config.eps, perturb_steps=config.steps)
        plus_logits = model(x_pgd)
        kl_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(target, dim=1))
        loss = (1.0 / N) * (kl_loss.sum(dim=1)).sum()

        student_logits = model(x)   # clean forward: updates student BN running stats
        if config.lamda is not None and config.lamda > 0:
            consistency_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(student_logits, dim=1))
            loss += annealing * config.lamda * (consistency_loss).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()

        if config.weight_avg == True:
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 - decay) * value + decay * exp_avg[key]

    _log_gain_stats(model, epoch)


def _featdir_class_proto(origin_model, train_loader, config):
    """Unit-norm class prototypes of the frozen teacher's DIRECTION field: c_k = normalize(mean_x
    phi_t_hat(x) | y=k).  Computed once on the first call and cached -- the teacher never changes.
    Averaging the NORMALIZED features (not the raw ones) keeps this a mean of directions, so a few
    large-norm samples cannot dominate the prototype.  Used by featdir_proto_gamma."""
    if getattr(_featdir_class_proto, "_cache", None) is not None:
        return _featdir_class_proto._cache
    n_cls = 100 if config.dataset == 'CIFAR100' else 10
    dev = next(origin_model.parameters()).device
    acc, cnt = None, torch.zeros(n_cls, device=dev)
    with torch.no_grad():
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            f, _ = origin_model(xb, feat=True)
            f = F.normalize(f, dim=1)
            if acc is None:
                acc = torch.zeros(n_cls, f.shape[1], device=dev)
            acc.index_add_(0, yb, f)
            cnt.index_add_(0, yb, torch.ones_like(yb, dtype=acc.dtype))
    C = F.normalize(acc / cnt.clamp(min=1).unsqueeze(1), dim=1)
    # diagnostics: how tight is each class, and how separated are the prototypes
    off = (C @ C.t()) - torch.eye(n_cls, device=dev)
    logging.info({"featdir_class_proto": n_cls,
                  "proto_offdiag_cos_mean": round(off.sum().item() / (n_cls * (n_cls - 1)), 4),
                  "proto_offdiag_cos_max": round(off.max().item(), 4)})
    _featdir_class_proto._cache = C
    return C


def _featdir_etf_frame(origin_model, train_loader, config):
    """TRICK B: simplex-ETF replacement for the teacher's CLASS geometry.

    The natural teacher's class prototypes are far from equiangular -- measured off-diagonal cosine
    on CIFAR100 is mean 0.32 / max 0.66, where a maximally separated frame gives -1/(K-1) = -0.01.
    We build unit vectors E with exactly that pairwise cosine and rotate each class's direction field
    onto them, so the ANGULAR MARGIN between classes is maximized while the instance residual is
    carried along (trick A showed the residual is worth keeping).

    Two deliberate choices:
      * E is built inside span(C), not in a random subspace -- the class information already lives
        there, so this changes separation without relocating the representation (minimal binding).
      * Within that subspace E is Procrustes-aligned to C, i.e. we pick the ETF closest to what the
        teacher already has. Every degree of freedom not needed for equiangularity is left alone.

    A single global rotation cannot do this: orthogonal maps preserve inner products, so class
    separation would be unchanged. The rotation has to be per-class, which is fine -- y is known at
    training time and the teacher is frozen.
    Returns (C, E), both [K, D] with unit rows.
    """
    if getattr(_featdir_etf_frame, "_cache", None) is not None:
        return _featdir_etf_frame._cache
    C = _featdir_class_proto(origin_model, train_loader, config)          # [K, D] unit rows
    K, D = C.shape
    U, _ = torch.linalg.qr(C.t().double())                               # [D, K] basis of span(C)
    M = math.sqrt(K / (K - 1.0)) * (torch.eye(K, dtype=torch.float64, device=C.device)
                                    - torch.ones(K, K, dtype=torch.float64, device=C.device) / K)
    A = U.t() @ C.t().double()                                           # prototypes in U-coords
    W, _, Vh = torch.linalg.svd(A @ M.t())                               # orthogonal Procrustes
    E = (U @ (W @ Vh) @ M).t().float()                                   # [K, D] unit rows
    E = F.normalize(E, dim=1)
    off = (E @ E.t()) - torch.eye(K, device=E.device)
    logging.info({"featdir_etf": K,
                  "etf_offdiag_cos_mean": round(off.sum().item() / (K * (K - 1)), 4),
                  "etf_offdiag_cos_max": round(off.max().item(), 4),
                  "rotation_cos_c_to_e_mean": round((C * E).sum(dim=1).mean().item(), 4)})
    _featdir_etf_frame._cache = (C, E)
    return _featdir_etf_frame._cache


def train_feat_direction(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """Feature-space DIRECTIONAL distillation (user, 2026-07-13) -- the loss-level version of the
    mechanism claim ("direction is the teacher's message, magnitude is the student's property"):

        L = || Phi_hat_s(x_adv) - Phi_hat_t(x) ||^2                       (backbone; = 2 - 2 cos)
          + beta * KL( head(scale * Phi_hat_s.DETACHED) || z_t / tau )    (head-only)

    The detach is the supervision split: the KL can only move the head (log_g / W_s, bias), the
    backbone sees nothing but the teacher's normalized (post-hoc) feature DIRECTION. The attack
    maximizes the same direction loss (inner_featdir_only_return) -- the teacher's classifier
    head w_t appears NOWHERE in the backbone path or the attack. Head soft target stays z_t/tau
    on purpose: a hard-CE head would flip to the HE regime and delete the per-class-confidence
    structure the gain channel is supposed to learn. With gain_head: True the student head is
    additionally w_s,c = exp(log_g_c) * w_t,c (the full decomposition: teacher head deleted from
    supervision, student head = 100 gains). tau = head-target sharpness, beta = head-loss weight
    (dir loss is bounded by 4, per-sample KL is O(1) at tau16 -> beta 1.0 is a sane start).
    Everything else (BN clean forward, lamda consistency, EMA) = train_temperature verbatim."""
    model.train()
    origin_model.eval()
    annealing = (epoch / config.epochs) ** 2
    decay = annealing * (1 - config.kappa) + config.kappa
    criterion_kl = nn.KLDivLoss(reduction='none')
    enc = model.encoder if hasattr(model, "encoder") else model
    beta = config.beta if config.beta is not None else 1.0
    scale = float(getattr(config, "feat_scale", 1.0) or 1.0)
    supp = float(getattr(config, "featdir_suppress", 0.0) or 0.0)      # moved up: needed by the AWP loss closure too
    alpha = float(getattr(config, "featdir_alpha", 0.0) or 0.0)
    npen = float(getattr(config, "featdir_norm_penalty", 0.0) or 0.0)  # T.4 mechanism cell: see _step_loss

    # freeze_head is resolved HERE, above the AWP block (moved 2026-09-01).  AdvWeightPerturb
    # deepcopies the model once and caches it on this function, and a deepcopy inherits
    # requires_grad, while `load_state_dict` in calc_awp copies values but not those flags.  If
    # the proxy is built before the head is frozen, its classifier stays trainable for the whole
    # run.  It was safe only because every config sets awp_warmup > 0, so earlier epochs froze
    # the head first; awp_warmup 0 would have broken it silently.
    # featdir_freeze_head (2026-08-02): keep the student head EXACTLY as loaded from the teacher --
    # no head loss, no head gradient. Direct test of Theorem 2: the student's feature distribution
    # differs from the teacher's (measured cos_adv 0.69-0.79, i.e. 38-46 degrees apart), so a head
    # calibrated on phi_t should be miscalibrated on phi_s(x_adv). A tie here kills Theorem 2 and
    # simplifies the method; a large loss confirms that re-solving the head is the necessary part.
    # NOTE: pair with feat_scale ~ mean||phi_t|| (11.23 on clean_200ep). The frozen head expects
    # inputs on the teacher's scale, and its bias is NOT rescaled, so feeding a unit vector would
    # let the bias dominate and confound "head is miscalibrated" with "input scale is wrong".
    freeze_head = bool(getattr(config, "featdir_freeze_head", False))
    if freeze_head:
        for _p in enc.linear.parameters():
            _p.requires_grad_(False)
        for _n in ("log_g", "log_s"):
            if hasattr(enc, _n):
                getattr(enc, _n).requires_grad_(False)
        if epoch == 0:
            logging.info({"featdir_freeze_head": True, "feat_scale": scale})

    # AWP (Wu et al. NeurIPS'20) -- optional, config.awp_gamma>0 and epoch>=awp_warmup. Off by
    # default (awp_gamma None/0): behavior identical to before this cell. config.awp_style
    # selects "proxy" (utils.AdvWeightPerturb, default) vs "sam" (utils.AdvWeightPerturbSAM,
    # the ADR-repo port) -- see train_temperature's comment for the mechanics difference.
    awp_style = str(getattr(config, "awp_style", "proxy") or "proxy")
    awp_gamma = float(getattr(config, "awp_gamma", 0.0) or 0.0)
    use_awp = awp_gamma > 0 and epoch >= int(getattr(config, "awp_warmup", 0) or 0)
    if use_awp:
        if awp_style == "sam":
            if not hasattr(train_feat_direction, "_awp_sam"):
                train_feat_direction._awp_sam = AdvWeightPerturbSAM(model, rho=awp_gamma)
            awp = train_feat_direction._awp_sam
            awp.rho = awp_gamma
        else:
            if not hasattr(train_feat_direction, "_awp"):
                train_feat_direction._awp = AdvWeightPerturb(model, gamma=awp_gamma)
            awp = train_feat_direction._awp
            awp.gamma = awp_gamma

    # SUBSPACE cells (exists-and-unique pair, 2026-07-13 evening): featdir_span 'teacher' projects
    # the direction loss (and attack) onto the k-dim orthonormalized span of the TEACHER head --
    # "defend only the classification subspace, still magnitude-free"; 'random' = same k, random
    # orthonormal subspace (RandomState-seeded) = the uniqueness placebo. None/absent = full 512-d.
    # train_eps (2026-07-14 night): optional TRAINING-attack radius override (numeric, e.g.
    # 0.0392157 = 10/255) -- config.eps stays 8/255 so evaluate() still tests at the standard
    # radius. Robust-leaning frontier points train harder, get evaluated identically.
    eps_train = float(getattr(config, "train_eps", 0.0) or 0.0) or config.eps

    # Direction-loss normalization switches (2026-08-01 grid). These scope to the DIRECTION LOSS and
    # its matched attack ONLY -- the head always reads the normalized student feature, so that head
    # logit scale stays identical across cells and the grid measures the backbone question rather
    # than a head-temperature artifact. `phi_t` raw vs `phi_t_hat` differ by a factor ||phi_t||~11.2
    # (std 2.1), so a mismatched pair (one side raw, one normalized) pins the student's feature norm.
    raw_s = bool(getattr(config, "featdir_rawstudent", False))
    raw_t = bool(getattr(config, "featdir_rawteacher", False))
    # The HEAD's input follows the architecture (student_norm), not the direction-loss switch.
    # ResNet18_z.forward normalizes before its linear layer while plain ResNet18 does not, so
    # feeding head_from_feat anything else creates a train/eval mismatch: the head would be trained
    # on phi_s_hat and then evaluated through forward() on the raw feature. Before 2026-08-02 the
    # head term was hard-wired to phi_s_hat, which made student_norm:False unusable with featdir.
    _snorm = getattr(config, "student_norm", None)
    if _snorm is None:
        _snorm = config.reformation
    head_hat = bool(_snorm)

    # ANGULAR TOLERANCE / deadband (featdir_angtol, 2026-08-19).  Sibling of angeps: angeps spends
    # the ATTACK budget where the loss's geometry says it matters, this spends the LOSS effort the
    # same way -- a sample already inside the tolerance is released instead of being dragged further
    # onto the natural teacher's direction.  Motivation is measured, not assumed: at convergence the
    # no-stack run is BETTER angularly aligned than the champion (mean cos_adv 0.842 vs 0.750, train,
    # matched attack, n=2048) and 5.8 AA worse, i.e. over-alignment to a non-robust anchor is a cost
    # and the stack is what currently prevents it.  This makes that brake intrinsic to the loss.
    # Implementation is deliberately ONE change: the 2-2cos value is kept exactly for active samples
    # and zeroed for released ones, so the loss shape is untouched (a hinge would change both).  The
    # ATTACK is left unmasked -- it still opens the angle for every sample; the mask only decides
    # what is learned from the resulting x_adv, and it is evaluated at x_adv, after the attack.
    angtol = float(getattr(config, "featdir_angtol", 0.0) or 0.0)
    _logged_angtol = []


    # Ported from the other server's champion recipe (2026-08-01), both OFF unless set so existing
    # configs are bit-identical to before:
    #   freeze_lr_epoch -- stop stepping OneCycleLR at this epoch, run the tail at constant LR.
    #   wa_start        -- don't touch the WA shadow before this epoch. main.py's
    #                      `exp_avg = model.state_dict()` (a live-tensor reference, NOT a deepcopy)
    #                      means the shadow tracks the model until the first update, so this
    #                      initializes the average at wa_start for free.
    # Both accept a fraction of the run (0.2) or an absolute epoch (20); see _resolve_epoch_arg.
    freeze_lr_epoch = _resolve_epoch_arg(getattr(config, "freeze_lr_epoch", None), config.epochs)
    wa_start = _resolve_epoch_arg(getattr(config, "wa_start", None), config.epochs) or 0

    span_mode = getattr(config, "featdir_span", None)
    Q = None
    if span_mode:
        if not hasattr(train_feat_direction, "_Q_cache"):
            train_feat_direction._Q_cache = {}
        t_enc = origin_model.encoder if hasattr(origin_model, "encoder") else origin_model
        Wt = t_enc.linear.weight.detach()
        # k dose ("how much of the teacher's direction to follow", 2026-07-14 k-curve): for
        # 'random' mode, --eta overrides the subspace dim (default = num_classes). Existing
        # curve points: k=100 pgd 30.12/30.50, k=512 (= plain featdir) 28.91.
        k = int(config.eta) if (span_mode != "teacher" and getattr(config, "eta", None)) else Wt.shape[0]
        ck = (span_mode, k)
        # featdir_span_resample (2026-07-19, user q): every prior "which k dims" experiment
        # (teacher-span, pca_natural, pca_robust, oracle) tied-or-lost to plain random -- content
        # doesn't matter, only the count (dimensionality bottleneck). This is a DIFFERENT axis: a
        # NEW random Q every epoch (instead of one Q fixed for the whole run via _Q_cache below)
        # so the student can't just dump its unconstrained freedom into one fixed static 162-dim
        # blind spot -- any given direction is graded in some epochs and free in others, closer to
        # dropout than to subspace *selection*.
        if span_mode == "random" and bool(getattr(config, "featdir_span_resample", False)):
            g = torch.Generator().manual_seed(int(getattr(config, "featdir_span_seed", 0) or 0) + epoch)
            base = torch.randn(Wt.shape[1], k, generator=g)
            Qm, _ = torch.linalg.qr(base.double().cpu())
            Q = Qm.float().cuda()
            logging.info({"featdir_span": span_mode, "span_k": k, "resample_epoch": epoch})
        else:
            if ck not in train_feat_direction._Q_cache and span_mode == "decorr":
                # decorr span (2026-07-19, user q): every content-based subspace pick (teacher/PCA/
                # oracle) tied-or-lost to random -- but a DIFFERENT (per-dim, not per-direction)
                # informed selection actually won once, in the old KL/carve pipeline
                # (train_carve_decorr_l1 above): fragility=|Phi_t(x)-Phi_t(x_adv)| correlates with
                # class_need=|W_t[pred]*Phi_t(x)| at only +0.49 -- decorr protects class-relevant
                # dims and only touches vulnerable-AND-class-irrelevant ones. Ported here as a HARD
                # channel selection (not a soft per-sample weight): estimate per-dim fragility/need
                # once over a handful of batches, put the k dims with the LOWEST
                # fragility*exp(-beta*need_rel) score (= safest to keep bound to the teacher) into
                # Q; the remaining (512-k) highest-score dims (vulnerable & class-irrelevant) are
                # left FREE, same slot the champion currently fills with a random pick.
                beta_decorr = float(getattr(config, "featdir_decorr_beta", 1.0) or 1.0)
                csteps = 2
                lin_w = Wt   # origin_model's own head weight, [num_classes, feat_dim]
                frag_sum = torch.zeros(Wt.shape[1], device=Wt.device)
                need_sum = torch.zeros(Wt.shape[1], device=Wt.device)
                n_seen = 0
                n_batches = 10
                probe_iter = iter(train_loader)
                for _ in range(n_batches):
                    try:
                        xb, yb = next(probe_iter)
                    except StopIteration:
                        break
                    xb, yb = xb.cuda(), yb.cuda()
                    with torch.no_grad():
                        feat_clean, logits_clean = origin_model(xb, feat=True)
                        pred = logits_clean.argmax(dim=1)
                    x_adv = xb.clone().detach()
                    cstep = config.eps / csteps
                    for _ in range(csteps):
                        x_adv.requires_grad_(True)
                        _, logits_adv = origin_model(x_adv, feat=True)
                        ce = F.cross_entropy(logits_adv, yb)
                        grad = torch.autograd.grad(ce, x_adv)[0]
                        x_adv = x_adv.detach() + cstep * grad.sign()
                        x_adv = torch.min(torch.max(x_adv, xb - config.eps), xb + config.eps).clamp(0.0, 1.0)
                    with torch.no_grad():
                        feat_adv, _ = origin_model(x_adv, feat=True)
                        fragility = (feat_clean - feat_adv).abs()                       # [N, 512]
                        class_need = (lin_w[pred] * feat_clean).abs()                   # [N, 512]
                        need_rel = class_need / (class_need.mean(dim=1, keepdim=True) + 1e-8)
                        frag_sum += fragility.sum(dim=0)
                        need_sum += need_rel.sum(dim=0)
                        n_seen += xb.shape[0]
                frag_mean = frag_sum / n_seen
                need_mean = need_sum / n_seen
                score = frag_mean * torch.exp(-beta_decorr * need_mean)   # high = safe to free
                keep_idx = torch.argsort(score)[:k]                       # k LOWEST score = kept/graded
                Qm = torch.zeros(Wt.shape[1], k, dtype=torch.float64)
                Qm[keep_idx, torch.arange(k)] = 1.0
                train_feat_direction._Q_cache[ck] = Qm.float().cuda()
                logging.info({"featdir_span": "decorr", "span_k": k, "decorr_beta": beta_decorr,
                              "frag_mean": round(frag_mean.mean().item(), 4),
                              "need_mean": round(need_mean.mean().item(), 4)})
            if ck not in train_feat_direction._Q_cache:
                if span_mode == "teacher":
                    base = Wt.t()
                elif span_mode.startswith("pca_"):    # pca_natural / pca_robust / pca_kdstudent(oracle)
                    # top-k feature-PCA directions of the natural / ROBUST teacher (user's
                    # "robust-important subspace" hypothesis; scripts/diag_feature_pca.py --
                    # eff. rank: natural 57.2, robust 8.7(!)). Target stays the NATURAL teacher's
                    # direction; only the grading metric is informed.
                    V = np.load("results/CIFAR100/feature_pca_bases.npz")[span_mode.split("_")[1]]
                    base = torch.tensor(V[:, :k])
                else:   # 'random' placebo / k-dose
                    g = torch.Generator().manual_seed(int(getattr(config, "featdir_span_seed", 0) or 0))
                    base = torch.randn(Wt.shape[1], k, generator=g)
                Qm, _ = torch.linalg.qr(base.double().cpu())
                train_feat_direction._Q_cache[ck] = Qm.float().cuda()
                logging.info({"featdir_span": span_mode, "span_k": k})
            Q = train_feat_direction._Q_cache[ck]

    dir_sum, cos_sum, n_sum = 0.0, 0.0, 0
    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        N, C, H, W = x.shape
        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()

        with torch.no_grad():
            phi_t, teacher_logits = origin_model(x, feat=True)          # raw teacher feat & logits
            phi_t_hat = F.normalize(phi_t, dim=1).detach()              # post-hoc directionalized teacher
            # featdir_proto_gamma (2026-08-03, trick A): shrink the DIRECTION target toward its class
            # prototype.  target <- normalize( (1-g) * phi_t_hat(x) + g * c_y ),  c_y = unit-norm mean
            # of phi_t_hat over class y.  Motivation (Ilyas frame): the natural teacher's direction
            # carries non-robust features, and those live in the INSTANCE-specific residual, not in
            # the class-level direction.  g=0 is the champion; g=1 is a pure class prototype (all
            # dark knowledge deleted).  Head target z_t/tau is deliberately left untouched -- this
            # dial is about what the BACKBONE is anchored to, not about the head's soft labels.
            _pg = float(getattr(config, "featdir_proto_gamma", 0.0) or 0.0)
            if _pg > 0:
                _C = _featdir_class_proto(origin_model, train_loader, config)
                phi_t_hat = F.normalize((1.0 - _pg) * phi_t_hat + _pg * _C[y], dim=1).detach()
            # featdir_etf_rotate (2026-08-03, trick B): per-class rotation c_y -> e_y applied to the
            # whole direction, so class prototypes become equiangular while the instance residual
            # rides along. R = I - ss^T/(1+u.v) + 2 v u^T  with u=c_y, v=e_y, s=u+v  (two reflections,
            # orthogonal, R u = v). Composes with proto_gamma: shrink first, then rotate.
            if bool(getattr(config, "featdir_etf_rotate", False)):
                _C, _E = _featdir_etf_frame(origin_model, train_loader, config)
                u, v = _C[y], _E[y]                                       # [N, D]
                s = u + v
                denom = (1.0 + (u * v).sum(dim=1, keepdim=True)).clamp(min=1e-6)
                phi_t_hat = (phi_t_hat
                             - s * ((s * phi_t_hat).sum(dim=1, keepdim=True) / denom)
                             + 2.0 * v * (u * phi_t_hat).sum(dim=1, keepdim=True))
                phi_t_hat = F.normalize(phi_t_hat, dim=1).detach()
            if bool(getattr(config, "featdir_normfeat_target", False)):
                # user q (2026-07-17): head target from the NORMALIZED teacher feature through
                # the teacher's own (raw, unnormalized) head, instead of z_t/tau. z_t/tau already
                # approximates this (z_t/tau = (||Phi_t||/tau)*(W_t.Phi_hat_t), ||Phi_t||~13≈tau);
                # this cell makes it exact (drops the tau-vs-||Phi_t|| approximation gap) while
                # keeping the teacher head's per-class weight-norm structure (unlike costeacher's
                # full W_t normalization, which already TIED at 41.61 vs 41.77 on 2026-07-08).
                t_enc = origin_model.encoder if hasattr(origin_model, "encoder") else origin_model
                Wt_tg = _featdir_target_head(t_enc.linear.weight.detach(), config)
                target = F.linear(phi_t_hat, Wt_tg, t_enc.linear.bias.detach()).detach()
            elif bool(getattr(config, "featdir_normhead_target", False)):
                # teacher-HEAD normalization for the featdir target (2026-08-01, normalization grid):
                # rebuild z_t with row-normalized W_t (per-class ||w_c|| structure deleted) while the
                # feature stays raw. Rows are rescaled to the original mean ||w_c|| so target
                # sharpness is unchanged -- the same fair-start convention resnet_zcos uses. This is
                # the featdir analogue of costeacher, which only ever ran on the baseline-KL method.
                t_enc = origin_model.encoder if hasattr(origin_model, "encoder") else origin_model
                Wt_tg = _featdir_target_head(t_enc.linear.weight.detach(), config)
                target = (F.linear(phi_t, Wt_tg, t_enc.linear.bias.detach()) / config.tau).detach()
            else:
                target = (teacher_logits / config.tau).detach()             # head-only soft target

        # SELF-METRIC cell (2026-07-13 night): reconstruct the baseline gradient geometry
        # W_s^T(p - p_t) WITHOUT teacher logits -- route the direction error through the student's
        # OWN head (detached snapshot = metric only, not trained by this term):
        #   L_backbone = KL( softmax(W_s.sg * Phi_hat_s(x_adv)) || softmax(W_s.sg * Phi_hat_t) )
        # Teacher still sends ONLY Phi_hat_t; span(W_s) + softmax saliency weighting come from the
        # student. ||Phi_t||~13 makes normalization auto-set ~tau-13 sharpness (the flat optimum).
        # Attack = matched KL to the same z_tdir target (dirattack crash showed KL-shaped losses
        # need their matched adversary). Free head only (gain_head unsupported here).
        # 'teachkl' variant (uniqueness partner, 2026-07-14): same construction but the metric
        # head is the TEACHER's (frozen). Both recover -> any decision-shaped metric suffices
        # (parallels random=teacher in the span pair); only selfkl recovers -> the metric must
        # TRACK the student. Either way the exists-and-unique structure closes.
        metric_mode = str(getattr(config, "featdir_metric", "l2"))
        selfkl = metric_mode in ("selfkl", "teachkl")
        if selfkl:
            m_enc = (origin_model.encoder if hasattr(origin_model, "encoder") else origin_model) \
                    if metric_mode == "teachkl" else enc
            W_det = m_enc.linear.weight.detach()
            b_det = m_enc.linear.bias.detach()
            z_tdir = F.linear(scale * phi_t_hat, W_det, b_det).detach()
            x_pgd = inner_loss_only_return(model, z_tdir, x, y, optimizer, config.step_size, eps_train, perturb_steps=config.steps)
        elif str(getattr(config, "featdir_attack", "dir")) == "dircons":
            # matched-adversary cell (user, 2026-07-15): x_adv must stress the consistency term
            # too, else lamda is inert on featdir (measured backbone g_cons ~4e-6 under the
            # dir-only adversary). w = featdir_atk_cons = relative weight inside the attack.
            with torch.no_grad():
                model.eval()
                clean_logits_atk = model(x).detach()
                model.train()
            x_pgd = inner_featdir_cons_return(model, phi_t_hat, x, optimizer, config.step_size, eps_train, perturb_steps=config.steps,
                                              Q=Q, w_cons=float(getattr(config, "featdir_atk_cons", 1.0) or 1.0), clean_logits=clean_logits_atk)
        elif str(getattr(config, "featdir_attack", "dir")) == "kl":
            # attack-isolation cell (2026-07-13): training LOSS unchanged, inner maximization
            # swapped back to the original KL-to-target attack (through the student head).
            # Discriminates "featdir's -1.8(pgd)+cw-tie = weaker direction-max adversary" vs
            # "= removing logit supervision from the backbone". Recovery -> story-faithful
            # method at baseline performance (the user's stated goal).
            x_pgd = inner_loss_only_return(model, target, x, y, optimizer, config.step_size, eps_train, perturb_steps=config.steps)
        else:
            phi_t_dir = phi_t.detach() if raw_t else phi_t_hat            # direction-loss target

            # featdir_teacher_at_adv (2026-09-01): READ THE TEACHER AT x_adv INSTEAD OF x.  This is the
            # direct test of the paper's central structural claim -- that the teacher's own instability
            # cannot be inherited because Phi_t is evaluated at the clean point and nowhere else.  With
            # this flag the objective becomes ||Phi_s(x') - Phi_t(x')||, i.e. our method's version of
            # what AdaAD does in logit space, and the teacher's 63.8-degree rotation under attack enters
            # the target.  Everything else -- the attack, the schedule, the head -- is unchanged, so the
            # comparison isolates the read point and nothing else.  Off by default; every existing run
            # is bit-identical.  Implemented inside the attack loop below via `_t_at`, because the
            # target has to move with x' rather than being fixed once.
            # ANGULAR-BUDGET EPS (featdir_angeps_p, 2026-08-04). Per-sample eps is an existing family
            # -- IAAT (1910.08051), MMA (1812.02637), CAT (2002.06789) all assign a per-sample radius
            # -- but all three set it from DIFFICULTY / input-space margin. Ours is set from the
            # geometry the loss actually lives in: this method attacks a FEATURE ANGLE, and the same
            # pixel radius rotates different samples by wildly different amounts. So equalize the
            # angular budget instead of the pixel budget.
            # First order, the angle moved under an L-inf ball of radius e is ~ e * ||grad_x L_dir||,
            # so eps_i ~ (gbar / g_i)^p equalizes it (p=0 uniform = champion, p=1 full equalization).
            # MEAN-PRESERVED after clipping: sum(eps_i) == N * eps_train, so this run and the champion
            # spend the exact same total budget and any difference is ALLOCATION, not more attack.
            eps_use, step_use = eps_train, config.step_size
            _ap = float(getattr(config, "featdir_angeps_p", 0.0) or 0.0)
            if _ap > 0:
                _lo = float(getattr(config, "featdir_angeps_lo", 0.5) or 0.5)
                _hi = float(getattr(config, "featdir_angeps_hi", 1.5) or 1.5)
                model.eval()
                _xg = x.clone().detach().requires_grad_(True)
                with torch.enable_grad():
                    _fs, _ = model(_xg, feat=True)
                    _fsh = _fs if raw_s else F.normalize(_fs, dim=1)
                    _d0 = _fsh - phi_t_dir
                    _l0 = (_d0 @ Q).pow(2).sum() if Q is not None else _d0.pow(2).sum()
                _g = torch.autograd.grad(_l0, [_xg])[0].detach()
                model.train()
                optimizer.zero_grad()
                # BASELINE SIGNAL (featdir_eps_signal, 2026-08-22).  T.5 differentiates this rule
                # from the existing per-sample-eps family (IAAT 1910.08051, MMA 1812.02637,
                # CAT 2002.06789) by WHAT the radius is allocated from: the input-sensitivity of the
                # training loss, rather than the sample's difficulty or input-space margin.  That
                # distinction had never been measured -- reimplementing those methods end-to-end
                # would compare whole recipes, not allocation rules, so instead this swaps the
                # per-sample signal in place and changes nothing else: same exponent p, same
                # [lo, hi] clamp, same mean restoration, same everything downstream.
                #   "grad"       (default) g_i = ||grad_x L_dir||_p    -- ours
                #   "difficulty"           g_i = per-sample CE on clean x, so easy samples (low
                #                          loss) receive the larger radius, which is the direction
                #                          IAAT/CAT assign it.
                #   "margin"               g_i = logit margin z_y - max_{c != y} z_c, MMA's actual
                #                          criterion.  MMA grows the radius until the sample is
                #                          misclassified, so a large margin earns a large radius;
                #                          under the shared `mean/g` form that means feeding the
                #                          RECIPROCAL, hence the negation below.  Clamped positive
                #                          because a misclassified sample has a negative margin and
                #                          the shared machinery needs a positive scale.
                _sig = str(getattr(config, "featdir_eps_signal", "grad") or "grad")
                if _sig in ("difficulty", "difficulty_rank", "margin"):
                    with torch.no_grad():
                        _z = model(x)
                        if _sig == "margin":
                            _zy = _z.gather(1, y.unsqueeze(1)).squeeze(1)
                            _zo = _z.scatter(1, y.unsqueeze(1), float("-inf")).max(dim=1).values
                            # mean/g is the shared form, so invert here: large margin -> large radius
                            _ce = 1.0 / (_zy - _zo).clamp(min=1e-3)
                        else:
                            _ce = F.cross_entropy(_z, y, reduction="none")
                    if _sig in ("difficulty", "margin"):
                        _g = None        # marks the grad path as unused below
                # SENSITIVITY NORM.  The first-order angle moved inside an L-INF ball of radius e is
                #   max_{||d||_inf <= e} <grad, d> = e * ||grad||_1   (the dual norm of L-inf is L1),
                # so equalizing the ANGULAR budget needs the L1 gradient norm.  featdir_angeps_gnorm
                # defaults to 2 (the 2026-08-04 champion used the L2 norm, and every logged angeps
                # run must stay reproducible); set it to 1 for the allocation the derivation
                # actually describes.
                _gp = int(getattr(config, "featdir_angeps_gnorm", 2) or 2)
                _gn = (_ce.clamp(min=1e-12) if _sig in ("difficulty", "margin")
                       else _g.flatten(1).norm(dim=1, p=_gp).clamp(min=1e-12))
                _r = (_gn.mean() / _gn).pow(_ap)
                if bool(getattr(config, "featdir_angeps_exact_budget", False)):
                    # EXACT box-constrained budget.  The plain "clip then rescale to mean 1" below
                    # rescales the clipped entries too, so they leave [lo,hi] again -- the champion
                    # log shows w_min 0.40 against a configured lo of 0.5.  Solving instead for the
                    # scale t with  sum_i clip(t*r_i, lo, hi) == N  keeps BOTH constraints exactly:
                    # f(t) is continuous and nondecreasing from N*lo to N*hi, so bisect.  With no
                    # clip active the solution is t = N/sum(r), i.e. this strictly generalizes the
                    # mean-restore.  Requires lo <= 1 <= hi for feasibility.
                    # bisect on log t: t=lo/max(r) saturates everything low (f=N*lo<=N) and
                    # t=hi/min(r) saturates everything high (f=N*hi>=N), so the root is bracketed;
                    # r spans orders of magnitude when CV(g) is large, hence log space.
                    _N = _r.numel()
                    _a = torch.log(_lo / _r.max())
                    _b = torch.log(_hi / _r.min())
                    _logr = torch.log(_r)
                    for _ in range(60):
                        _t = 0.5 * (_a + _b)
                        _s = torch.exp(_t + _logr).clamp(_lo, _hi).sum()
                        _a, _b = torch.where(_s < _N, _t, _a), torch.where(_s < _N, _b, _t)
                    _w = torch.exp(0.5 * (_a + _b) + _logr).clamp(_lo, _hi)
                else:
                    _w = _r.clamp(_lo, _hi)
                    _w = _w * (_w.numel() / _w.sum())                      # restore mean 1 AFTER clip
                if _sig == "difficulty_rank":
                    # RANK-MATCHED DIFFICULTY BASELINE.  Feeding a difficulty score through the same
                    # (gbar/g)^p formula does not work here: the student reads a unit-norm feature,
                    # so its logits are small, its softmax is near uniform, and per-sample CE sits at
                    # ln(100) for everyone -- measured CV 0.027 (epoch 0) / 0.041 (epoch 1) against
                    # the gradient signal's 0.68, i.e. an essentially uniform allocation that would
                    # reproduce p=0 for a reason having nothing to do with difficulty.
                    # Instead keep the gradient rule's weight MULTISET exactly -- same values, same
                    # clamp, same mean, so the same total budget -- and only re-assign which sample
                    # gets which, ordering by difficulty so the easiest sample receives the largest
                    # radius (the direction IAAT/CAT assign it).  The two runs then differ in nothing
                    # but the ordering, which is exactly the claim under test: that allocating by the
                    # loss's input-sensitivity is not the same as allocating by difficulty.
                    _hard = torch.argsort(_ce, descending=True)            # hardest first
                    _wsorted, _ = torch.sort(_w)                           # smallest first
                    _wperm = torch.empty_like(_w)
                    _wperm[_hard] = _wsorted                               # hardest <- smallest
                    _w = _wperm
                eps_use = eps_train * _w.view(-1, 1, 1, 1)
                step_use = config.step_size * _w.view(-1, 1, 1, 1)         # keep steps-per-radius fixed
                if batch_idx == 0:
                    logging.info({"angeps_p": _ap, "gnorm": _gp, "signal": _sig,
                                  "exact_budget": bool(getattr(config, "featdir_angeps_exact_budget", False)),
                                  "w_min": round(_w.min().item(), 3),
                                  "w_max": round(_w.max().item(), 3),
                                  "w_mean": round(_w.mean().item(), 4),
                                  "w_std": round(_w.std().item(), 3),
                                  "gradnorm_cv": round((_gn.std() / _gn.mean()).item(), 3),
                                  "epoch": epoch})
            # featdir_attack_full_rank (2026-08-08): Q normally projects BOTH the outer loss and the
            # inner attack, so sweeping k confounds "how many directions are SUPERVISED" with "how
            # many directions the adversary may rotate in" -- and the two push clean accuracy in
            # opposite directions, which makes the k sweep uninformative unless it comes out one
            # particular way.  Setting this True leaves the attack at full rank so that k varies the
            # supervision rank alone.  Default False = unchanged behaviour.
            _Q_atk = None if bool(getattr(config, "featdir_attack_full_rank", False)) else Q
            # featdir_ce_attack (2026-09-02): keep the ANCHOR as the training loss but generate x_adv
            # with a plain true-label CE-PGD instead.  This is the missing isolation between the two
            # things that differ between our method and PGD-AT at matched stack and matched
            # initialization: the training objective AND the attack objective both change, and the
            # +2.31 AA the anchor gains (62.65 / 28.77 against 57.73 / 26.46) has never been attributed
            # to either.  With this flag only the loss is ours; if the gain survives, the anchor
            # carries it, and if it disappears, the credit belongs to attacking in feature space.
            if bool(getattr(config, "featdir_ce_attack", False)):
                x_pgd = _pgd_attack_true_label(model, x, y, config.step_size, eps_use if not torch.is_tensor(eps_use) else eps_train, config.steps)
            elif bool(getattr(config, "featdir_teacher_at_adv", False)):
                x_pgd = inner_featdir_teacher_at_adv(model, origin_model, x, optimizer, step_use,
                                                     eps_use, perturb_steps=config.steps, Q=_Q_atk,
                                                     raw_student=raw_s, raw_teacher=raw_t)
                # the OUTER target has to move with x' as well, or the ablation would only change the
                # attack and leave the loss anchored at x -- which is a third setting, not the one
                # under test.  Rebinding phi_t_dir here is enough: _step_loss and _awp_loss_fn both
                # read it, and the AWP closure captures it as a default argument defined after this.
                with torch.no_grad():
                    _pt_a, _ = origin_model(x_pgd, feat=True)
                phi_t_dir = _pt_a.detach() if raw_t else F.normalize(_pt_a, dim=1).detach()
            else:
                x_pgd = inner_featdir_only_return(model, phi_t_dir, x, optimizer, step_use, eps_use, perturb_steps=config.steps, Q=_Q_atk, raw_student=raw_s)

        def _step_loss():
            feat_s, plus_logits = model(x_pgd, feat=True)                   # raw student feat (pre-norm)
            fs_hat = F.normalize(feat_s, dim=1)                             # head input: ALWAYS normalized
            fs_dir = feat_s if raw_s else fs_hat                            # direction-loss student side
            d_feat = fs_dir - phi_t_dir
            if selfkl:
                z_adv_m = F.linear(scale * fs_hat, W_det, b_det)            # metric head: params detached
                dir_loss = criterion_kl(F.log_softmax(z_adv_m, dim=1), F.softmax(z_tdir, dim=1)).sum(dim=1)
            elif Q is not None:
                dir_loss = (d_feat @ Q).pow(2).sum(dim=1)                   # subspace-projected direction loss
            else:
                dir_loss = d_feat.pow(2).sum(dim=1)                        # 2 - 2cos per sample
            if angtol > 0:
                with torch.no_grad():
                    _cos_i = F.cosine_similarity(fs_dir, phi_t_dir, dim=1)
                    _act = (_cos_i < angtol).float()
                dir_loss = dir_loss * _act
                if batch_idx == 0 and not _logged_angtol:
                    logging.info({"angtol": angtol, "active_frac": round(_act.mean().item(), 3),
                                  "cos_mean": round(_cos_i.mean().item(), 4),
                                  "epoch": epoch})
                    _logged_angtol.append(1)
            loss = dir_loss.mean()

            # featdir_norm_penalty (2026-08-09, theory_v1 T.4 mechanism cell 1): direction loss
            # UNCHANGED (pure cos, attack unchanged), plus an explicit penalty binding the raw
            # student feature norm to the teacher's: npen * (||phi_s|| - ||phi_t||)^2 per sample.
            # Separates the two open accounts of rawfeat's clean -0.95: if binding the norm PER SE
            # is the cost (capacity account), this cell drops clean too; if the cost was the raw
            # L2 gradient's geometry/scale (dynamics account), this cell holds clean -- the
            # direction gradient here is still the pure cos one.
            if npen > 0:
                loss = loss + npen * (feat_s.norm(dim=1) - phi_t.detach().norm(dim=1)).pow(2).mean()

            # Mechanism cell (5): FORBID USE of the free subspace (2026-07-14 evening) — energy
            # penalty on the complement of span(Q): the student is still ungraded there wrt the
            # teacher, but may not PUT anything there. Prediction: robustness falls toward k512
            # (dissociates "freedom to build" from "merely not being punished"). Weight via
            # featdir_suppress (0/absent = off, defined above with beta/scale). Applies at train;
            # at convergence complement energy ~0 so eval sees no train/test mismatch.
            if supp > 0 and Q is not None:
                comp = fs_hat - (fs_hat @ Q) @ Q.T                          # (I - QQ^T) Phi_hat_s
                loss = loss + supp * comp.pow(2).sum(dim=1).mean()

            # featdir_alpha (2026-07-13 night): partial-undetach dial -- fraction alpha of the head-KL
            # gradient reaches the backbone (alpha 0 = pure firewall/current, 1 = full flow). Tests
            # whether TEACHER-confidence routing adds anything beyond the self-metric. (defined above)
            # 2026-08-03: pick the head's input FIRST, then apply alpha. The old order applied the
            # alpha mix to fs_hat and then discarded it whenever head_hat was False, so student_norm
            # False silently bypassed the detach and the head KD kept pushing the backbone. alpha is
            # 1.0 in every config run before this date, so no existing result changes.
            _feat_head_in = fs_hat if head_hat else feat_s
            feat_for_head = _feat_head_in if alpha >= 1.0 else (
                alpha * _feat_head_in + (1.0 - alpha) * _feat_head_in.detach())
            head_logits = enc.head_from_feat(scale * feat_for_head)
            # featdir_head_ce (2026-07-17, user q): the backbone is already teacher-anchored (L_dir),
            # so does the head even need soft-target KD, or can it just fit the TRUE label directly
            # (hard CE on x_adv)? Loses teacher dark-knowledge, but removes the head's objective-eval
            # mismatch (KD divergence vs the CE/margin metrics AA/CW actually score).
            if freeze_head:
                pass                                     # head stays at the teacher's solution
            elif bool(getattr(config, "featdir_head_ce", False)):
                loss = loss + beta * F.cross_entropy(head_logits, y)
            else:
                kl_loss = criterion_kl(F.log_softmax(head_logits, dim=1), F.softmax(target, dim=1))
                loss = loss + beta * (1.0 / N) * (kl_loss.sum(dim=1)).sum()

            student_logits = model(x)   # clean forward: updates student BN running stats
            if config.lamda is not None and config.lamda > 0:
                # cons_detach (2026-07-16, user q "왜 detach 없나"): default here is BIDIRECTIONAL
                # (neither side detached -> adv pulled to clean AND clean pulled to adv). TRADES-style
                # is one-directional (clean side fixed as target). featdir_cons_detach:True switches to
                # the TRADES form for direct comparison against the champion (k350+WA+lamda4).
                target_logits = student_logits.detach() if bool(getattr(config, "featdir_cons_detach", False)) else student_logits
                consistency_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(target_logits, dim=1))
                # featdir_lamda_noanneal (2026-07-19, user q): WA's decay keeps its own epoch
                # ramp (annealing*(1-kappa)+kappa above), but lamda is doubly-damped by this SAME
                # annealing=(ep/epochs)^2 factor -- full dose only at the very end (lamda4 with
                # annealing =~ lamda0.03 for most of training, see lamda-scale note in memory).
                # This flag applies config.lamda at FULL strength from epoch 0 instead -- sweep
                # SMALL values here (0.1/0.3), not lamda4-scale, since the effective early-training
                # dose is now ~30x higher than the annealed champion ever saw at the same lamda.
                lamda_dose = config.lamda if bool(getattr(config, "featdir_lamda_noanneal", False)) else annealing * config.lamda
                loss = loss + lamda_dose * (consistency_loss).mean()

            return loss, dir_loss, fs_hat, plus_logits

        if use_awp and awp_style == "sam":
            # ADR/SAM-style: ascent direction from the gradient at CLEAN weights; the gradient
            # the optimizer actually applies is measured at the perturbed weights, then weights
            # are restored to w before optimizer.step() (see AdvWeightPerturbSAM docstring).
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
            def _awp_loss_fn(pm, _x_pgd=x_pgd, _phi_t_hat=phi_t_dir, _target=target):
                p_enc = pm.encoder if hasattr(pm, "encoder") else pm
                fs_, _ = pm(_x_pgd, feat=True)
                fh_ = F.normalize(fs_, dim=1)
                fd_ = fs_ if raw_s else fh_
                if selfkl:
                    za_ = F.linear(scale * fh_, W_det, b_det)
                    dl_ = criterion_kl(F.log_softmax(za_, dim=1), F.softmax(z_tdir, dim=1)).sum(dim=1)
                elif Q is not None:
                    dl_ = ((fh_ - _phi_t_hat) @ Q).pow(2).sum(dim=1)
                else:
                    dl_ = (fh_ - _phi_t_hat).pow(2).sum(dim=1)
                l_ = dl_.mean()
                if supp > 0 and Q is not None:
                    comp_ = fh_ - (fh_ @ Q) @ Q.T
                    l_ = l_ + supp * comp_.pow(2).sum(dim=1).mean()
                # 2026-09-01: this closure diverged from _step_loss in two ways, so AWP ascended on
                # an objective the model does not train.  (a) It had no `freeze_head` branch, while
                # _step_loss skips the head term entirely.  (b) The alpha detach was applied to `ff_`
                # while the head_hat=False path feeds `fs_`, so on the RAW design featdir_alpha=0
                # detached nothing -- it did detach on the DIRECTIONAL design, which is what the
                # 2026-08-26 "impact 0" note measured, before the shipped design changed underneath
                # it.  Measured on the shipped raw champion before this fix: head term 7.15 against
                # the anchor's 4.15, supplying 40.8% of the AWP backbone gradient norm, cosine 0.918
                # between the two directions.  Both branches now mirror _step_loss exactly.
                if freeze_head:
                    return l_
                _fhi_ = fh_ if head_hat else fs_
                ff_ = _fhi_ if alpha >= 1.0 else (alpha * _fhi_ + (1.0 - alpha) * _fhi_.detach())
                hl_ = p_enc.head_from_feat(scale * ff_)
                if bool(getattr(config, "featdir_head_ce", False)):
                    return l_ + beta * F.cross_entropy(hl_, y)
                kl_ = criterion_kl(F.log_softmax(hl_, dim=1), F.softmax(_target, dim=1))
                return l_ + beta * (1.0 / N) * kl_.sum(dim=1).sum()
            awp_diff = awp.calc_awp(_awp_loss_fn)
            awp.perturb(awp_diff)
            loss, dir_loss, fs_hat, plus_logits = _step_loss()
            loss.backward()
            optimizer.step()
            awp.restore(awp_diff)     # net effect: grad direction from a locally-worst nearby
                                       # weight, applied from the actual (unperturbed) position
        else:
            loss, dir_loss, fs_hat, plus_logits = _step_loss()
            loss.backward()
            optimizer.step()

        if freeze_lr_epoch is None or epoch < freeze_lr_epoch:
            scheduler.step()
        elif epoch == freeze_lr_epoch and batch_idx == 0:
            logging.info({"freeze_lr_epoch": freeze_lr_epoch,
                          "frozen_lr": optimizer.param_groups[0]["lr"]})

        if config.weight_avg == True and epoch >= wa_start:
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 - decay) * value + decay * exp_avg[key]

        with torch.no_grad():
            dir_sum += dir_loss.sum().item()
            cos_sum += (fs_hat * phi_t_hat).sum(dim=1).sum().item()
            n_sum += N

    logging.info({"dir_loss_adv": round(dir_sum / max(n_sum, 1), 4),
                  "cos_adv": round(cos_sum / max(n_sum, 1), 4), "epoch": epoch})
    _log_gain_stats(model, epoch)


def train_temperature_cemix(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """train_temperature + a true-label CE term on the ADVERSARIAL example (user's idea, 2026-07-07):

        loss = KL(student(x_adv) || teacher(x)/tau) + beta * CE(student(x_adv), y)

    Rationale: the pipeline distills from a NATURAL teacher and currently sees no label at all --
    RSLAD's "pure KD beats CE-mix" verdict assumed a ROBUST teacher, so here the label may supply
    exactly the robust signal the natural teacher lacks. Distinct from dualkd (which added a CLEAN
    KD term and lost monotonically): this term is the classic AT objective on x_adv. beta=0 ==
    train_temperature exactly (50k H 41.77 seed0). Sweep beta via --beta.
    """
    model.train()
    origin_model.eval()
    annealing = (epoch / config.epochs) ** 2
    decay = annealing * (1 - config.kappa) + config.kappa
    criterion_kl = nn.KLDivLoss(reduction='none')

    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        N, C, H, W = x.shape
        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()

        with torch.no_grad():
            _, teacher_logits = origin_model(x, feat=True)
            target = (teacher_logits / config.tau).detach()

        x_pgd = inner_loss_only_return(model, target, x, y, optimizer, config.step_size, config.eps, perturb_steps=config.steps)
        plus_logits = model(x_pgd)
        kl_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(target, dim=1))
        loss = (1.0 / N) * (kl_loss.sum(dim=1)).sum()

        if config.beta is not None and config.beta > 0:
            loss += config.beta * F.cross_entropy(plus_logits, y)

        student_logits = model(x)   # clean forward: updates student BN running stats
        if config.lamda is not None and config.lamda > 0:
            consistency_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(student_logits, dim=1))
            loss += annealing * config.lamda * (consistency_loss).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()

        if config.weight_avg == True:
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 - decay) * value + decay * exp_avg[key]


def train_madry_at(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """Plain Madry PGD-AT, no distillation. CE(model(x_adv), y) with a true-label PGD attack.
    origin_model is unused -- but the checkpoint it was built from is still what the student is
    initialized to when `finetune: True`, which is the point of the teacher-init control below.

    Stack knobs added 2026-08-30 so this can be run at the SAME regime as the anchored cells:
    `train_eps`, `wa_start`, `freeze_lr_epoch` and AWP, all inert unless set, so every earlier
    madry_at run is bit-identical to before.  The control this exists for -- initialize at the clean
    teacher, then train with label CE instead of the feature anchor -- is otherwise unanswerable: the
    existing `at_ce_*` cells are 3-step / 50-epoch / no-AWP and land at PGD 28.28 against ~35 for
    every anchored cell, so they cannot be compared with them."""
    model.train()
    eps_train = float(getattr(config, "train_eps", 0.0) or 0.0) or config.eps
    freeze_lr_epoch = _resolve_epoch_arg(getattr(config, "freeze_lr_epoch", None), config.epochs)
    wa_start = _resolve_epoch_arg(getattr(config, "wa_start", None), config.epochs) or 0
    awp_gamma = float(getattr(config, "awp_gamma", 0.0) or 0.0)
    # featdir_freeze_head, honoured here too (2026-09-06).  The anchored cells inherit the teacher's
    # classifier and never train it, so "the gain is the free, well-calibrated head" is a live
    # objection and the CE control has to be runnable with the same head frozen.  Inert unless the
    # flag is set, so every earlier madry_at run is unchanged.
    if bool(getattr(config, "featdir_freeze_head", False)):
        _enc = model.encoder if hasattr(model, "encoder") else model
        for _p in _enc.linear.parameters():
            _p.requires_grad_(False)
        if epoch == 0:
            logging.info({"madry_at_freeze_head": True})
    use_awp = awp_gamma > 0 and epoch >= int(getattr(config, "awp_warmup", 0) or 0)
    if use_awp:
        if not hasattr(train_madry_at, "_awp"):
            train_madry_at._awp = AdvWeightPerturb(model, gamma=awp_gamma)
        awp = train_madry_at._awp
        awp.gamma = awp_gamma
    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        x, y = x.cuda(), y.cuda()
        x_adv = _pgd_attack_true_label(model, x, y, config.step_size, eps_train, perturb_steps=config.steps)
        if use_awp:
            def _awp_loss_fn(pm, _xa=x_adv, _y=y):
                return F.cross_entropy(pm(_xa), _y)
            awp_diff = awp.calc_awp(_awp_loss_fn)
            awp.perturb(awp_diff)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x_adv), y)
        loss.backward()
        optimizer.step()
        if use_awp:
            awp.restore(awp_diff)
        if freeze_lr_epoch is None or epoch < freeze_lr_epoch:
            scheduler.step()
        if config.weight_avg == True and epoch >= wa_start:
            annealing = (epoch / config.epochs) ** 2
            decay = annealing * (1 - config.kappa) + config.kappa
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 - decay) * value + decay * exp_avg[key]


def _soft_ce(logits, target_prob):
    """Cross-entropy against a soft target distribution."""
    return -(target_prob * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def _pgd_attack_soft_label(model, x_natural, target_prob, step_size, epsilon, perturb_steps):
    """PGD maximizing soft-target cross-entropy. ADR attacks the rectified label, not the hard one."""
    was_training = model.training
    model.eval()
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss = _soft_ce(model(x_adv), target_prob)
        grad = torch.autograd.grad(loss, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    if was_training:
        model.train()
    return x_adv.detach()


def _cosine_sched(base, final, total_steps):
    """ADR's cosine_scheduler, as a closure over the step index rather than a materialized array."""
    def at(step):
        t = min(max(step, 0), max(total_steps - 1, 1)) / max(total_steps - 1, 1)
        return final + 0.5 * (base - final) * (1.0 + math.cos(math.pi * t))
    return at


def train_adr(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """ADR (Annealing Self-Distillation Rectification, ICLR'24), ported from the official repo.

    Why this is ported rather than quoted.  ADR at AT+WA+AWP is the row our headline comparison is
    read against -- 57.36 clean / 28.50 AA on CIFAR-100 against our 62.17 / 28.86, the pair the paper
    describes as "the same robustness, five points of accuracy kept."  Quoting the number from ADR's
    paper makes that comparison cross-codebase; running it here makes it a measurement.

    The method, from `src/adr.py` (49 lines) and `src/advTrainer.py`.  An EMA of the model itself is
    the teacher -- there is no external network.  Its softened prediction is interpolated with the
    one-hot label, per sample, by an amount that shrinks when the teacher is confidently wrong:

        p_t     = softmax(teacher(x) / T(step))
        lambda  = clamp(interp(step) - (max_c p_t[c] - p_t[y]), 0, 1)
        target  = lambda * p_t + (1 - lambda) * onehot(y)

    so a sample the teacher already ranks correctly keeps most of the teacher's distribution, and one
    where some other class dominates falls back toward the hard label.  T anneals 2.0 -> 1.0 and the
    interpolation ceiling 0.7 -> 0.85 on cosine schedules across the whole run.  The rectified target
    replaces the hard label in BOTH the attack and the loss, which is what `advTrainer` does.

    Defaults are ADR's gin config for `resnet18_pgd_awp_adr`: ema decay 0.995, T 2.0 -> 1.0, interp
    0.7 -> 0.85.  ADR trains 200 epochs with SGD 0.1 and steps at [100, 150]; the config carries that
    rather than our 100-epoch protocol, because the point of this cell is to reproduce their number
    and a shortened schedule would confound the comparison.

    ADR's AWP is SAM-style (first/second step at rho 0.005), so `awp_style: sam` matches their code;
    our proxy style is a different mechanism and would be a silent substitution.
    """
    model.train()
    eps_train = float(getattr(config, "train_eps", 0.0) or 0.0) or config.eps
    wa_start = _resolve_epoch_arg(getattr(config, "wa_start", None), config.epochs) or 0
    n_cls = int(getattr(config, "num_classes", 0) or (100 if "100" in str(config.dataset) else
                                                      200 if "Tiny" in str(config.dataset) else 10))
    ema_decay = float(getattr(config, "adr_ema_decay", 0.0) or 0.0) or 0.995
    t_hi = float(getattr(config, "adr_temp_high", 0.0) or 0.0) or 2.0
    t_lo = float(getattr(config, "adr_temp_low", 0.0) or 0.0) or 1.0
    i_lo = float(getattr(config, "adr_interp_low", 0.0) or 0.0) or 0.7
    i_hi = float(getattr(config, "adr_interp_high", 0.0) or 0.0) or 0.85

    total_steps = config.epochs * len(train_loader)
    if not hasattr(train_adr, "_sched") or train_adr._total != total_steps:
        train_adr._temp = _cosine_sched(t_hi, t_lo, total_steps)
        train_adr._interp = _cosine_sched(i_lo, i_hi, total_steps)
        train_adr._sched, train_adr._total = True, total_steps
    # ADR's teacher is an EMA of the student kept for the whole run, separate from the weight-averaged
    # shadow `exp_avg` that main.py evaluates: different decay, and it is READ every step rather than
    # only written.  Keeping them separate is deliberate -- sharing one would silently change both.
    if not hasattr(train_adr, "_ema") or train_adr._ema_epochs != config.epochs:
        train_adr._ema = copy.deepcopy(model).eval()
        for _p in train_adr._ema.parameters():
            _p.requires_grad_(False)
        train_adr._ema_epochs = config.epochs
    ema = train_adr._ema

    awp_style = str(getattr(config, "awp_style", "proxy") or "proxy")
    awp_gamma = float(getattr(config, "awp_gamma", 0.0) or 0.0)
    use_awp = awp_gamma > 0 and epoch >= int(getattr(config, "awp_warmup", 0) or 0)
    if use_awp:
        if awp_style == "sam":                    # ADR's own AWP: first_step / second_step at rho
            if not hasattr(train_adr, "_awp_sam"):
                train_adr._awp_sam = AdvWeightPerturbSAM(model, rho=awp_gamma)
            awp = train_adr._awp_sam
            awp.rho = awp_gamma
        else:
            if not hasattr(train_adr, "_awp"):
                train_adr._awp = AdvWeightPerturb(model, gamma=awp_gamma)
            awp = train_adr._awp
            awp.gamma = awp_gamma

    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        x, y = x.cuda(), y.cuda()
        step = epoch * len(train_loader) + batch_idx
        with torch.no_grad():
            p_t = F.softmax(ema(x) / train_adr._temp(step), dim=1)
            onehot = F.one_hot(y, num_classes=n_cls).float()
            gap = p_t.max(dim=1).values - p_t.gather(1, y.unsqueeze(1)).squeeze(1)
            lam = torch.clamp(train_adr._interp(step) - gap, 0.0, 1.0).unsqueeze(1)
            target = lam * p_t + (1.0 - lam) * onehot

        x_adv = _pgd_attack_soft_label(model, x, target, config.step_size, eps_train,
                                       perturb_steps=config.steps)
        if use_awp and awp_style == "sam":
            # ADR's own order: ascend from the gradient at clean weights, measure the gradient the
            # optimizer applies at the perturbed weights, restore BEFORE optimizer.step().  The two
            # AWP classes have different call patterns, so this branch is not cosmetic.
            optimizer.zero_grad()
            _soft_ce(model(x_adv), target).backward()
            awp.first_step()
            _awp_bn_disable(model)
            optimizer.zero_grad()
            loss = _soft_ce(model(x_adv), target)
            loss.backward()
            awp.restore()
            _awp_bn_enable(model)
            optimizer.step()
        elif use_awp:
            def _awp_loss_fn(pm, _xa=x_adv, _t=target):
                return _soft_ce(pm(_xa), _t)
            awp_diff = awp.calc_awp(_awp_loss_fn)
            awp.perturb(awp_diff)
            optimizer.zero_grad()
            loss = _soft_ce(model(x_adv), target)
            loss.backward()
            optimizer.step()
            awp.restore(awp_diff)
        else:
            optimizer.zero_grad()
            loss = _soft_ce(model(x_adv), target)
            loss.backward()
            optimizer.step()
        scheduler.step()
        with torch.no_grad():                                  # ADR's own EMA teacher update
            msd = model.state_dict()
            for k, v in ema.state_dict().items():
                v.copy_(v * ema_decay + msd[k].detach() * (1.0 - ema_decay)
                        if v.dtype.is_floating_point else msd[k])
        if config.weight_avg == True and epoch >= wa_start:
            annealing = (epoch / config.epochs) ** 2
            decay = annealing * (1 - config.kappa) + config.kappa
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 - decay) * value + decay * exp_avg[key]


def train_hat(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """HAT, Helper-based Adversarial Training (Rade & Moosavi-Dezfooli, ICLR'22), ported from the
    authors' `core/utils/hat.py`.

    HAT belongs with the trade-off methods, not with the distillation ones, even though it consumes a
    naturally trained network like we do.  The natural model supplies a LABEL for a point beyond the
    ball and never a target the student matches, so none of its representation transfers:

        x_hr = x + h (x_adv - x)                       helper point, h = 2.0
        y_hr = argmax std_model(x_adv)                 label read at x_adv, NOT at x or at x_hr
        L    = CE(f(x), y) + beta KL(f(x_adv) || f(x)) + gamma CE(f(x_hr), y_hr)

    That is TRADES plus a helper term.  The reading is that adversarial training pushes the boundary
    further than it needs to; the helper term pulls it back by asserting that at 2 delta the model
    should already be wrong, and in the particular way a standard model is wrong.

    The read point of `y_hr` is worth stating because it is easy to get wrong and it changes the
    objective: the label comes from the standard model evaluated at x_adv, then supervises the
    student at x_hr.  `at_hat_loss` in their repo is the PGD-AT variant; the published CIFAR results
    use the TRADES form ported here.

    Defaults are their CIFAR-10 ResNet-18 command: beta 2.5, gamma 0.5, h 2.0 (their parser's default;
    the 3.5 in the function signature is not what the README passes).  `origin_model` is our natural
    teacher, which is exactly what their `--helper-model std-cifar10` is -- 50 epochs of plain CE.
    """
    model.train()
    origin_model.eval()
    eps_train = float(getattr(config, "train_eps", 0.0) or 0.0) or config.eps
    wa_start = _resolve_epoch_arg(getattr(config, "wa_start", None), config.epochs) or 0
    beta = float(getattr(config, "beta", 0.0) or 0.0) or 2.5
    gamma = float(getattr(config, "hat_gamma", 0.0) or 0.0) or 0.5
    h = float(getattr(config, "hat_h", 0.0) or 0.0) or 2.0
    criterion_kl = nn.KLDivLoss(reduction='sum')

    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        x, y = x.cuda(), y.cuda()
        x_adv = _trades_inner_attack(model, x, config.step_size, eps_train,
                                     perturb_steps=config.steps)
        x_hr = x + h * (x_adv - x)                      # NOT clamped: their code does not clamp it
        with torch.no_grad():
            y_hr = origin_model(x_adv).argmax(dim=1)

        optimizer.zero_grad()
        out_clean, out_adv, out_help = model(x), model(x_adv), model(x_hr)
        loss = (F.cross_entropy(out_clean, y)
                + beta * (1.0 / x.size(0)) * criterion_kl(F.log_softmax(out_adv, dim=1),
                                                          F.softmax(out_clean, dim=1))
                + gamma * F.cross_entropy(out_help, y_hr))
        loss.backward()
        optimizer.step()
        scheduler.step()
        if config.weight_avg == True and epoch >= wa_start:
            annealing = (epoch / config.epochs) ** 2
            decay = annealing * (1 - config.kappa) + config.kappa
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 - decay) * value + decay * exp_avg[key]


def train_lbgat(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """LBGAT, Learnable Boundary Guided Adversarial Training (Cui et al., ICCV'21), ported from the
    authors' `lbgat.py`.

    The closest published method to ours by mechanism, which is why it is worth having as a measured
    row rather than a citation.  It reads a naturally trained network at the CLEAN point and matches
    it with a squared error -- no softmax, no temperature:

        L = MSE(f_s(x_adv), f_t(x)) + CE(f_t(x), y) + beta KL(f_s(x_adv) || f_s(x))

    Two differences from an anchor remain, and they are the comparison this row exists to make.  The
    target is the LOGIT, read after the classifier, where ours is the feature one layer earlier.  And
    the second term trains the natural branch JOINTLY -- `main.py` puts the teacher in the optimizer
    when `joint_teacher: True` -- so the network supplying the target is a second model paid for
    during training rather than one obtained beforehand.  Freezing it would be a different and weaker
    method, so the port keeps the joint training even though it makes the cost claim asymmetric.

    The attack is TRADES-style: KL between the student at x_adv and the student at x, teacher not
    involved.  beta = 6 is what their CIFAR-100 script uses ("lbgat6" in their checkpoint names).
    """
    model.train()
    origin_model.train()                    # the natural branch is being trained, not evaluated
    eps_train = float(getattr(config, "train_eps", 0.0) or 0.0) or config.eps
    wa_start = _resolve_epoch_arg(getattr(config, "wa_start", None), config.epochs) or 0
    # 2026-09-05: this was `float(getattr(config, "beta", 0.0) or 0.0) or 6.0`, in which beta = 0 is
    # unreachable -- `0.0 or 6.0` is 6.0 -- so LBGAT0 could not be run at all.  That matters because
    # LBGAT0 is the only CIFAR-10 configuration the authors publish (their README lists CIFAR-10 under
    # LBGAT0 alone, and their own script is `--beta 0`); beta = 6 is the CIFAR-100 setting.  Asking
    # for beta = 0 silently trained beta = 6, and the CIFAR-10 cell sat at chance from epoch 0.
    _b = getattr(config, "beta", None)
    beta = 6.0 if _b is None else float(_b)
    criterion_kl = nn.KLDivLoss(reduction='sum')
    mse = nn.MSELoss()

    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        x, y = x.cuda(), y.cuda()
        x_adv = _trades_inner_attack(model, x, config.step_size, eps_train,
                                     perturb_steps=config.steps)
        model.train(); origin_model.train()
        optimizer.zero_grad()
        out_adv, out_nat = model(x_adv), model(x)
        out_t = origin_model(x)
        loss = (mse(out_adv, out_t) + F.cross_entropy(out_t, y)
                + beta * (1.0 / out_t.size(0)) * criterion_kl(F.log_softmax(out_adv, dim=1),
                                                              F.softmax(out_nat, dim=1)))
        loss.backward()
        optimizer.step()
        scheduler.step()
        if config.weight_avg == True and epoch >= wa_start:
            annealing = (epoch / config.epochs) ** 2
            decay = annealing * (1 - config.kappa) + config.kappa
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 - decay) * value + decay * exp_avg[key]


def _jensen_shannon_div(logit1, logit2, T=1.0):
    """JS divergence between two softened logit distributions, as RPAT's benchmark computes it."""
    prob1 = F.softmax(logit1 / T, dim=1)
    prob2 = F.softmax(logit2 / T, dim=1)
    mean_prob = 0.5 * (prob1 + prob2)
    logsoftmax = torch.log(mean_prob.clamp(min=1e-8))
    jsd = F.kl_div(logsoftmax, prob1, reduction='batchmean')
    jsd += F.kl_div(logsoftmax, prob2, reduction='batchmean')
    return jsd * 0.5


def train_consistency(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """Consistency-AT (Tack et al., 2022), ported from RPAT's benchmark implementation.

    Why this one and not the rest of RPAT's leaderboard.  Consistency-AT is the strongest of the four
    methods RPAT benchmarks against on the axis we care about -- 58.53 clean on CIFAR-100 against
    PGD-AT's 56.56 -- and it earns that with a consistency term rather than a teacher, so it is the
    natural "you can raise clean accuracy without distilling at all" control.  The reweighting family
    (GAIRAT, MAIL, EWAT, SOVR, MMA) sits far enough below on RPAT's own numbers that running it would
    only pad the table.  ReBAT is skipped for a different reason: RPAT++ dominates it on both axes in
    RPAT's Table 3, and we already reproduce RPAT++.

    The objective.  Two independent augmentations of each image are drawn by the loader, adversarial
    examples are built for both against the true label, and to the cross-entropy on those we add a
    Jensen-Shannon divergence between the two adversarial outputs:

        L = CE(f(x'_1), y) + CE(f(x'_2), y) + lam * JSD_T(f(x'_1), f(x'_2))

    Defaults are RPAT's: lam = 1.0 (config.lamda), T = 0.5 (config.tau_consistency).  Note T < 1
    SHARPENS rather than softens, which is the opposite of the distillation temperatures elsewhere in
    this file; it is theirs and we keep it.

    Cost.  The batch is doubled before the attack, so a step costs about twice a PGD-AT step.  Both
    views are attacked together in one call, exactly as the reference does, so the attack sees the
    concatenated batch and not two separate ones.

    Stack knobs (train_eps, wa_start, AWP) are honoured for the same reason train_madry_at honours
    them: the cell has to be runnable at the same regime as everything it is compared with.
    """
    model.train()
    eps_train = float(getattr(config, "train_eps", 0.0) or 0.0) or config.eps
    wa_start = _resolve_epoch_arg(getattr(config, "wa_start", None), config.epochs) or 0
    lam = float(getattr(config, "lamda", 0.0) or 0.0) or 1.0
    T = float(getattr(config, "tau_consistency", 0.0) or 0.0) or 0.5
    awp_gamma = float(getattr(config, "awp_gamma", 0.0) or 0.0)
    use_awp = awp_gamma > 0 and epoch >= int(getattr(config, "awp_warmup", 0) or 0)
    if use_awp:
        if not hasattr(train_consistency, "_awp"):
            train_consistency._awp = AdvWeightPerturb(model, gamma=awp_gamma)
        awp = train_consistency._awp
        awp.gamma = awp_gamma
    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise RuntimeError(
                "train_consistency needs paired views; set `two_views: True` in the config so "
                "dataset.MultiDataTransform is applied (see dataset.py)")
        x = torch.cat([x[0].cuda(), x[1].cuda()], dim=0)     # 2B
        y = y.cuda()
        y2 = y.repeat(2)
        x_adv = _pgd_attack_true_label(model, x, y2, config.step_size, eps_train,
                                       perturb_steps=config.steps)

        def _loss_from(pm, _xa=None, _y2=None):
            out = pm(x_adv if _xa is None else _xa)
            tgt = y2 if _y2 is None else _y2
            o1, o2 = out.chunk(2)
            return F.cross_entropy(out, tgt) + lam * _jensen_shannon_div(o1, o2, T)

        if use_awp:
            awp_diff = awp.calc_awp(_loss_from)
            awp.perturb(awp_diff)
        optimizer.zero_grad()
        loss = _loss_from(model)
        loss.backward()
        optimizer.step()
        if use_awp:
            awp.restore(awp_diff)
        scheduler.step()
        if config.weight_avg == True and epoch >= wa_start:
            annealing = (epoch / config.epochs) ** 2
            decay = annealing * (1 - config.kappa) + config.kappa
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 - decay) * value + decay * exp_avg[key]


def train_temperature_tpnorm(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """TEACHER-side Lq-norm POWER sweep on top of the norm student (user's idea, 2026-07-06 night).

    Record check that motivated this: the old powernorm p-sweep (Phi/||Phi||^p optimal at p=1) was run
    with a RAW student during the un-normalization quest -- the combination "norm student + teacher
    ||Phi||^p" was never swept, and non-L2 norm orders q were never tried anywhere. This closes both.

    target = teacher_logits / ( tau * r(x) ),   r(x) = ( ||Phi_t(x)||_q / batch_mean(||Phi_t||_q) )^p
      - p = config.gamma (power), q = config.eta (norm order; eta >= 99 -> L-infinity).
      - MATCHED EFFECTIVE TEMPERATURE by construction: r is mean-normalized per batch, so the batch's
        average temperature stays ~tau for ANY (p,q) -- p only reshapes WHICH samples get softer or
        sharper (p>0: big-norm softer; p<0: inverted; p=0: r==1 == train_temperature EXACTLY). This is
        the matched-softness control the earlier per-sample-temp experiments taught us to build in.
      - Since teacher logits are linear in Phi, dividing logits by ||Phi||_2 IS W(Phi/||Phi||_2); for
        q != 2 this is the scalar-equivalent of an Lq-normalized teacher feature.
    Expectation is honest-flat (every teacher-side per-sample SCALAR has died), but the (norm-student x
    p-power x Lq) cells are genuinely unswept and the dose-response curve is the cheap decisive test.
    Sweep via --gamma (p) and --eta (q); both logged.
    """
    model.train()
    origin_model.eval()
    annealing = (epoch / config.epochs) ** 2
    decay = annealing * (1 - config.kappa) + config.kappa
    criterion_kl = nn.KLDivLoss(reduction='none')
    p_pow = float(config.gamma) if config.gamma is not None else 0.0
    q_ord = float(config.eta) if config.eta is not None else 2.0
    q_ord = float('inf') if q_ord >= 99 else q_ord

    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        N, C, H, W = x.shape
        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()

        with torch.no_grad():
            teacher_feat, teacher_logits = origin_model(x, feat=True)
            nq = teacher_feat.norm(p=q_ord, dim=1, keepdim=True)
            # GEOMETRIC-mean anchor (log-centering, same convention as taunet): temperature is a
            # multiplicative quantity, so arithmetic mean-normalization of r drifts the effective
            # average softness as |p| grows (Jensen gap). This pins geomean(T)==tau EXACTLY for
            # every (p,q), so the sweep moves per-sample SHAPE only, never global softness.
            logn = nq.clamp_min(1e-12).log()
            r = torch.exp(p_pow * (logn - logn.mean()))
            target = (teacher_logits / (config.tau * r)).detach()

        x_pgd = inner_loss_only_return(model, target, x, y, optimizer, config.step_size, config.eps, perturb_steps=config.steps)
        plus_logits = model(x_pgd)
        kl_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(target, dim=1))
        loss = (1.0 / N) * (kl_loss.sum(dim=1)).sum()

        student_logits = model(x)   # clean forward: updates student BN running stats
        if config.lamda is not None and config.lamda > 0:
            consistency_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(student_logits, dim=1))
            loss += annealing * config.lamda * (consistency_loss).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()

        if config.weight_avg == True:
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 - decay) * value + decay * exp_avg[key]


def train_temperature_dualkd(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """train_temperature + an EXPLICIT clean-image KD term (user's idea): currently `student_logits =
    model(x)` is computed ONLY to update BN running stats -- the student's CLEAN prediction is never
    directly supervised to match the teacher target at all (the lamda consistency term matches x_adv
    to the student's OWN clean prediction, not to the teacher). This adds:
        loss += beta * KL( student(x) || target )     [target = teacher(x)/tau, same fixed clean target]
    beta=0 reduces EXACTLY to train_temperature. Hypothesis: directly anchoring student(x) to the
    teacher lifts clean acc without touching the x_adv term that carries robustness.
    """
    model.train()
    origin_model.eval()
    annealing = (epoch / config.epochs) ** 2
    decay = annealing * (1 - config.kappa) + config.kappa
    criterion_kl = nn.KLDivLoss(reduction='none')
    beta = config.beta if config.beta is not None else 0.0

    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        N, C, H, W = x.shape
        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()

        with torch.no_grad():
            _, teacher_logits = origin_model(x, feat=True)      # raw teacher logits = linear(Phi)
            target = (teacher_logits / config.tau).detach()     # global temperature; tau=1 == raw teacher

        x_pgd = inner_loss_only_return(model, target, x, y, optimizer, config.step_size, config.eps, perturb_steps=config.steps)
        plus_logits = model(x_pgd)
        kl_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(target, dim=1))
        loss = (1.0 / N) * (kl_loss.sum(dim=1)).sum()

        student_logits = model(x)   # clean forward: updates student BN running stats (matches train_temperature)
        if beta > 0:
            clean_kd_loss = criterion_kl(F.log_softmax(student_logits, dim=1), F.softmax(target, dim=1))
            loss += beta * clean_kd_loss.sum(dim=1).mean()

        if config.lamda is not None and config.lamda > 0:
            consistency_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(student_logits, dim=1))
            loss += annealing * config.lamda * (consistency_loss).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()

        if config.weight_avg == True:
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 - decay) * value + decay * exp_avg[key]


def train_temperature_std(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """STANDARD Hinton KD temperature (control for the asymmetric train_temperature):
      - BOTH teacher AND student logits divided by T before softmax
      - KD loss scaled by T^2
      - inner PGD is INLINED (not the shared inner_loss_only_return) so the attack ALSO uses student/T,
        keeping inner & outer objectives consistent.
    T = config.tau. student L2-norm, teacher raw. NOTE: with the cosine head (feat_scale=1) student logits
    are already ~O(1), so /T over-softens the student -> this tests whether textbook KD scaling even fits
    this architecture (vs the asymmetric version where feat_scale is the student-side temperature).
    """
    model.train()
    origin_model.eval()
    annealing = (epoch / config.epochs) ** 2
    decay = annealing * (1 - config.kappa) + config.kappa
    criterion_kl = nn.KLDivLoss(reduction='none')
    T = config.tau

    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        N = x.shape[0]
        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()

        with torch.no_grad():
            _, teacher_logits = origin_model(x, feat=True)
            soft_target = F.softmax(teacher_logits / T, dim=1).detach()      # teacher /T

        # inner PGD: maximize the SAME standard-KD KL (student also /T) vs the soft target
        model.eval()
        x_adv = x.detach() + 0.001 * torch.randn_like(x).detach()
        for _ in range(config.steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                lk = F.kl_div(F.log_softmax(model(x_adv) / T, dim=1), soft_target, reduction='sum')
            grad = torch.autograd.grad(lk, [x_adv])[0]
            x_adv = x_adv.detach() + config.step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x - config.eps), x + config.eps).clamp(0.0, 1.0)
        model.train()
        optimizer.zero_grad()

        plus_logits = model(x_adv)
        kl_loss = criterion_kl(F.log_softmax(plus_logits / T, dim=1), soft_target)   # student /T
        loss = (T * T) * (1.0 / N) * (kl_loss.sum(dim=1)).sum()                       # standard T^2 scaling

        student_logits = model(x)
        if config.lamda is not None and config.lamda > 0:
            consistency_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(student_logits, dim=1))
            loss += annealing * config.lamda * (consistency_loss).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()

        if config.weight_avg == True:
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 - decay) * value + decay * exp_avg[key]


def train_temperature_vuln(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """Per-sample VULNERABILITY-modulated temperature (input-wise; diagnostic-greenlit: vulnerability is
    ORTHOGONAL to confidence, Spearman 0.13, unlike entropy/margin ~0.99). v(x) = KL(softmax(teacher(x)) ||
    softmax(teacher(x_adv))) via a cheap 2-step random-start teacher PGD = how much this sample's prediction
    moves under attack. Per-sample temperature:  T(x) = T0 * exp(gamma * zscore_batch(v))
        gamma > 0 -> vulnerable samples SOFTER (hedge on the fragile teacher)
        gamma < 0 -> vulnerable samples SHARPER (stronger correct signal where it matters)
    swap (vuln_swap, default True) rectifies the target so the true label is the top logit (commutes with
    /Tx). gamma=0 & swap on == swap-baseline (42.17 control); gamma=0 & swap off == plain temp (41.62).
    knobs: tau=T0, gamma (signed vuln strength), vuln_swap. student L2-norm, teacher raw.
    """
    model.train()
    origin_model.eval()
    annealing = (epoch / config.epochs) ** 2
    decay = annealing * (1 - config.kappa) + config.kappa
    criterion_kl = nn.KLDivLoss(reduction='none')
    T0 = config.tau
    gamma = getattr(config, "gamma", 0.0) or 0.0
    use_swap = bool(getattr(config, "vuln_swap", True))

    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        N = x.shape[0]
        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()

        with torch.no_grad():
            _, teacher_logits = origin_model(x, feat=True)
            p_clean = F.softmax(teacher_logits, dim=1)

        # vulnerability: cheap 2-step random-start PGD on the TEACHER (maximize CE)
        xa = (x + (torch.rand_like(x) * 2 - 1) * config.eps).clamp(0.0, 1.0).detach()
        for _ in range(2):
            xa.requires_grad_(True)
            with torch.enable_grad():
                _, za = origin_model(xa, feat=True)
                ce = F.cross_entropy(za, y)
            g = torch.autograd.grad(ce, xa)[0]
            xa = torch.min(torch.max(xa.detach() + (config.eps / 2) * g.sign(), x - config.eps), x + config.eps).clamp(0.0, 1.0)
        with torch.no_grad():
            _, za = origin_model(xa, feat=True)
            p_adv = F.softmax(za, dim=1)
            v = (p_clean * ((p_clean + 1e-12).log() - (p_adv + 1e-12).log())).sum(1)   # KL(clean||adv) per sample
            u = ((v - v.mean()) / (v.std() + 1e-6)).clamp(-2.0, 2.0)                     # z-scored, clamped
            Tx = (T0 * torch.exp(gamma * u)).reshape(-1, 1)                              # always positive
            tgt = rectify_swap(teacher_logits, y) if use_swap else teacher_logits        # swap commutes with /Tx
            target = (tgt / Tx).detach()

        x_pgd = inner_loss_only_return(model, target, x, y, optimizer, config.step_size, config.eps, perturb_steps=config.steps)
        plus_logits = model(x_pgd)
        kl_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(target, dim=1))
        loss = (1.0 / N) * (kl_loss.sum(dim=1)).sum()

        student_logits = model(x)
        if config.lamda is not None and config.lamda > 0:
            consistency_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(student_logits, dim=1))
            loss += annealing * config.lamda * (consistency_loss).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()

        if config.weight_avg == True:
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 - decay) * value + decay * exp_avg[key]


def train_temperature_gradnorm(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """Per-sample TEACHER INPUT-GRADIENT-NORM-modulated temperature (user's idea, 2026-07-05, their
    stated favorite of the teacher-only signals). g(x) = ||d CE(teacher(x), y) / dx||_2 at the CLEAN
    point (single backward pass, no attack needed) -- a large gradient norm means the teacher's loss
    surface is locally steep at this input, a plausible proxy for "how much non-robust-feature content
    this sample carries" (small input perturbations would swing the loss a lot). The STUDENT NEVER
    APPEARS in this signal, matching the project's philosophy directly (per user, 2026-07-05): keep the
    natural teacher's clean knowledge maximally intact, and let any calibration toward robustness come
    ONLY from the teacher's OWN robustness-relevant properties -- never from watching what the student
    currently gets right or wrong (which is exactly the self-referential "curve-fitting" risk this whole
    teacher-only family of signals (see also train_temperature_vuln) is designed to avoid).

    T(x) = T0 * exp(gamma * zscore_batch(g))
        gamma > 0 -> high-gradient-norm (locally fragile) samples get a SOFTER target
        gamma < 0 -> high-gradient-norm samples get a SHARPER target
    swap (vuln_swap, default True) rectifies the target so the true label is the top logit (commutes
    with /Tx). gamma=0 & swap on == swap-baseline (42.17 control); gamma=0 & swap off == plain temp.
    knobs: tau=T0, gamma (signed strength), vuln_swap. Mirrors train_temperature_vuln's structure/knobs
    exactly -- only the per-sample signal differs (input-gradient-norm vs. attacked-teacher KL).
    """
    model.train()
    origin_model.eval()
    annealing = (epoch / config.epochs) ** 2
    decay = annealing * (1 - config.kappa) + config.kappa
    criterion_kl = nn.KLDivLoss(reduction='none')
    T0 = config.tau
    gamma = getattr(config, "gamma", 0.0) or 0.0
    use_swap = bool(getattr(config, "vuln_swap", True))

    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        N = x.shape[0]
        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()

        # teacher input-gradient norm at the CLEAN point (no attack -- one backward pass through the teacher)
        x_req = x.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            _, z_req = origin_model(x_req, feat=True)
            ce = F.cross_entropy(z_req, y)
        g_in = torch.autograd.grad(ce, x_req)[0]
        gnorm = g_in.flatten(1).norm(dim=1)   # [N], student never touched

        with torch.no_grad():
            _, teacher_logits = origin_model(x, feat=True)
            u = ((gnorm - gnorm.mean()) / (gnorm.std() + 1e-6)).clamp(-2.0, 2.0)   # z-scored, clamped
            Tx = (T0 * torch.exp(gamma * u)).reshape(-1, 1)                        # always positive
            tgt = rectify_swap(teacher_logits, y) if use_swap else teacher_logits  # swap commutes with /Tx
            target = (tgt / Tx).detach()

        x_pgd = inner_loss_only_return(model, target, x, y, optimizer, config.step_size, config.eps, perturb_steps=config.steps)
        plus_logits = model(x_pgd)
        kl_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(target, dim=1))
        loss = (1.0 / N) * (kl_loss.sum(dim=1)).sum()

        student_logits = model(x)
        if config.lamda is not None and config.lamda > 0:
            consistency_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(student_logits, dim=1))
            loss += annealing * config.lamda * (consistency_loss).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()

        if config.weight_avg == True:
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 - decay) * value + decay * exp_avg[key]


def train_smooth_temp(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """eps-ball SMOOTHED teacher target + global temperature (golden-logit direction).

    Instead of distilling to the teacher's single clean point (which is non-robust / varies fast
    over the eps-ball), distill to the RANDOMIZED-SMOOTHING teacher: average the teacher logits over
    K uniform perturbations in the L_inf eps-ball, then soften by T. Diagnostic: this target function
    is ~4x FLATTER over the ball (lower local KL) than plain temperature at matched softness = more
    robust-shaped by construction. Student L2-normalizes (student_norm=True), teacher raw, NO /13.

    knobs: temperature = T (softness), smooth_k = K (ball samples; K=1 == plain temperature, more K =
    smoother but K extra teacher forward passes -> the 'fast version drops it' resource lever).
    """
    model.train()
    origin_model.eval()
    annealing = (epoch / config.epochs) ** 2
    decay = annealing * (1 - config.kappa) + config.kappa
    criterion_kl = nn.KLDivLoss(reduction='none')
    T = getattr(config, "temperature", 1.0) or 1.0
    K = int(getattr(config, "smooth_k", 8) or 8)

    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        N = x.shape[0]
        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()

        with torch.no_grad():
            zsum = 0
            for _ in range(K):
                d = (torch.rand_like(x) * 2 - 1) * config.eps      # uniform noise in the L_inf eps-ball
                _, zk = origin_model((x + d).clamp(0.0, 1.0), feat=True)
                zsum = zsum + zk
            z_smooth = zsum / K                                    # eps-ball averaged teacher logits
            target = (z_smooth / T).detach()                       # + global temperature softening

        x_pgd = inner_loss_only_return(model, target, x, y, optimizer, config.step_size, config.eps, perturb_steps=config.steps)
        plus_logits = model(x_pgd)
        kl_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(target, dim=1))
        loss = (1.0 / N) * (kl_loss.sum(dim=1)).sum()

        student_logits = model(x)   # clean forward: keeps student BN running stats in sync
        if config.lamda is not None and config.lamda > 0:
            consistency_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(student_logits, dim=1))
            loss += annealing * config.lamda * (consistency_loss).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()

        if config.weight_avg == True:
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 - decay) * value + decay * exp_avg[key]


def train_smooth_temp_swap(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """GOLDEN-LOGIT combo: eps-ball smoothed teacher (FLAT) + swap-rectify (CORRECT) + temperature (SOFT).
    = train_smooth_temp with rectify_swap on the smoothed logits before /T. All three golden conditions.
    knobs: temperature=T, smooth_k=K. student L2-norm, teacher raw, NO /13."""
    model.train()
    origin_model.eval()
    annealing = (epoch / config.epochs) ** 2
    decay = annealing * (1 - config.kappa) + config.kappa
    criterion_kl = nn.KLDivLoss(reduction='none')
    T = getattr(config, "temperature", 1.0) or 1.0
    K = int(getattr(config, "smooth_k", 8) or 8)

    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        N = x.shape[0]
        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()

        with torch.no_grad():
            zsum = 0
            for _ in range(K):
                d = (torch.rand_like(x) * 2 - 1) * config.eps
                _, zk = origin_model((x + d).clamp(0.0, 1.0), feat=True)
                zsum = zsum + zk
            z_smooth = zsum / K                                    # eps-ball averaged (FLAT)
            target = (rectify_swap(z_smooth, y) / T).detach()      # swap (CORRECT) then /T (SOFT)

        x_pgd = inner_loss_only_return(model, target, x, y, optimizer, config.step_size, config.eps, perturb_steps=config.steps)
        plus_logits = model(x_pgd)
        kl_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(target, dim=1))
        loss = (1.0 / N) * (kl_loss.sum(dim=1)).sum()

        student_logits = model(x)
        if config.lamda is not None and config.lamda > 0:
            consistency_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(student_logits, dim=1))
            loss += annealing * config.lamda * (consistency_loss).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()

        if config.weight_avg == True:
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 - decay) * value + decay * exp_avg[key]


def train_carve_decorr_l1(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """Decorrelated carve, L1 fragility -- separate method (refactor later).

    Plain carve down-weights fragile channels, but fragility is ~0.49 correlated with class-need
    (contribution to the true-class logit), so it mostly just softens class info. This variant
    PROTECTS class-relevant channels and carves only the vulnerable-but-class-IRRELEVANT ones:

        fragility  = |Phi_t(x) - Phi_t(x_adv)|                       (L1; PGD-2 teacher carve)
        class_need = |W[pred] * Phi_t(x)|                            (per-dim contribution to teacher's OWN top class)
        need_rel   = class_need / mean_dim(class_need)               (per-sample normalized, mean=1)
        gate       = exp(-beta * need_rel)                           (class-relevant dim -> gate->0, protected)
        w          = exp(-tau * fragility * gate)                    (carve only vulnerable & class-irrelevant)

    class_need uses the teacher's PREDICTED class (argmax), NOT the ground-truth label -- using the
    true label leaks it into the target (carved acc jumps above the teacher's own ceiling, sharpening
    toward truth and corrupting the soft dark-knowledge). pred keeps it honest denoising.

    Two knobs: tau = fragility carve strength, beta = class-protection strength.
    beta=0 => gate=1 => EXACTLY train_carve_only_l1 (plain carve); so a beta sweep IS the ablation.
    Student untouched (student_norm=True -> ResNet18_z, L2-normalized). NO /13, NO teacher norm.
    """
    model.train()
    origin_model.eval()
    annealing = (epoch / config.epochs) ** 2
    decay = annealing * (1 - config.kappa) + config.kappa
    criterion_kl = nn.KLDivLoss(reduction='none')

    tau = getattr(config, "tau", 1.0)                       # fragility carve strength
    beta = getattr(config, "beta", 1.0)                     # class-protection strength (0 == plain carve)
    csteps = int(getattr(config, "gamma", 2) or 2)          # carve PGD steps (2 == PGD-2)
    cstep = config.eps / csteps
    lin_w = (origin_model.encoder.linear.weight if hasattr(origin_model, "encoder")
             else origin_model.linear.weight)              # [num_classes, feat_dim]

    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        N = x.shape[0]
        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()

        with torch.no_grad():
            teacher_feat, teacher_logits = origin_model(x, feat=True)   # raw teacher clean feature Phi_t(x)
            pred = teacher_logits.argmax(dim=1)                         # teacher's OWN top class (no label leak)

        # --- carve: gamma-step PGD on the teacher to expose fragile channels ---
        x_adv = x.clone().detach()
        for _ in range(csteps):
            x_adv.requires_grad_(True)
            _, logits_adv = origin_model(x_adv, feat=True)
            ce = F.cross_entropy(logits_adv, y)
            grad = torch.autograd.grad(ce, x_adv)[0]
            x_adv = x_adv.detach() + cstep * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x - config.eps), x + config.eps)
            x_adv = x_adv.clamp(0.0, 1.0)

        with torch.no_grad():
            z_adv, _ = origin_model(x_adv, feat=True)
            fragility = (teacher_feat - z_adv).abs()                        # per-channel fragility (L1)
            class_need = (lin_w[pred] * teacher_feat).abs()                 # per-channel contribution to teacher's OWN class
            need_rel = class_need / (class_need.mean(dim=1, keepdim=True) + 1e-8)
            gate = torch.exp(-beta * need_rel)                             # protect class-relevant channels
            w = torch.exp(-tau * fragility * gate)                        # carve only vulnerable & class-irrelevant
            target = origin_model.linear(teacher_feat * w).detach()        # carved logits; NO /13, NO norm

        # --- student PGD-distillation to the carved target ---
        x_pgd = inner_loss_only_return(model, target, x, y, optimizer, config.step_size, config.eps, perturb_steps=config.steps)
        plus_logits = model(x_pgd)
        kl_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(target, dim=1))
        loss = (1.0 / N) * (kl_loss.sum(dim=1)).sum()

        student_logits = model(x)   # clean forward: keeps student BN running stats in sync
        if config.lamda is not None and config.lamda > 0:
            consistency_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(student_logits, dim=1))
            loss += annealing * config.lamda * (consistency_loss).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()

        if config.weight_avg == True:
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 - decay) * value + decay * exp_avg[key]


def train_carve_decorr_temp_l1(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """DECOUPLED: decorr carve (orthogonal denoise) + GLOBAL temperature (softness) -- separate method.

    Identical decorr carve as train_carve_decorr_l1 (protect teacher's own class dims, carve only
    vulnerable-&-class-irrelevant), THEN soften the carved logits by a global temperature T:

        target = teacher.linear(Phi_t(x) * w) / T          (w = decorr carve weight)

    Rationale (two diagnostics): softness is the winning lever (global temp best H 41.62), but it is
    uniform; decorr removes vulnerable dims global temp can't touch -- but decorr RE-SHARPENS the
    target. So couple them: carve denoises, T restores softness. Because decorr sharpens, T must be
    pushed HIGHER than the pure-global optimum (16) to reach the same net softness.
    Three knobs: tau = carve strength, beta = class protection, T = config.temperature (softness).
    T=1, beta=0 => plain sharp carve; beta=0 => carve_only_l1 / T. Student L2-norm, teacher raw, NO /13.
    """
    model.train()
    origin_model.eval()
    annealing = (epoch / config.epochs) ** 2
    decay = annealing * (1 - config.kappa) + config.kappa
    criterion_kl = nn.KLDivLoss(reduction='none')

    tau = getattr(config, "tau", 1.0)                       # fragility carve strength
    beta = getattr(config, "beta", 1.0)                     # class-protection strength
    T = getattr(config, "temperature", 1.0) or 1.0         # global softening temperature (sweep HIGH)
    csteps = int(getattr(config, "gamma", 2) or 2)
    cstep = config.eps / csteps
    lin_w = (origin_model.encoder.linear.weight if hasattr(origin_model, "encoder")
             else origin_model.linear.weight)

    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        N = x.shape[0]
        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()

        with torch.no_grad():
            teacher_feat, teacher_logits = origin_model(x, feat=True)
            pred = teacher_logits.argmax(dim=1)                         # teacher's own class (no label leak)

        x_adv = x.clone().detach()
        for _ in range(csteps):
            x_adv.requires_grad_(True)
            _, logits_adv = origin_model(x_adv, feat=True)
            ce = F.cross_entropy(logits_adv, y)
            grad = torch.autograd.grad(ce, x_adv)[0]
            x_adv = x_adv.detach() + cstep * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x - config.eps), x + config.eps)
            x_adv = x_adv.clamp(0.0, 1.0)

        with torch.no_grad():
            z_adv, _ = origin_model(x_adv, feat=True)
            fragility = (teacher_feat - z_adv).abs()
            class_need = (lin_w[pred] * teacher_feat).abs()
            need_rel = class_need / (class_need.mean(dim=1, keepdim=True) + 1e-8)
            gate = torch.exp(-beta * need_rel)                         # protect class-relevant channels
            w = torch.exp(-tau * fragility * gate)                    # decorr carve
            target = (origin_model.linear(teacher_feat * w) / T).detach()   # + GLOBAL temperature softening

        x_pgd = inner_loss_only_return(model, target, x, y, optimizer, config.step_size, config.eps, perturb_steps=config.steps)
        plus_logits = model(x_pgd)
        kl_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(target, dim=1))
        loss = (1.0 / N) * (kl_loss.sum(dim=1)).sum()

        student_logits = model(x)
        if config.lamda is not None and config.lamda > 0:
            consistency_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(student_logits, dim=1))
            loss += annealing * config.lamda * (consistency_loss).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()

        if config.weight_avg == True:
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 - decay) * value + decay * exp_avg[key]


def train_carve_decorr_temp_swap_l1(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """decorr carve + global temperature + teacher SWAP-rectification. = train_carve_decorr_temp_l1
    with rectify_swap applied to the carved logits (true class -> top) BEFORE /T. Compare against
    train_temperature_swap (the fair swap-baseline). knobs: tau, beta, temperature (T)."""
    model.train()
    origin_model.eval()
    annealing = (epoch / config.epochs) ** 2
    decay = annealing * (1 - config.kappa) + config.kappa
    criterion_kl = nn.KLDivLoss(reduction='none')

    tau = getattr(config, "tau", 1.0)
    beta = getattr(config, "beta", 1.0)
    T = getattr(config, "temperature", 1.0) or 1.0
    csteps = int(getattr(config, "gamma", 2) or 2)
    cstep = config.eps / csteps
    lin_w = (origin_model.encoder.linear.weight if hasattr(origin_model, "encoder")
             else origin_model.linear.weight)

    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        N = x.shape[0]
        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()

        with torch.no_grad():
            teacher_feat, teacher_logits = origin_model(x, feat=True)
            pred = teacher_logits.argmax(dim=1)

        x_adv = x.clone().detach()
        for _ in range(csteps):
            x_adv.requires_grad_(True)
            _, logits_adv = origin_model(x_adv, feat=True)
            ce = F.cross_entropy(logits_adv, y)
            grad = torch.autograd.grad(ce, x_adv)[0]
            x_adv = x_adv.detach() + cstep * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x - config.eps), x + config.eps)
            x_adv = x_adv.clamp(0.0, 1.0)

        with torch.no_grad():
            z_adv, _ = origin_model(x_adv, feat=True)
            fragility = (teacher_feat - z_adv).abs()
            class_need = (lin_w[pred] * teacher_feat).abs()
            need_rel = class_need / (class_need.mean(dim=1, keepdim=True) + 1e-8)
            gate = torch.exp(-beta * need_rel)
            w = torch.exp(-tau * fragility * gate)
            carved = origin_model.linear(teacher_feat * w)
            target = (rectify_swap(carved, y) / T).detach()   # SWAP-rectify carved logits THEN soften

        x_pgd = inner_loss_only_return(model, target, x, y, optimizer, config.step_size, config.eps, perturb_steps=config.steps)
        plus_logits = model(x_pgd)
        kl_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(target, dim=1))
        loss = (1.0 / N) * (kl_loss.sum(dim=1)).sum()

        student_logits = model(x)
        if config.lamda is not None and config.lamda > 0:
            consistency_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(student_logits, dim=1))
            loss += annealing * config.lamda * (consistency_loss).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()

        if config.weight_avg == True:
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 - decay) * value + decay * exp_avg[key]


def train_carve_only_l1(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """Carve-only teacher target -- its own clean method, NOT the transform dispatcher.

    Fragility metric = L1 (per-channel absolute deviation). (An L2 / squared-deviation variant
    is a separate method: train_carve_only_l2.) 'l1' here names the FRAGILITY metric; the student
    always L2-normalizes its own features (student_norm=True -> ResNet18_z) in both variants.

    Teacher-side operation is ONLY the carve:
      1. find fragile feature channels with a gamma-step PGD on the TEACHER (config.gamma, default 2 --
         i.e. PGD-2, NOT FGSM), maximizing teacher CE;
      2. per-channel fragility = |Phi_t(x) - Phi_t(x_adv)|  (L1);
      3. down-weight fragile channels: w = exp(-tau * fragility)  (tau = config.tau = carve strength;
         it plays the temperature role -- bigger tau -> softer/more-carved target);
      4. push the carved clean feature through the teacher's OWN linear head -> carved teacher logits.
    NO global /13 rescale, NO teacher feature-normalization -- carve is the whole teacher knob.
    Student is left untouched by carve; it is ResNet18_z (student_norm=True => L2-normalized features),
    trained by PGD-distillation to the carved target.
    """
    model.train()
    origin_model.eval()
    annealing = (epoch / config.epochs) ** 2
    decay = annealing * (1 - config.kappa) + config.kappa
    criterion_kl = nn.KLDivLoss(reduction='none')

    tau = getattr(config, "tau", 1.0)                       # carve strength (== temperature role)
    csteps = int(getattr(config, "gamma", 2) or 2)          # carve PGD steps (2 == PGD-2)
    cstep = config.eps / csteps                             # carve step size (independent of AT step_size)

    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        N = x.shape[0]
        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()

        with torch.no_grad():
            teacher_feat, _ = origin_model(x, feat=True)    # raw teacher clean feature Phi_t(x)

        # --- carve: gamma-step PGD on the teacher to expose fragile channels ---
        x_adv = x.clone().detach()
        for _ in range(csteps):
            x_adv.requires_grad_(True)
            _, logits_adv = origin_model(x_adv, feat=True)
            ce = F.cross_entropy(logits_adv, y)
            grad = torch.autograd.grad(ce, x_adv)[0]
            x_adv = x_adv.detach() + cstep * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x - config.eps), x + config.eps)
            x_adv = x_adv.clamp(0.0, 1.0)

        with torch.no_grad():
            z_adv, _ = origin_model(x_adv, feat=True)
            fragility = (teacher_feat - z_adv).abs()        # per-channel fragility
            w = torch.exp(-tau * fragility)                 # fragile channels -> ~0
            target = origin_model.linear(teacher_feat * w).detach()   # carved logits; NO /13, NO norm

        # --- student PGD-distillation to the carved target ---
        x_pgd = inner_loss_only_return(model, target, x, y, optimizer, config.step_size, config.eps, perturb_steps=config.steps)
        plus_logits = model(x_pgd)
        kl_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(target, dim=1))
        loss = (1.0 / N) * (kl_loss.sum(dim=1)).sum()

        student_logits = model(x)   # clean forward: keeps student BN running stats in sync
        if config.lamda is not None and config.lamda > 0:
            consistency_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(student_logits, dim=1))
            loss += annealing * config.lamda * (consistency_loss).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()

        if config.weight_avg == True:
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 - decay) * value + decay * exp_avg[key]


def train_carve_only_l2(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """Carve-only teacher target, L2 fragility -- separate copy of train_carve_only_l1 (refactor later).

    IDENTICAL to train_carve_only_l1 except the fragility metric is L2 / squared per-channel deviation:
        fragility = (Phi_t(x) - Phi_t(x_adv))**2     (vs. .abs() in the l1 variant).
    Squaring shrinks per-channel deviations that are < 1, so the same tau carves LESS than in l1 --
    the l2 tau sweep is scaled up accordingly. Student still L2-normalizes its own features.
    """
    model.train()
    origin_model.eval()
    annealing = (epoch / config.epochs) ** 2
    decay = annealing * (1 - config.kappa) + config.kappa
    criterion_kl = nn.KLDivLoss(reduction='none')

    tau = getattr(config, "tau", 1.0)                       # carve strength (== temperature role)
    csteps = int(getattr(config, "gamma", 2) or 2)          # carve PGD steps (2 == PGD-2)
    cstep = config.eps / csteps                             # carve step size (independent of AT step_size)

    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        N = x.shape[0]
        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()

        with torch.no_grad():
            teacher_feat, _ = origin_model(x, feat=True)    # raw teacher clean feature Phi_t(x)

        # --- carve: gamma-step PGD on the teacher to expose fragile channels ---
        x_adv = x.clone().detach()
        for _ in range(csteps):
            x_adv.requires_grad_(True)
            _, logits_adv = origin_model(x_adv, feat=True)
            ce = F.cross_entropy(logits_adv, y)
            grad = torch.autograd.grad(ce, x_adv)[0]
            x_adv = x_adv.detach() + cstep * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x - config.eps), x + config.eps)
            x_adv = x_adv.clamp(0.0, 1.0)

        with torch.no_grad():
            z_adv, _ = origin_model(x_adv, feat=True)
            fragility = (teacher_feat - z_adv) ** 2         # per-channel fragility (L2 / squared)
            w = torch.exp(-tau * fragility)                 # fragile channels -> ~0
            target = origin_model.linear(teacher_feat * w).detach()   # carved logits; NO /13, NO norm

        # --- student PGD-distillation to the carved target ---
        x_pgd = inner_loss_only_return(model, target, x, y, optimizer, config.step_size, config.eps, perturb_steps=config.steps)
        plus_logits = model(x_pgd)
        kl_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(target, dim=1))
        loss = (1.0 / N) * (kl_loss.sum(dim=1)).sum()

        student_logits = model(x)   # clean forward: keeps student BN running stats in sync
        if config.lamda is not None and config.lamda > 0:
            consistency_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(student_logits, dim=1))
            loss += annealing * config.lamda * (consistency_loss).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()

        if config.weight_avg == True:
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 - decay) * value + decay * exp_avg[key]


def train_FANM(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    model.train()

    origin_model.eval()
    annealing =  (epoch/config.epochs)**2
    decay = annealing * (1 - config.kappa) + config.kappa
    criterion_kl = nn.KLDivLoss(reduction='none')

    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        N, C, H, W = x.shape

        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()
        teacher_logits = origin_model(x)
        teacher_logits = teacher_logits/(teacher_logits.norm(dim = 1).reshape([-1,1]) * config.tau)
        x_pgd = inner_loss_only_return(model, teacher_logits, x, y, optimizer, config.step_size, config.eps, perturb_steps=config.steps)
        plus_logits = model(x_pgd)

        kl_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(teacher_logits.detach(), dim=1))
        loss = (1.0 / N) * (kl_loss.sum(dim = 1)).sum()

        student_logits = model(x)
        consistency_loss  = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(student_logits, dim=1))        
        loss += annealing * config.lamda * (consistency_loss).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()
             
        if config.weight_avg == True:
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 -decay) * value + decay * exp_avg[key]



def train_FANM2(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    model.train()

    origin_model.eval()
    annealing =  (epoch/config.epochs)**2
    decay = annealing * (1 - config.kappa) + config.kappa
    criterion_kl = nn.KLDivLoss(reduction='none')

    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        N, C, H, W = x.shape

        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()
        teacher_logits = origin_model(x)
        teacher_logits = teacher_logits/config.tau
        x_pgd = inner_loss_only_return(model, teacher_logits, x, y, optimizer, config.step_size, config.eps, perturb_steps=config.steps)
        plus_logits = model(x_pgd)

        kl_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(teacher_logits.detach(), dim=1))
        loss = (1.0 / N) * (kl_loss.sum(dim = 1)).sum()

        student_logits = model(x)
        consistency_loss  = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(student_logits, dim=1))        
        loss += annealing * config.lamda * (consistency_loss).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()
             
        if config.weight_avg == True:
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 -decay) * value + decay * exp_avg[key]



def train_teacherSharpening(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    model.train()

    origin_model.eval()
    annealing =  (epoch/config.epochs)**2
    decay = annealing * (1 - config.kappa) + config.kappa
    criterion_kl = nn.KLDivLoss(reduction='none')

    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        N, C, H, W = x.shape

        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()
        teacher_logits = origin_model(x)
        teacher_logits = teacher_logits/config.tau
        x_pgd = inner_loss_only_return(model, teacher_logits, x, y, optimizer, config.step_size, config.eps, perturb_steps=config.steps)
        plus_logits = model(x_pgd)

        kl_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(teacher_logits.detach(), dim=1))
        loss = (1.0 / N) * (kl_loss.sum(dim = 1)).sum()

        loss.backward()
        optimizer.step()
        scheduler.step()
             


def train_labelSmoothing(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    model.train()

    origin_model.eval()
    annealing =  (epoch/config.epochs)**2
    decay = annealing * (1 - config.kappa) + config.kappa
    criterion = nn.CrossEntropyLoss(label_smoothing=config.tau)
    pgd_attack = torchattacks.PGD(model,eps = config.eps, steps = 10, alpha = 2/255, random_start = True)
    # x_pgd = pgd_attack(x,y)
    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        N, C, H, W = x.shape

        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()
        teacher_logits = origin_model(x)
        teacher_logits = teacher_logits/config.tau
        x_pgd = pgd_attack(x,y)
        logits = model(x_pgd)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()
        scheduler.step()


def _cure_generate_adv(model, x_natural, step_size, epsilon, perturb_steps):
    """CURE Eq.4: maximize KL( p(x_nat) || p(x_adv) )  (TRADES-style L_inf adversary)."""
    criterion_kl = nn.KLDivLoss(reduction='sum')
    model.eval()
    out_nat = model(x_natural).detach()
    x_adv = x_natural.detach() + 0.001 * torch.randn_like(x_natural).detach()
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                   F.softmax(out_nat, dim=1))
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()
    return x_adv.detach()


def train_cure(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """Faithful port of CURE's rgp_soft path (CURE/utilities/train.py train_cure +
    cure_loss_dual + RGP.fuse_grads + update_ema_model_variables, aux_loss_type=['kl']).

    Starts from the natural model FAT loads (config.load/finetune); evaluated FAT-style
    via exp_avg, which is kept in sync with the EMA "revision" model (model_ema).
    """
    import copy
    model.train()

    # CURE LR recipe (scheduler='None'): multistep decay at config.epoch_step by lr_decay_ratio.
    if getattr(config, "epoch_step", None) is not None:
        n_drops = sum(1 for m in config.epoch_step if (epoch + 1) >= m)
        new_lr = config.lr * (config.lr_decay_ratio ** n_drops)
        for g in optimizer.param_groups:
            g['lr'] = new_lr

    # model_ema (revision). Persistent via config; CURE keeps it in train() mode.
    ema_model = getattr(config, "_ema_model", None)
    if ema_model is None:
        ema_model = copy.deepcopy(model)
        config._ema_model = ema_model
        config._global_step = 0
    ema_model.train()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    criterion_kl = nn.KLDivLoss(reduction='sum')   # == size_average=False in CURE
    w_nat = config.alpha          # CURE w_nat = alpha
    w_rob = 1.0 - config.alpha    # CURE w_rob = 1 - alpha
    beta = config.beta            # trades_beta
    aux_wt = config.gamma         # aux_loss_wt_kl1
    p_pct = config.percentile * 100.0   # RGP percentile (e.g. 30)
    ema_alpha = config.revision_decay
    ema_freq = config.revision_rate

    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        x, y = x.cuda(), y.cuda()
        N = x.shape[0]
        optimizer.zero_grad()

        # ---- Eq.4: KL-based PGD adversary ----
        x_adv = _cure_generate_adv(model, x, config.step_size, config.eps, config.steps)

        out_nat = model(x)
        out_adv = model(x_adv)

        # ---- Eq.5: L = CE(nat) + beta * KL(nat || adv) ----
        loss_nat_ce = F.cross_entropy(out_nat, y)
        loss_rob_ce = (1.0 / N) * criterion_kl(F.log_softmax(out_adv, dim=1),
                                               F.softmax(out_nat, dim=1))
        loss_main = loss_nat_ce + beta * loss_rob_ce

        # ---- Eq.8: consistency to model_ema (dml_Loss kl on nat + adv) ----
        with torch.no_grad():
            ema_nat = ema_model(x)
            ema_adv = ema_model(x_adv)
        loss_cons = aux_wt * (
            criterion_kl(F.log_softmax(out_nat, dim=1), F.softmax(ema_nat, dim=1)) / N
            + criterion_kl(F.log_softmax(out_adv, dim=1), F.softmax(ema_adv, dim=1)) / N)

        loss_total = loss_main + loss_cons   # collate_loss: loss_dml['loss'] + _loss_main

        # ---- RGP fuse_grads: base = grad(total loss); prominence = w_nat*g_nat + w_rob*g_rob ----
        optimizer.zero_grad()
        loss_nat_ce.backward(retain_graph=True)
        g_nat = {n: (p.grad.detach().clone() if p.grad is not None else None)
                 for n, p in model.named_parameters()}
        optimizer.zero_grad()
        loss_rob_ce.backward(retain_graph=True)
        g_rob = {n: (p.grad.detach().clone() if p.grad is not None else None)
                 for n, p in model.named_parameters()}
        optimizer.zero_grad()
        loss_total.backward()   # this is grads['loss'] (the applied base gradient)

        # mask only block conv1 params (CURE: l>0 and 'conv1' in name), signed grad_rgp percentile
        for n, p in model.named_parameters():
            if p.grad is None or '.conv1.' not in n:
                continue
            if g_nat[n] is None or g_rob[n] is None:
                continue
            grad_rgp = w_nat * g_nat[n] + w_rob * g_rob[n]
            thr = np.percentile(grad_rgp.detach().cpu().numpy(), p_pct)
            p.grad[grad_rgp < thr] = 0.0

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # ---- Eq.7 revision: EMA of parameters only, warmup alpha, prob ema_freq ----
        config._global_step += 1
        if torch.rand(1).item() < ema_freq:
            a = min(1.0 - 1.0 / (config._global_step + 1), ema_alpha)
            with torch.no_grad():
                for ep_, p_ in zip(ema_model.parameters(), model.parameters()):
                    ep_.mul_(a).add_(p_.detach(), alpha=(1.0 - a))

    # eval target (exp_avg) = model_ema (revision) state at end of epoch
    ema_sd = ema_model.state_dict()
    for k in list(exp_avg.keys()):
        exp_avg[k] = ema_sd[k].clone()


def train_temperature_reweight(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """train_temperature + per-sample KD-loss reweighting by TEACHER instability (input-dependent idea #2).

    Teacher instability at x = KL( p_t(x) || p_t(x_adv) ) on RAW teacher logits, where x_adv is the
    student's inner-loop adversary. Samples where the teacher's own prediction moves a lot are treated
    as unreliable targets and downweighted:
        w_i = exp(-gamma * kl_i / mean(kl)),  then  w <- w / mean(w)   (mean-1: keeps the loss scale)
    gamma=0 -> w == 1 == train_temperature EXACTLY (baseline). Sweep via --gamma.
    NOT a temperature reparametrization: acts on per-sample GRADIENT magnitude, not on target shape,
    so it cannot be absorbed by softmax the way the scalar-temperature signals were.
    """
    model.train()
    origin_model.eval()
    annealing = (epoch / config.epochs) ** 2
    decay = annealing * (1 - config.kappa) + config.kappa
    criterion_kl = nn.KLDivLoss(reduction='none')
    gamma = config.gamma if config.gamma is not None else 0.0

    kl_sum = 0.0; kl_max = 0.0; n_seen = 0
    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        N, C, H, W = x.shape
        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()

        with torch.no_grad():
            _, teacher_logits = origin_model(x, feat=True)      # raw teacher logits
            target = (teacher_logits / config.tau).detach()     # global temperature target

        x_pgd = inner_loss_only_return(model, target, x, y, optimizer, config.step_size, config.eps, perturb_steps=config.steps)

        with torch.no_grad():
            _, teacher_logits_adv = origin_model(x_pgd, feat=True)
            kl_t = criterion_kl(F.log_softmax(teacher_logits_adv, dim=1),
                                F.softmax(teacher_logits, dim=1)).sum(dim=1)        # [N] KL(t(x) || t(x_adv))
            # clamp the normalized signal: kl is heavy-tailed (max ~15x mean) and gamma<0 (UPWEIGHT)
            # would otherwise explode exp() so one sample dominates the batch
            kl_n = (kl_t / kl_t.mean().clamp_min(1e-8)).clamp(max=5.0)
            w = torch.exp(-gamma * kl_n)
            w = (w / w.mean().clamp_min(1e-8)).detach()                             # renormalize to mean 1

        plus_logits = model(x_pgd)
        kl_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(target, dim=1))
        loss = (w * kl_loss.sum(dim=1)).mean()

        student_logits = model(x)   # clean forward: updates student BN running stats (matches train_temperature)
        if config.lamda is not None and config.lamda > 0:
            consistency_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(student_logits, dim=1))
            loss += annealing * config.lamda * (consistency_loss).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()

        if config.weight_avg == True:
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 - decay) * value + decay * exp_avg[key]

        kl_sum += kl_t.sum().item(); kl_max = max(kl_max, kl_t.max().item()); n_seen += N

    logging.info({"teacher_kl_mean": round(kl_sum / max(n_seen, 1), 4), "teacher_kl_max": round(kl_max, 4), "epoch": epoch})


def train_temperature_decompw(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """Idea A: per-sample KD weight from the DECOMPOSITION of the teacher's vulnerability
    (not its magnitude — answers 'every sample shakes under a natural teacher').

    Teacher logit change under the student's adversary is split into two channels
    (raw teacher: z = linear(Phi), so magnitude-only change is exactly linear(s*Phi)):
        z_mag = linear( (||Phi_adv||/||Phi||) * Phi )     # pure magnitude/norm channel
        rot   = || z_adv - z_mag ||                        # rotation (direction) residual
        r     = rot / ||z_adv - z_clean||                  # rotation SHARE in (0,~1)
    norm-dominant vulnerability (r small) is what normalization fixes -> trust the target;
    rotation-dominant (r large) means the teacher's DIRECTIONAL info is corrupted near x ->
    downweight:  w = exp(-gamma * r / mean(r)), renormalized to mean 1. gamma=0 == baseline.
    """
    model.train()
    origin_model.eval()
    annealing = (epoch / config.epochs) ** 2
    decay = annealing * (1 - config.kappa) + config.kappa
    criterion_kl = nn.KLDivLoss(reduction='none')
    gamma = config.gamma if config.gamma is not None else 0.0

    r_all = []
    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        N, C, H, W = x.shape
        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()

        with torch.no_grad():
            f_c, teacher_logits = origin_model(x, feat=True)    # raw teacher feat & logits
            target = (teacher_logits / config.tau).detach()

        x_pgd = inner_loss_only_return(model, target, x, y, optimizer, config.step_size, config.eps, perturb_steps=config.steps)

        with torch.no_grad():
            f_a, z_a = origin_model(x_pgd, feat=True)
            s = (f_a.norm(dim=1) / f_c.norm(dim=1).clamp_min(1e-8)).unsqueeze(1)
            z_mag = origin_model.linear(s * f_c)                             # magnitude-only logits
            rot = (z_a - z_mag).norm(dim=1)
            tot = (z_a - teacher_logits).norm(dim=1).clamp_min(1e-8)
            r = rot / tot                                                    # rotation share [N]
            w = torch.exp(-gamma * r / r.mean().clamp_min(1e-8))
            w = (w / w.mean().clamp_min(1e-8)).detach()

        plus_logits = model(x_pgd)
        kl_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(target, dim=1))
        loss = (w * kl_loss.sum(dim=1)).mean()

        student_logits = model(x)   # clean forward: updates student BN running stats (matches train_temperature)
        if config.lamda is not None and config.lamda > 0:
            consistency_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(student_logits, dim=1))
            loss += annealing * config.lamda * (consistency_loss).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()

        if config.weight_avg == True:
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 - decay) * value + decay * exp_avg[key]

        r_all.append(r.cpu())

    r_cat = torch.cat(r_all)
    q = torch.quantile(r_cat, torch.tensor([0.05, 0.5, 0.95]))
    logging.info({"rot_share_mean": round(r_cat.mean().item(), 4), "rot_share_p5": round(q[0].item(), 4),
                  "rot_share_p50": round(q[1].item(), 4), "rot_share_p95": round(q[2].item(), 4), "epoch": epoch})


def train_temperature_padapt(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """train_temperature with the ResNet18_zp student (p_adapt: True) — input-dependent idea #3.

    The student's feature-normalization STRENGTH is a learned per-sample head:
        Phi_hat = Phi / ||Phi||^{p(x)},  p(x) = sigmoid(p_head(Phi)), zero-init -> starts at 0.5.
    Training loop identical to train_temperature; additionally logs the clean-batch p(x)
    distribution every epoch. p(x)->1 everywhere == full L2 norm (iso3); p(x)->0 == raw student;
    meaningful per-sample variation == the input-dependent mechanism we are hunting.
    """
    model.train()
    origin_model.eval()
    annealing = (epoch / config.epochs) ** 2
    decay = annealing * (1 - config.kappa) + config.kappa
    criterion_kl = nn.KLDivLoss(reduction='none')

    p_all = []
    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        N, C, H, W = x.shape
        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()

        with torch.no_grad():
            _, teacher_logits = origin_model(x, feat=True)      # raw teacher logits
            target = (teacher_logits / config.tau).detach()     # global temperature target

        x_pgd = inner_loss_only_return(model, target, x, y, optimizer, config.step_size, config.eps, perturb_steps=config.steps)
        plus_logits = model(x_pgd)
        kl_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(target, dim=1))
        loss = (1.0 / N) * (kl_loss.sum(dim=1)).sum()

        student_logits = model(x)   # clean forward: updates student BN running stats (matches train_temperature)
        if config.lamda is not None and config.lamda > 0:
            consistency_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(student_logits, dim=1))
            loss += annealing * config.lamda * (consistency_loss).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()

        if config.weight_avg == True:
            for key, value in model.state_dict().items():
                exp_avg[key] = (1 - decay) * value + decay * exp_avg[key]

        # _p_last was set by the LAST forward = the clean model(x) pass above
        for m in model.modules():
            if getattr(m, "_p_last", None) is not None:
                p_all.append(m._p_last.cpu())
                break

    if p_all:
        p_cat = torch.cat(p_all).flatten()
        q = torch.quantile(p_cat, torch.tensor([0.05, 0.5, 0.95]))
        logging.info({"p_mean": round(p_cat.mean().item(), 4), "p_std": round(p_cat.std().item(), 4),
                      "p_p5": round(q[0].item(), 4), "p_p50": round(q[1].item(), 4),
                      "p_p95": round(q[2].item(), 4), "epoch": epoch})


# =====================================================================================================
# ROBUST-DISTILLATION BASELINES (2026-08-30) -- ARD / RSLAD / AdaAD / AdaAD+IGDM.
#
# Ported from the official IGDM release (../IGDM: `ard_cifar100.py`, `rslad_cifar100.py`,
# `adaad_cifar100.py`, `adaad_IGDM_cifar100.py`, `rslad_loss.py`) into THIS framework, so that they
# share our data pipeline, our schedule, our Converter and our evaluation (PGD-20 / CW / AutoAttack
# out of `utils.evaluate`) instead of theirs.  Without that they cannot be put in the same table:
# their scripts run a different PGD-20 implementation each epoch and AutoAttack exactly once, at the
# end of a 200-epoch run.
#
# The reason to run them here at all is the TEACHER.  All four were published with a large ROBUST
# teacher -- their `cifar100.sh` default is `Wang2023Better_WRN-28-10` off RobustBench.  This port
# feeds them OUR clean ResNet-18 instead, the same natural teacher the feature anchor uses, so the
# table answers "what do the published robust-distillation objectives do when no robust teacher
# exists", which is the regime this paper is about.  Run against their own robust teacher they are a
# different family of method and not a baseline for us.
#
# A PREDICTION THAT FAILED, recorded because it was made before the runs (2026-08-31).  We expected
# the ORDER to follow where each method evaluates the teacher: ARD and RSLAD query it at x only,
# AdaAD at x_adv, AdaAD+IGDM additionally at x - delta, and off-manifold queries to a naturally
# trained teacher should carry nothing usable -- so AdaAD should have degraded the most.  Measured,
# the order is the opposite: ARD 57.61/20.24, RSLAD 59.68/21.30, AdaAD 59.79/23.19 (clean/AA).
# AdaAD is the BEST of the three.  What the ordering tracks instead is whether the inner and outer
# problems are matched: ARD attacks with label CE and trains a KL, RSLAD matches both to a fixed
# teacher target, AdaAD matches both to a target that moves with the attack -- which is the same
# structural property section 3 credits for the anchor, not a refutation of it.  The claim that
# survives is the one that does not depend on ordering: given a natural teacher, ALL of these
# objectives land below both our anchor (61.21/25.24) and plain PGD-AT initialized at the same
# teacher (57.73/26.46).  Do not restate the off-manifold prediction anywhere.
#
# CHANGED FROM THE ORIGINAL SCRIPTS (everything not listed is line-for-line):
#   * attack step size / random start.  Their ARD attack calls `PGD(...)`, whose default step is
#     `alpha=2/225` -- a typo for 2/255 -- and which starts from uniform(-eps,eps).  Here all four
#     use `config.step_size` and the 0.001*randn start that our own cells and their OWN rslad/adaad
#     inner losses use, so every row in the table attacks identically.  Internal consistency beats
#     reproducing a typo.
#   * IGDM warm-up.  Theirs is `alpha * (epoch/200)` with 200 hardcoded against a 1-indexed loop.
#     Here it is `alpha * (epoch+1)/igdm_warm_epochs`, defaulting to config.epochs, so the ramp keeps
#     its relative position when the schedule length changes.
#   * log_softmax.  RSLAD's outer term writes `torch.log(F.softmax(z))`, which underflows to -inf for
#     a confident 100-way logit; we use the mathematically identical `F.log_softmax`.
#   * stack knobs.  `train_eps` / `wa_start` / `freeze_lr_epoch` / AWP are wired in as in
#     `train_madry_at` and are inert unless set.  They are OFF for the headline baseline cells.
# Loss weights, reductions, attack objectives, the 5:1 RSLAD mix, the hardcoded alpha=1 & temp=1 in
# ARD and beta=1 in AdaAD are all reproduced as published, including the reduction quirk documented
# on `train_rslad`.
# =====================================================================================================


class _DistillStack:
    """The `train_madry_at` stack knobs (train_eps / WA / freeze_lr / AWP) factored out, so each
    baseline's body can stay as close to the original script as possible.  Everything here is inert
    unless the corresponding config key is set."""

    def __init__(self, owner, model, epoch, config):
        self.config, self.model, self.epoch = config, model, epoch
        self.eps = float(getattr(config, "train_eps", 0.0) or 0.0) or config.eps
        self.freeze_lr_epoch = _resolve_epoch_arg(getattr(config, "freeze_lr_epoch", None), config.epochs)
        self.wa_start = _resolve_epoch_arg(getattr(config, "wa_start", None), config.epochs) or 0
        gamma = float(getattr(config, "awp_gamma", 0.0) or 0.0)
        self.use_awp = gamma > 0 and epoch >= int(getattr(config, "awp_warmup", 0) or 0)
        self._diff = None
        if self.use_awp:
            if not hasattr(owner, "_awp"):
                owner._awp = AdvWeightPerturb(model, gamma=gamma)
            self.awp = owner._awp
            self.awp.gamma = gamma

    def perturb(self, loss_fn):
        if self.use_awp:
            self._diff = self.awp.calc_awp(loss_fn)
            self.awp.perturb(self._diff)

    def restore(self):
        if self.use_awp:
            self.awp.restore(self._diff)

    def after_step(self, scheduler, exp_avg):
        if self.freeze_lr_epoch is None or self.epoch < self.freeze_lr_epoch:
            scheduler.step()
        if getattr(self.config, "weight_avg", False) == True and self.epoch >= self.wa_start:
            annealing = (self.epoch / self.config.epochs) ** 2
            decay = annealing * (1 - self.config.kappa) + self.config.kappa
            for key, value in self.model.state_dict().items():
                exp_avg[key] = (1 - decay) * value + decay * exp_avg[key]


def _kd_inner_attack(model, teacher_logits, x_natural, step_size, epsilon, perturb_steps):
    """RSLAD-style inner max: maximize KL(student(x') || teacher(x)).  The target is a FIXED tensor
    computed once at x, so the teacher is never queried off the data manifold (rslad_loss.py:27-60)."""
    criterion_kl = nn.KLDivLoss(reduction='none')
    was_training = model.training
    model.eval()
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = torch.sum(criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                             F.softmax(teacher_logits, dim=1)))
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    if was_training:
        model.train()
    return x_adv.detach()


def _adaad_inner_attack(model, teacher, x_natural, step_size, epsilon, perturb_steps):
    """AdaAD inner max: maximize KL(student(x') || teacher(x')) (rslad_loss.py:266-306).  The teacher
    is RE-EVALUATED at the perturbed point every step -- that is what makes the target adaptive, and
    it is also what puts a non-robust teacher's off-manifold logits inside the objective.  The
    teacher branch is deliberately NOT detached, exactly as published: the attack gradient flows
    through both networks."""
    criterion_kl = nn.KLDivLoss(reduction='none')
    was_training = model.training
    model.eval()
    teacher.eval()
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = torch.sum(criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                             F.softmax(teacher(x_adv), dim=1)))
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    if was_training:
        model.train()
    return x_adv.detach()


def train_ard(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """ARD, Goldblum et al. AAAI 2020 -- as implemented in ../IGDM/ard_cifar100.py:130-143.

    x_adv is a plain TRUE-LABEL CE-PGD on the student; the loss then pulls the student's ADVERSARIAL
    logits onto the teacher's CLEAN logits.  Their script hardcodes alpha = 1 and temp = 1, which
    kills the CE term and leaves plain KD at tau = 1; `ard_alpha` / `ard_temp` are exposed but
    default to exactly those values, so an untouched config reproduces their code.

    With our natural teacher this is the closest published method to the logit-anchor row of our own
    base table (tau = 1: clean 58.26 / AA 20.84), differing only in that the attack is CE on labels
    rather than KL to the teacher."""
    model.train()
    origin_model.eval()
    st = _DistillStack(train_ard, model, epoch, config)
    alpha = float(getattr(config, "ard_alpha", 1.0))
    temp = float(getattr(config, "ard_temp", 1.0))
    criterion_kl = nn.KLDivLoss(reduction="batchmean")
    for x, y in tqdm(train_loader):
        x, y = x.cuda(), y.cuda()
        x_adv = _pgd_attack_true_label(model, x, y, config.step_size, st.eps, config.steps)
        with torch.no_grad():
            teacher_logits = origin_model(x).detach()

        def _loss(m, _x=x, _xa=x_adv, _y=y, _t=teacher_logits):
            l = alpha * temp * temp * criterion_kl(F.log_softmax(m(_xa) / temp, dim=1),
                                                   F.softmax(_t / temp, dim=1))
            if alpha < 1.0:
                l = l + (1.0 - alpha) * F.cross_entropy(m(_x), _y)
            return l

        st.perturb(_loss)
        optimizer.zero_grad()
        _loss(model).backward()
        optimizer.step()
        st.restore()
        st.after_step(scheduler, exp_avg)


def train_rslad(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """RSLAD, Zi et al. ICCV 2021 -- as implemented in ../IGDM/rslad_cifar100.py:126-132.

    Labels are absent from the entire pipeline: the inner max is KL to the teacher's SOFT labels at
    x, and the outer loss mixes adversarial and clean KD 5:1 against that same fixed target.

    REDUCTION QUIRK, reproduced as published.  Their term is `10 * mean(elementwise_KL)`, i.e. 10/C
    times the batch-mean KL.  The comment in their file says the 10 is there "to keep consistent with
    CIFAR-10", where C = 10 makes it exactly batchmean -- but on CIFAR-100 the same line scales the
    ONLY loss term by 0.1, an effective 10x learning-rate cut.  `rslad_scale` (default 10.0)
    reproduces it; setting it to the class count undoes it.  Worth an ablation before concluding
    anything from a weak RSLAD number, since on this dataset the published constant is arguably a
    bug rather than a design choice."""
    model.train()
    origin_model.eval()
    st = _DistillStack(train_rslad, model, epoch, config)
    scale = float(getattr(config, "rslad_scale", 10.0))
    w_adv = float(getattr(config, "rslad_w_adv", 5.0 / 6.0))
    for x, y in tqdm(train_loader):
        x = x.cuda()
        with torch.no_grad():
            teacher_logits = origin_model(x).detach()
        p_t = F.softmax(teacher_logits, dim=1)
        x_adv = _kd_inner_attack(model, teacher_logits, x, config.step_size, st.eps, config.steps)

        def _loss(m, _x=x, _xa=x_adv, _pt=p_t):
            # their kl_loss(a, b) = -a*b + log(b+1e-5)*b, pointwise, with a = log p_student
            ent = torch.log(_pt + 1e-5) * _pt
            l_adv = scale * torch.mean(-F.log_softmax(m(_xa), dim=1) * _pt + ent)
            l_nat = scale * torch.mean(-F.log_softmax(m(_x), dim=1) * _pt + ent)
            return w_adv * l_adv + (1.0 - w_adv) * l_nat

        st.perturb(_loss)
        optimizer.zero_grad()
        _loss(model).backward()
        optimizer.step()
        st.restore()
        st.after_step(scheduler, exp_avg)


def train_adaad(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """AdaAD, Huang et al. CVPR 2023 -- as implemented in ../IGDM/adaad_cifar100.py:135-142.

    The teacher is evaluated at x_adv on BOTH sides: the inner max looks for the point where student
    and teacher disagree most, and the outer loss closes that gap there.  Their `beta` is hardcoded
    to 1, so the clean term is computed and then multiplied by zero; `adaad_beta` is exposed and
    defaults to 1 to reproduce that.

    This is the cell Proposition 2 says should suffer most under a natural teacher: t(x_adv) is a
    query the teacher was never trained to answer, and it appears in the objective, not just as a
    fixed anchor."""
    model.train()
    origin_model.eval()
    st = _DistillStack(train_adaad, model, epoch, config)
    beta = float(getattr(config, "adaad_beta", 1.0))
    for x, y in tqdm(train_loader):
        x = x.cuda()
        x_adv = _adaad_inner_attack(model, origin_model, x, config.step_size, st.eps, config.steps)
        with torch.no_grad():
            t_adv = origin_model(x_adv).detach()
            t_nat = origin_model(x).detach() if beta < 1.0 else None

        def _loss(m, _x=x, _xa=x_adv, _ta=t_adv, _tn=t_nat):
            l = beta * F.kl_div(F.log_softmax(m(_xa), dim=1), F.softmax(_ta, dim=1),
                                reduction='batchmean')
            if beta < 1.0:
                l = l + (1.0 - beta) * F.kl_div(F.log_softmax(m(_x), dim=1), F.softmax(_tn, dim=1),
                                                reduction='batchmean')
            return l

        st.perturb(_loss)
        optimizer.zero_grad()
        _loss(model).backward()
        optimizer.step()
        st.restore()
        st.after_step(scheduler, exp_avg)


def train_adaad_igdm(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """AdaAD + IGDM, Lee & Kim -- as implemented in ../IGDM/adaad_IGDM_cifar100.py:118-147.

    AdaAD plus one extra term.  With delta = x_adv - x and their default b = g = 1,

        s+ = s(x + b*delta) = s(x_adv),   s- = s(x - g*delta)
        t+ = t(x + b*delta),              t- = t(x - g*delta)
        L_igdm = KL( softmax(s+ - s-) || softmax(t+ - t-) )

    i.e. a CENTRAL DIFFERENCE along delta, which is a finite-difference stand-in for the input
    gradient -- matching gradients directly needs double backprop, matching two-point differences
    does not.  Hence "indirect".  Its weight is ramped linearly over the run (`alpha * (epoch+1)/E`),
    which is their `alpha * epoch/200` made schedule-length-portable.

    Note this is the only cell that queries the teacher at x - delta, a point on the far side of the
    clean image from the attack.  For a robust teacher that is still a sensible query.  For a natural
    one it is a second off-manifold evaluation on top of AdaAD's, so under Proposition 2 the IGDM
    term should not rescue AdaAD here even though it does with their own teacher."""
    model.train()
    origin_model.eval()
    st = _DistillStack(train_adaad_igdm, model, epoch, config)
    alpha = float(getattr(config, "igdm_alpha", 20.0))
    beta = float(getattr(config, "igdm_beta", 1.0))
    gamma = float(getattr(config, "igdm_gamma", 1.0))
    warm = float(_resolve_epoch_arg(getattr(config, "igdm_warm_epochs", None), config.epochs) or config.epochs)
    w_igdm = alpha * min(1.0, (epoch + 1) / warm)
    criterion_kl = nn.KLDivLoss(reduction="batchmean")
    for x, y in tqdm(train_loader):
        x = x.cuda()
        x_adv = _adaad_inner_attack(model, origin_model, x, config.step_size, st.eps, config.steps)
        delta = (x_adv - x).detach()
        x_plus, x_minus = (x + beta * delta).detach(), (x - gamma * delta).detach()
        with torch.no_grad():
            t_adv = origin_model(x_adv).detach()
            t_plus = origin_model(x_plus).detach()
            t_minus = origin_model(x_minus).detach()
        t_diff = F.softmax(t_plus - t_minus, dim=1)

        def _loss(m, _xa=x_adv, _xp=x_plus, _xm=x_minus, _ta=t_adv, _td=t_diff):
            l = F.kl_div(F.log_softmax(m(_xa), dim=1), F.softmax(_ta, dim=1), reduction='batchmean')
            s_diff = m(_xp) - m(_xm)
            return l + w_igdm * criterion_kl(F.log_softmax(s_diff, dim=1), _td)

        st.perturb(_loss)
        optimizer.zero_grad()
        _loss(model).backward()
        optimizer.step()
        st.restore()
        st.after_step(scheduler, exp_avg)


# =====================================================================================================
# TRADES AND MART (2026-09-02) -- the two standard adversarial-training baselines the main tables were
# leaving empty.  Every other row of those tables is measured in this framework; quoting these two from
# the literature would have reintroduced exactly the cross-codebase mixing the tables exist to avoid.
#
# Both are implemented as published and share the protocol of the other baseline cells (SGD 0.1,
# momentum 0.9, step decay x0.1 at epochs 70 and 90, 100 epochs, eps 8/255, 10-step attack), so the
# whole baseline block is internally comparable.  Neither uses a teacher: `origin_model` is ignored.
# The `train_madry_at` stack knobs are wired in and inert unless set.
# =====================================================================================================


def _trades_inner_attack(model, x_natural, step_size, epsilon, perturb_steps):
    """TRADES inner maximization: maximize KL(f(x') || f(x)) with the CLEAN prediction as the fixed
    target for the duration of the attack (Zhang et al., ICML 2019, Eq. 6).  Unlike a label attack it
    never sees y, which is the whole point -- the perturbation is chosen to move the prediction, not to
    cross a labelled boundary."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        p_nat = F.softmax(model(x_natural), dim=1)
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = F.kl_div(F.log_softmax(model(x_adv), dim=1), p_nat, reduction='sum')
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    if was_training:
        model.train()
    return x_adv.detach()


def train_trades(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """TRADES (Zhang et al., ICML 2019).

        L = CE(f(x), y) + beta * KL(f(x_adv) || f(x)) / N

    with x_adv from the KL attack above.  The clean term keeps accuracy and the KL term buys
    smoothness; `beta` (config.beta, 6.0 as published) is the trade-off knob, and it is the thing our
    own method does not have.  origin_model is unused."""
    model.train()
    st = _DistillStack(train_trades, model, epoch, config)
    beta = float(config.beta if config.beta is not None else 6.0)
    for x, y in tqdm(train_loader):
        x, y = x.cuda(), y.cuda()
        x_adv = _trades_inner_attack(model, x, config.step_size, st.eps, config.steps)

        def _loss(m, _x=x, _xa=x_adv, _y=y):
            logits_nat = m(_x)
            return (F.cross_entropy(logits_nat, _y)
                    + beta * F.kl_div(F.log_softmax(m(_xa), dim=1),
                                      F.softmax(logits_nat, dim=1), reduction='batchmean'))

        st.perturb(_loss)
        optimizer.zero_grad()
        _loss(model).backward()
        optimizer.step()
        st.restore()
        st.after_step(scheduler, exp_avg)


def train_mart(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """MART (Wang et al., ICLR 2020).

        L = BCE(f(x_adv), y) + lambda * E_i [ KL(f(x_adv) || f(x))_i * (1 - p_y(x)_i) ]
        BCE(f(x'), y) = -log p_y(x') - log(1 - max_{k != y} p_k(x'))

    Two differences from TRADES.  The adversarial term is a boosted cross-entropy that also pushes down
    the strongest wrong class, and the KL term is re-weighted per sample by the model's clean error
    probability, so examples the model already gets wrong on clean data are regularized harder.  The
    attack is a plain true-label CE-PGD.  `lambda` is config.beta (5.0 as published).  origin_model is
    unused."""
    model.train()
    st = _DistillStack(train_mart, model, epoch, config)
    lam = float(config.beta if config.beta is not None else 5.0)
    eps_small = 1e-12
    for x, y in tqdm(train_loader):
        x, y = x.cuda(), y.cuda()
        x_adv = _pgd_attack_true_label(model, x, y, config.step_size, st.eps, config.steps)

        def _loss(m, _x=x, _xa=x_adv, _y=y):
            logits_adv, logits_nat = m(_xa), m(_x)
            p_adv = F.softmax(logits_adv, dim=1)
            p_nat = F.softmax(logits_nat, dim=1)
            # strongest wrong class under the attack
            tmp = torch.argsort(p_adv, dim=1, descending=True)[:, :2]
            other = torch.where(tmp[:, 0] == _y, tmp[:, 1], tmp[:, 0])
            bce = (F.cross_entropy(logits_adv, _y)
                   + F.nll_loss(torch.log(1.0001 - p_adv + eps_small), other))
            kl = (F.kl_div(F.log_softmax(logits_adv, dim=1), p_nat, reduction='none').sum(dim=1)
                  * (1.0 - p_nat.gather(1, _y.unsqueeze(1)).squeeze(1)))
            return bce + lam * kl.mean()

        st.perturb(_loss)
        optimizer.zero_grad()
        _loss(model).backward()
        optimizer.step()
        st.restore()
        st.after_step(scheduler, exp_avg)
