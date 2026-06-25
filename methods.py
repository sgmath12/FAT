from asyncio.unix_events import BaseChildWatcher
from cmath import tanh
import torch.nn as nn
from torch.distributions import Beta
from tqdm import tqdm
from utils import *
import pdb
import torch.nn.functional as F

import numpy as np
from torch.autograd import Variable

import torch
import torch.nn.functional as F
import torchattacks
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


def train_DPFAT_rank(model, train_loader, optimizer, origin_model, epoch, config, scheduler, exp_avg):
    """DPFAT_adaptive with a configurable per-sample temperature SIGNAL.

    Same pipeline as train_DPFAT_adaptive; only the per-sample temperature
    T_i = config.tau * signal_i + config.alpha changes:
      config.signal : 'norm'   -> ||teacher_feat||      (original DPFAT)
                      'margin'  -> top1-top2 teacher logit  (high dispersion, robust-predictive)
      config.rank   : True      -> replace signal by its batch rank/N in (0,1]
                                   (forces full per-sample spread; fixes low-dispersion failure)
    4-cell ablation: {norm,margin} x {raw,rank}.
    """
    model.train()
    origin_model.eval()
    annealing = (epoch / config.epochs) ** 2
    decay = annealing * (1 - config.kappa) + config.kappa
    criterion_kl = nn.KLDivLoss(reduction='none')
    use_rank = getattr(config, "rank", True)
    signal_kind = getattr(config, "signal", "margin")

    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        N, C, H, W = x.shape
        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()
        teacher_feat, teacher_logits = origin_model(x, feat=True)

        # ---- per-sample temperature signal (from clean teacher) ----
        if signal_kind == "norm":
            s = teacher_feat.norm(dim=1)
        else:  # 'margin' : top1 - top2 logit
            top2 = teacher_logits.topk(2, dim=1).values
            s = top2[:, 0] - top2[:, 1]
        if use_rank:  # batch rank -> (0,1], forces full dispersion
            s = s.argsort().argsort().float().add(1).div(N)
        T = (config.tau * s + config.alpha).reshape([-1, 1])
        teacher_logits = teacher_logits / T

        x_pgd = inner_loss_only_return(model, teacher_logits, x, y, optimizer, config.step_size, config.eps, perturb_steps=config.steps)
        plus_logits = model(x_pgd)

        kl_loss = criterion_kl(F.log_softmax(plus_logits, dim=1), F.softmax(teacher_logits.detach(), dim=1))
        loss = (1.0 / N) * (kl_loss.sum(dim=1)).sum()

        student_logits = model(x)
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
