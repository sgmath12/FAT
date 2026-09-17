import torch
import torch.nn.functional as F


def _kl_div(logit1, logit2):
    # Target floored for the same reason as in _jensen_shannon_div below.
    return F.kl_div(F.log_softmax(logit1, dim=1), F.softmax(logit2, dim=1).clamp(min=1e-8),
                    reduction='batchmean')


def _jensen_shannon_div(logit1, logit2, T=1.):
    prob1 = F.softmax(logit1/T, dim=1)
    prob2 = F.softmax(logit2/T, dim=1)
    mean_prob = 0.5 * (prob1 + prob2)

    logsoftmax = torch.log(mean_prob.clamp(min=1e-8))
    # 2026-09-17: the KL TARGETS have to be floored too.  With torch >= 1.13, F.kl_div's backward is
    # NaN wherever the target is exactly 0 (pytorch/pytorch#89558), and at T = 0.5 the softmax is
    # sharpened enough that exact zeros are routine: this run went NaN at epoch 1.
    jsd = F.kl_div(logsoftmax, prob1.clamp(min=1e-8), reduction='batchmean')
    jsd += F.kl_div(logsoftmax, prob2.clamp(min=1e-8), reduction='batchmean')
    return jsd * 0.5


def _jensen_shannon_div_without_reduction(logit1, logit2, T=1.):
    prob1 = F.softmax(logit1/T, dim=1)
    prob2 = F.softmax(logit2/T, dim=1)
    mean_prob = 0.5 * (prob1 + prob2)

    logsoftmax = torch.log(mean_prob.clamp(min=1e-8))
    jsd = F.kl_div(logsoftmax, prob1.clamp(min=1e-8), reduction='none')
    jsd += F.kl_div(logsoftmax, prob2.clamp(min=1e-8), reduction='none')
    return jsd * 0.5
