import torch
import torch.nn as nn
import torch.nn.functional as F

from CIFAR10.models.resnet_z import BasicBlock


class ResNetGain(nn.Module):
    """ResNet18_z with a GAIN-ONLY head (user's Gram-Schmidt idea as PARAMETRIZATION, 2026-07-13).

    Decompose the student head against the teacher direction it is finetune-initialized from:
        w_s,c = g_c * w_hat_t,c + r_c   (r_c orthogonal to w_t,c)
    and DELETE r_c structurally: here  w_s,c = exp(log_g_c) * w_t,c  with linear.weight FROZEN
    (requires_grad False; the teacher row w_t,c arrives via the existing finetune load), so the
    only learnable head DOF are the 100 per-class log-gains (+ the tiny free bias, kept to match
    the baseline in everything except direction). exp(0)=1 -> exact teacher head at init = the
    same fair start as ResNet18_z under finetune.

    This is the {direction FROZEN, ||w_c|| free} cell of the head-side 2x2; its diagonal partner
    is coshead {direction free, ||w_c|| frozen} which LOSES -1.4~-2.7. Diagnostic basis
    (scripts/diag_head_rotation.py, 2026-07-13): the baseline head's rotation away from w_t is
    almost entirely ONE shared 512-d rotation (Procrustes cos 0.97 vs random null 0.88; per-class
    residual only ~25% of norm) which a free backbone can absorb, while the difficulty-aligned
    per-class structure lives in the gains (spearman(log g_c, gnorm) = +0.79 after alignment).
    Registered prediction: TIE or small loss. The soft penalty version (L_orth on the residual)
    is held only as an optional dose-response figure -- every added regularizer in this project
    has died, the parametrization is the clean isolation.

    Keeps nn.Linear so the clean checkpoint loads via the existing strict=False fallback
    (missing key: log_g only). The observed 5x ||w_c|| growth is reachable (log_g ~ 1.6).
    """
    def __init__(self, block, num_blocks, num_classes=10, scale=1.0):
        super(ResNetGain, self).__init__()
        self.scale = scale
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.linear.weight.requires_grad_(False)   # direction (and init norm) frozen at the teacher head
        self.log_g = nn.Parameter(torch.zeros(num_classes))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def head_from_feat(self, out):
        w = self.linear.weight * self.log_g.exp().unsqueeze(1)
        return F.linear(out, w, self.linear.bias)

    def forward(self, x, feat=None):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        original_out = out.view(out.size(0), -1)
        out = self.scale * original_out/original_out.norm(dim=1).reshape([-1, 1])

        if feat:
            return original_out, self.head_from_feat(out)
        return self.head_from_feat(out)

    def extract_feature(self, x, only_feature=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        feat = F.avg_pool2d(out, 4)
        feat = feat.view(feat.size(0), -1)
        feat = self.scale * feat/feat.norm(dim=1).reshape([-1, 1])
        out = self.head_from_feat(feat)
        if only_feature:
            return feat, None
        return feat, out


def ResNet18_zgain(num_classes=10, scale=1.0):
    return ResNetGain(BasicBlock, [2, 2, 2, 2], num_classes, scale=scale)
