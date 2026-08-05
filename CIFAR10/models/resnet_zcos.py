import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from CIFAR10.models.resnet_z import BasicBlock


class ResNetCos(nn.Module):
    """ResNet18_z with a FULLY directional (cosine) head (user's idea, 2026-07-07).

    ResNet18_z normalizes only the feature: logits = W (scale*Phi_hat) + b, so a per-class magnitude
    channel survives in ||w_c|| -- and it is USED: the clean ckpt has ||w_c|| ~ 1.85 (1.25x class
    spread) but an AT-trained student grows it 5x to ~9.5 (1.37x spread). This head closes it:

        logits = s * cos(theta_c) = exp(log_s) * (Phi_hat . w_c_hat),   no bias.

    s is a single LEARNABLE global scalar (log-parametrized): the observed 5x ||W|| growth is the
    net's global softmax-sharpening knob, and freezing it would confound "remove per-class magnitude"
    with "forbid global sharpening" (a global temperature is allowed by the project's own rules --
    only per-class/per-sample magnitude freedom is the target). log_s init = log(1.853) = the clean
    ckpt's mean ||w_c||, so at finetune-init this head's logits ~= ResNet18_z's (bias std 0.02,
    class-norm spread 1.25x -> collapses to 1) = fair start. Keeps nn.Linear so the clean checkpoint
    loads via the existing strict=False fallback (missing key: log_s only; linear.bias loads but is
    unused).
    """
    def __init__(self, block, num_blocks, num_classes=10, scale=1.0, s_init=1.853):
        super(ResNetCos, self).__init__()
        self.scale = scale
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.log_s = nn.Parameter(torch.tensor(float(np.log(s_init))))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _head(self, feat_hat):
        w_hat = F.normalize(self.linear.weight, dim=1)
        return torch.exp(self.log_s) * self.scale * F.linear(feat_hat, w_hat)

    def head_from_feat(self, out):
        """Classifier applied to an already normalized(+scaled) feature, for methods that build the
        head term separately from forward() (train_feat_direction). The cosine head is defined on the
        unit direction, so the caller's scaling is renormalized away here and the logit scale comes
        from the module's own scale * exp(log_s) -- making this bit-identical to forward()'s head.
        Added 2026-08-01: without it the cosine-head cell cannot run under feat_direction at all,
        which is why every previous coshead measurement came from the baseline-KL/madry_at methods."""
        return self._head(F.normalize(out, dim=1))

    def forward(self, x, feat=None):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        original_out = out.view(out.size(0), -1)
        out = original_out / original_out.norm(dim=1).reshape([-1, 1])

        if feat:
            return original_out, self._head(out)
        return self._head(out)

    def extract_feature(self, x, only_feature=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        feat = F.avg_pool2d(out, 4)
        feat = feat.view(feat.size(0), -1)
        feat = feat / feat.norm(dim=1).reshape([-1, 1])
        out = self._head(feat)
        if only_feature:
            return feat, None
        return feat, out


def ResNet18_zcos(num_classes=10, scale=1.0, s_init=1.853):
    return ResNetCos(BasicBlock, [2, 2, 2, 2], num_classes, scale=scale, s_init=s_init)
