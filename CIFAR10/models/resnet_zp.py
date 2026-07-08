import torch
import torch.nn as nn
import torch.nn.functional as F

from CIFAR10.models.resnet_z import BasicBlock, ResNet


class ResNetP(ResNet):
    """ResNet18_z variant with a LEARNED per-sample normalization strength p(x):

        out = scale * Phi / ||Phi||^{p(x)},   p(x) = sigmoid(p_head(Phi)) in (0,1)

    p==1 recovers full L2 normalization (ResNet18_z); p==0 the raw student.
    p_head is zero-initialized -> starts at p=0.5 for every input, free to move either way;
    if training pushes p(x)->1 everywhere, that is learned evidence that full normalization
    is per-sample optimal. Stores last-batch p in self._p_last for epoch-level logging.
    """

    def __init__(self, block, num_blocks, num_classes=10, scale=1.0):
        super(ResNetP, self).__init__(block, num_blocks, num_classes, scale=scale)
        self.p_head = nn.Linear(512 * block.expansion, 1)
        nn.init.zeros_(self.p_head.weight)
        nn.init.zeros_(self.p_head.bias)
        self._p_last = None

    def forward(self, x, feat=None):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        original_out = out.view(out.size(0), -1)
        p = torch.sigmoid(self.p_head(original_out))                 # [N,1]
        norm = original_out.norm(dim=1).reshape([-1, 1])
        out = self.scale * original_out / norm.pow(p)
        self._p_last = p.detach()

        if feat:
            return original_out, self.linear(out)
        return self.linear(out)


def ResNet18_zp(num_classes=10, scale=1.0):
    return ResNetP(BasicBlock, [2, 2, 2, 2], num_classes, scale=scale)


class ResNetPGlobal(ResNet):
    """Ablation between fixed p=1 and per-sample p(x): a single LEARNED GLOBAL scalar p.

        out = scale * Phi / ||Phi||^{p},   p = sigmoid(p_logit), one parameter shared by all inputs

    zero-init -> starts at p=0.5. Same _p_last logging interface as ResNetP (constant across batch)."""

    def __init__(self, block, num_blocks, num_classes=10, scale=1.0):
        super(ResNetPGlobal, self).__init__(block, num_blocks, num_classes, scale=scale)
        self.p_logit = nn.Parameter(torch.zeros(1))
        self._p_last = None

    def forward(self, x, feat=None):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        original_out = out.view(out.size(0), -1)
        p = torch.sigmoid(self.p_logit)                              # scalar, shared
        norm = original_out.norm(dim=1).reshape([-1, 1])
        out = self.scale * original_out / norm.pow(p)
        self._p_last = p.detach().expand(original_out.size(0), 1)

        if feat:
            return original_out, self.linear(out)
        return self.linear(out)


def ResNet18_zpg(num_classes=10, scale=1.0):
    return ResNetPGlobal(BasicBlock, [2, 2, 2, 2], num_classes, scale=scale)
