"""ResNet18_zbn: ResNet18_z (cosine head) + per-stage RMS normalization at the 4 stage outputs.
Separate file so the shared resnet_z.py is untouched. Adds NO parameters (state_dict keys identical to
ResNet_z), so the clean checkpoint loads exactly as before. block_norm=True applies, per sample, at each of
layer1..layer4 output:  x <- x / RMS(x)   (RMS = ||x||/sqrt(numel); L2 projection to the sphere rescaled so
element magnitude stays O(1) -> no vanishing). Robustness rationale: bound per-stage perturbation growth."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = out + (self.shortcut(x) if self.shortcut else x)
        out = F.relu(out)
        return out


class ResNet_zbn(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, scale=1.0, block_norm=True):
        super(ResNet_zbn, self).__init__()
        self.scale = scale
        self.block_norm = block_norm
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.nChannels = [64, 128, 256, 512]
        self.nWidths = [32, 16, 8, 4]
        self.nHeights = [32, 16, 8, 4]

        for channel in self.nChannels:
            self.alphas = nn.Parameter(torch.ones(channel))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _blk_norm(self, out):
        # per-sample RMS normalization at a stage output. NO parameters. block_norm=False -> no-op.
        if not self.block_norm:
            return out
        rms = out.flatten(1).pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
        return out / rms.view(-1, 1, 1, 1)

    def forward(self, x, feat=None):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self._blk_norm(self.layer1(out))
        out = self._blk_norm(self.layer2(out))
        out = self._blk_norm(self.layer3(out))
        out = self._blk_norm(self.layer4(out))
        out = F.avg_pool2d(out, 4)
        original_out = out.view(out.size(0), -1)
        out = self.scale * original_out/original_out.norm(dim=1).reshape([-1, 1])
        if feat:
            return original_out, self.linear(out)
        return self.linear(out)

    def extract_feature(self, x, only_feature=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self._blk_norm(self.layer1(out))
        out = self._blk_norm(self.layer2(out))
        out = self._blk_norm(self.layer3(out))
        out = self._blk_norm(self.layer4(out))
        out = F.avg_pool2d(out, 4)
        feat = out.view(out.size(0), -1)
        feat = self.scale * feat/feat.norm(dim=1).reshape([-1, 1])
        out = self.linear(feat)
        if only_feature:
            return feat, None
        return feat, out


def ResNet18_zbn(num_classes=10, scale=1.0, block_norm=True):
    return ResNet_zbn(BasicBlock, [2, 2, 2, 2], num_classes, scale=scale, block_norm=block_norm)
