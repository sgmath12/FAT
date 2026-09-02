"""Guard against the bug fixed in c5d4458: CIFAR-100 `arch: WideResNet` silently built a ResNet-18.

Builds the student the way main.py would for each dataset and prints its parameter count.  A
CIFAR-100 WideResNet that reports ~11.2M rather than ~48.3M means the checkout predates the fix, and
every CIFAR-100 WideResNet cell run on it is a ResNet-18 wearing the wrong label.
"""
import sys; sys.path.insert(0, '/mnt/d/research/FAT')
import types, utils

ok = True
for ds, arch, want in [('CIFAR10', 'WideResNet', 48.3), ('CIFAR100', 'WideResNet', 48.3),
                       ('CIFAR100', 'ResNet18', 11.2)]:
    c = types.SimpleNamespace(dataset=ds, arch=arch, reformation=False, student_norm=False,
                              teacher_norm=False, load=False, finetune=False, eta=512,
                              gain_head=False, checkpoint='', method='madry_at', convert=False)
    _, s = utils.get_model(c)
    n = sum(p.numel() for p in s.parameters()) / 1e6
    good = abs(n - want) < 1.0
    ok &= good
    print(f'{ds:9s} {arch:11s} {n:6.1f}M  expected ~{want}M  {"OK" if good else "*** WRONG ***"}')
print('\nall good' if ok else '\nFAILED -- see the note at the top of this file')
sys.exit(0 if ok else 1)
