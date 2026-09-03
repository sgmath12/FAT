"""Guard against the bug fixed in c5d4458: CIFAR-100 `arch: WideResNet` silently built a ResNet-18.

Builds the student the way main.py would for each dataset and prints its parameter count.  A
CIFAR-100 WideResNet that reports ~11.2M rather than ~48.3M means the checkout predates the fix, and
every CIFAR-100 WideResNet cell run on it is a ResNet-18 wearing the wrong label.

WHY 48.3M AND NOT THE 46.2M THE LITERATURE REPORTS (checked 2026-09-03).  Our WideResNet.py carries a
`sub_block1`, a duplicate of block1 that is constructed but never referenced in forward -- measured,
it receives zero gradient -- and it accounts for exactly 2.10M parameters.  48.26 - 2.10 = 46.16M,
which is the standard WRN-34-10 to the second decimal, and matches RPAT's and HAT's implementations
exactly.  The dead block is inherited from the TRADES codebase; LBGAT's released WideResNet has it
too, at the same 48.26M.  So the network we train IS WRN-34-10 and is comparable with every published
WideResNet number; only the parameter count printed here is inflated.  Do not "fix" it by deleting
sub_block1 -- every existing WideResNet checkpoint has those keys and would fail to load.
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
