# MEMO — 다른 서버(GPU 4장)에 부탁하는 일

작성 2026-09-03. 브랜치 `awp-longschedule-20260730`. 논문 PDF는 `writting_docs/paper/main.pdf`.

---

## 1. 부탁하는 일 (요약)

**`main.pdf`의 Table 3과 Table 4를 채우는 것.** 둘 다 우리 행만 있거나 비어 있고, 베이스라인이
전부 `--`입니다.

| | 내용 | 지금 상태 |
|---|---|---|
| **Table 3** (p11) | Tiny-ImageNet-200, ResNet-18 | 우리 행(55.16 / 20.54)만 있고 **베이스라인 전부 비어 있음** |
| **Table 4** (p12) | WideResNet-34-10, CIFAR-10 + CIFAR-100 | CIFAR-10 우리 행(88.67 / 55.29)만. **CIFAR-100은 우리 것조차 없음**, 베이스라인 전부 비어 있음 |

이 머신(GPU 1장)은 애블레이션과 ResNet-18 CIFAR 실험을 계속합니다. **WRN과 Tiny-ImageNet은
그쪽 4장에서 해주시면 됩니다.**

---

## 2. 지금까지 한 실험 요약

### 방법 (한 줄)

자연 학습한 같은 구조의 네트워크를 티처로 두고, **학생의 적대 피처를 티처의 clean 피처에 앵커**한다.

```
L = E ‖ Φ_s(x_adv) − Φ_t(x) ‖²        (공격도 같은 양을 최대화)
```

티처는 **clean 지점에서만** 읽는다. 로스 가중치 없음, 온도 없음, 분류기는 티처 것을 그대로 씀
(학습 안 함). 여기에 per-sample 공격 반경(총예산 보존)을 얹은 것이 전부.

### 우리 결과 (전부 AutoAttack)

| 데이터셋 / 구조 | clean | AA | NRR | 비고 |
|---|---:|---:|---:|---|
| CIFAR-100 / ResNet-18 | 62.17 | 28.86 | 39.42 | 챔피언 |
| CIFAR-10 / ResNet-18 | 84.96 | 51.74 | 64.31 | 챔피언 |
| Tiny-ImageNet / ResNet-18 | 55.16 | 20.54 | 29.93 | **발표값 전부를 양축으로 이김** |
| **CIFAR-10 / WRN-34-10** | **88.67** | **55.29** | **68.11** | **발표값 31행 중 30행을 양축으로 이김** |
| CIFAR-100 / WRN-34-10 | — | — | — | **미실행 — 그쪽에서 해주실 것** |

### 이번 주 핵심 발견 세 가지

1. **앵커가 강건성을 직접 산다.** 스택(WA·AWP·per-sample ε)을 다 빼고 라벨 CE와 맞대면
   **AA +5.79, clean +6.61**. 전엔 "앵커는 clean만 올리고 강건성은 WA/AWP가 준다"고 썼는데 틀렸음.
2. **메커니즘은 국소 안정성이 아니라 클래스 분리.** 두 모델의 피처 진동은 사실상 같은데
   (0.275 vs 0.254), 클래스 분리도가 **5.5배** 차이 남 (Sw/Sb 0.898 vs 4.960).
3. **티처를 x_adv에서 읽으면 강건성이 0이 된다** (76.11 / 0.00). `Φ_s = Φ_t`가 최적해가 되고
   그건 자연 모델이라서. clean 지점에서 읽는 게 목적함수가 자명해지지 않는 유일한 이유.

### 포팅해서 직접 돌린/돌릴 베이스라인

전부 `methods.py`에 들어가 있음. **`../IGDM` 같은 데서 가져올 것 없음.**

| `method:` | 방법 | 상태 |
|---|---|---|
| `madry_at` | PGD-AT | ✅ |
| `trades` / `mart` | TRADES / MART | 진행 중 |
| `ard` / `rslad` / `adaad` / `adaad_igdm` | 증류 4종 | ✅ CIFAR 양쪽 |
| `hat` | HAT (ICLR'22) | 진행 중 |
| `lbgat` | LBGAT (ICCV'21) | 진행 중 |
| `adr` | ADR (ICLR'24) | 대기 |
| `consistency` | Consistency-AT | 대기 |

### 알아두면 좋은 함정

- **CIFAR-100 `arch: WideResNet`이 예전엔 조용히 ResNet-18을 만들었다** (커밋 `c5d4458`에서 수정).
  체크아웃이 낡으면 로그는 WRN인데 실제론 11.2M입니다. `python scripts/check_arch.py`로 확인.
- **우리 WRN이 48.3M로 찍히는데 정상**입니다. `sub_block1`이라는 죽은 블록(forward에서 안 씀,
  gradient 0)이 2.10M이고, 빼면 46.16M = 표준 WRN-34-10. TRADES 코드에서 물려받은 것이고
  LBGAT 공개 코드에도 똑같이 있습니다. **지우지 마세요** — 기존 체크포인트가 그 키를 갖고 있습니다.
- **GPU 하나에 학습 하나.** `main.py`에 flock 기반 락이 들어 있고 **GPU별로 분리**돼 있습니다
  (`CUDA_VISIBLE_DEVICES`로 키를 잡음). 실패가 아니라 **대기**하므로 큐만 걸어두면 됩니다.

---

## 3. TODO — 그쪽 서버

### 3.0 먼저 (필수)

```bash
git fetch && git checkout awp-longschedule-20260730 && git pull
python scripts/check_arch.py      # CIFAR-100 WRN 이 ~48.3M 이어야 함. 11.2M 이면 pull 안 된 것
```

### 3.1 티처가 선행 — 이게 제일 긴 항목

**세 개 다 없으면 아무것도 못 돌립니다.** GPU 3장에 동시에 올리면 하루 안에 끝납니다.

| 체크포인트 | config | 대략 |
|---|---|---|
| `CIFAR10/checkpoint/clean_wrn_200ep` | `config/CIFAR10/clean_wrn_200ep.yaml` | ~20h |
| `CIFAR100/checkpoint/clean_wrn_200ep` | `config/CIFAR100/clean_wrn_200ep.yaml` | ~20h |
| `TinyImageNet/checkpoint/clean_200ep` | `config/TinyImageNet/clean_200ep.yaml` | ~25h |

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config_name clean_wrn_200ep.yaml --dataset CIFAR10  --seed 0 &
CUDA_VISIBLE_DEVICES=1 python main.py --config_name clean_wrn_200ep.yaml --dataset CIFAR100 --seed 0 &
CUDA_VISIBLE_DEVICES=2 python main.py --config_name clean_200ep.yaml --dataset TinyImageNet --seed 0 &
```

⚠ Tiny-ImageNet은 **반드시 200에폭 티처**입니다. 80에폭짜리는 학생을 55.16/20.54가 아니라
57.08/18.96으로 만들고, 그건 논문에서 별도 애블레이션으로 씁니다.

### 3.2 Table 4 — WideResNet-34-10, CIFAR-10 과 CIFAR-100

**우리 셀 (제일 중요, 먼저)**

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config_name wrn_champ_freezehead.yaml --dataset CIFAR100 --seed 0
```

CIFAR-10은 이미 나왔습니다(88.67/55.29). **CIFAR-100이 우리한테 제일 큰 기회**입니다 —
발표값이 clean 끝(ARREST 73.05/24.32, LBGAT 70.03/27.05)과 robust 끝(ADR 62.21/31.60)으로 갈려
있고 **가운데가 통째로 비어 있습니다.** 대략 67/31이면 두 지표(Avg, NRR)를 동시에 1위로 잡습니다.

**베이스라인** — `run_tin_wrn_fill.sh`가 순서대로 돌립니다.

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/run_tin_wrn_fill.sh WRN100 core
CUDA_VISIBLE_DEVICES=2 bash scripts/run_tin_wrn_fill.sh WRN10  core
```

`core` = 우리 것 + PGD-AT + AdaAD + PGD-AT@티처init (4행). 시간 남으면 `core` 대신 인자 없이
돌리면 TRADES / MART / ARD / RSLAD / IGDM / LBGAT / Consistency / ADR 까지 8행 더 붙습니다.

**추가로 돌려주시면 좋은 것** (CIFAR-10 WRN, 우선순위 낮지만 값어치 큼):

```bash
CUDA_VISIBLE_DEVICES=3 python main.py --config_name wrn_champ_eps8.yaml --dataset CIFAR10 --seed 0
```

우리 88.67/55.29가 발표값 31행 중 30행을 양축으로 이기는데 **ARREST(90.24/50.20)만 못 이깁니다**
(clean이 1.57 낮아서). 학습 반경을 8.8/255 → 8/255로 내리면 **90.93/54.70 정도로 투영**되고,
그러면 ARREST도 양축으로 이깁니다. 8/255는 이 분야 표준 반경이라 제일 트집 잡기 어려운 형태입니다.

### 3.3 Table 3 — Tiny-ImageNet-200, ResNet-18

우리 행은 이미 있고 **베이스라인이 전부 비어 있습니다.**

```bash
CUDA_VISIBLE_DEVICES=3 bash scripts/run_tin_wrn_fill.sh TIN core
```

`core`에 **HAT이 들어가 있습니다.** 지금 논문이 인용하는 HAT의 Tiny-ImageNet 값(52.60/18.14)이
사실 **PreActResNet-18**이라(ADR이 ResNet-18 항목 아래 잘못 넣음, HAT 논문 Table 9가 PreAct라고
명시) 우리 ResNet-18 측정으로 바꿔야 합니다. **발표값 중 clean 최고 행이라 우리 주장이 그 상대로
서술됩니다.**

### 3.4 4장 배분 제안

티처가 끝난 뒤 기준입니다.

| GPU | 할 일 |
|---|---|
| 0 | `wrn_champ_freezehead` CIFAR-100 → 끝나면 `WRN100 full` 나머지 |
| 1 | `run_tin_wrn_fill.sh WRN100 core` |
| 2 | `run_tin_wrn_fill.sh WRN10 core` |
| 3 | `run_tin_wrn_fill.sh TIN core` → 끝나면 `wrn_champ_eps8` CIFAR-10 |

WRN 셀당 CIFAR/ResNet-18의 약 **4.5배**, Tiny-ImageNet은 약 **4배**입니다.

---

## 4. 결과 돌려줄 때

`results/<DATASET>/<ARCH>/<config>/<timestamp>.log` 마지막 두 줄이면 충분합니다:

```
{'last_clean_acc': ..., 'last_fgsm_acc': ..., 'last_pgd20_acc': ..., 'last_pgd10_acc': ...,
 'last_pgd50_acc': ..., 'last_cw_acc': ...}
{'last_aa_acc': ...}
```

⚠ **숫자만 보내주실 거면 어느 지표인지 순서를 같이 적어주세요.** 지난번에 순서가 엇갈려서
PGD-10과 PGD-20을 바꿔 읽을 뻔했습니다. 논문이 읽는 건 **clean과 AA** 둘이고 PGD는 참고입니다.

더 자세한 내용(챔피언 레시피 31개 노브 전체, config 이름, 함정)은
`writting_docs/paper/TODO.md` **§4**에 있습니다.
