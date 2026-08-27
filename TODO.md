# TODO — 선생 기하가 학생 운용점을 정하는가 (Tiny-ImageNet, 다른 서버)

작성 2026-08-27. 이 파일 하나로 실행 가능하도록 씀. 배경 이론은 `METHOD.md` §8.

---

## 0. 왜 하는가 — 관측 하나와 측정 하나

**관측 (Tiny-ImageNet, 다른 서버) — 선생 3점, 세 열 모두 단조.** config는 안 바꾸고 선생만 교체.

| 선생 | 선생 clean | 학생 clean | 학생 AA | NRR |
|---|---:|---:|---:|---:|
| `clean_80ep` (80ep) | 65.97 | 57.08 | 18.96 | 28.46 |
| 100ep | ? | 56.74 | 19.09 | 28.57 |
| 200ep | **66.29** | **55.16** | **20.54** | **29.93** |
| ADR+WA+AWP (80ep) | — | 48.27 | 20.10 | 28.38 |

- 200ep 셀 vs ADR: clean **+6.89** / AA **+0.44** / NRR **+1.55** — 프로젝트 최초로 **양 축 동시 승**.
- 80→200ep: clean **−1.92**, AA **+1.58**, NRR **+1.47**.
- 그런데 **선생 정확도는 +0.32밖에 차이 안 남**(65.97 → 66.29). 전달된 건 선생의
  *정확도*가 아니라 *기하*라는 뜻.
- ⚠ **NRR이 아직 봉우리를 안 지났다** (28.46 → 28.57 → 29.93, 계속 상승). 200ep이 끝이 아닐 수
  있으므로 **사다리를 위로 늘리는 것이 최우선**이다. 100ep 선생의 clean 값은 아직 미기록 — 채울 것.

**측정 (CIFAR-100 선생 사다리, 이쪽 서버, 학습 0회).** 정확도가 포화한 뒤에도 기하는 단조로 계속 변한다.
`results/CIFAR100/teacher_collapse_ladder_20260827.log`

| 선생 | clean | ‖Φ‖ | eff.rank | **Sw/Sb** | 공격 회전각 | 노름팽창 |
|---|---:|---:|---:|---:|---:|---:|
| 50ep | 75.81 | 13.25 | 56.2 | 1.100 | 59.2° | ×2.06 |
| 100ep | 76.62 | 12.44 | 60.2 | 0.982 | 61.1° | ×2.22 |
| 150ep | 77.52 | 11.80 | 61.8 | 0.887 | 62.4° | ×2.35 |
| 200ep | 77.65 | 11.20 | 62.4 | 0.808 | 63.8° | ×2.45 |
| 300ep | 78.32 | 10.59 | 61.8 | 0.712 | 65.4° | ×2.53 |

150→300ep: 정확도 **+0.80**인데 Sw/Sb **−19.7%**. 100→200ep(=TIN 스왑 구간): 정확도 +1.03, Sw/Sb **−17.7%**.

**빠져 있는 것: 선생별 학생 런.** 선생 기하 → 학생 운용점의 인과 고리를 지지하는 건 위 TIN 스왑 **1건뿐**이다.
CIFAR에는 선행 증거가 **없다**(옛 `featdir_awp_100ep_eps10`은 ε_tr 10/255·λ 1.5 등 레시피가 달라 증거로 못 씀).

---

## 1. 가설과 **사전 등록 예측**

> **가설 H.** 자연 선생을 오래 학습시키면 정확도가 포화한 뒤에도 피처가 클래스 평균으로 수축한다
> (neural collapse, Sw/Sb ↓). 앵커는 그 기하를 복사하므로, 더 수축한 선생은 학생을
> clean↓ / AA↑ 쪽으로 민다.
>
> - 수축한 타깃 = 샘플별 구조(dark knowledge)가 적음 → 물려받을 clean 정보가 적음 → **clean ↓**
> - 수축한 타깃 = K개 점에 가까움 → ε-볼 위에서 상수로 잡기 쉬움 → $O$ 작아짐 → **AA ↑**
> - $F$/$O$ 언어로: **선생의 붕괴가 fidelity 정보량을 oscillation 달성가능성과 맞바꾼다.**

**돌리기 전에 박아두는 예측:**

- **P1. ✅ 지지됨 (선생 3점, 2026-08-27).** 선생 에폭 80→100→200에서 학생 clean 57.08→56.74→55.16,
  AA 18.96→19.09→20.54 — 단조. 남은 것은 이것이 **에폭**이 아니라 **Sw/Sb**를 따라간다는 확인(§3).
- **P2.** 그 관계는 선생 *정확도*보다 Sw/Sb와 더 잘 맞는다. TIN에서 선생 정확도는 사실상 상수인데
  학생이 크게 움직였으므로, 정확도로는 설명이 안 돼야 한다.
- **P3. ⏳ 아직 안 맞음 — 이게 지금 제일 정보량 큰 셀.** NRR이 28.46→28.57→**29.93**으로 200ep까지
  계속 오른다. 봉우리가 200ep보다 위에 있거나 없다. **400ep(가능하면 800ep)이 최우선 런.**
  거기서 NRR이 꺾이면 P3 확인 + 최적 선생 길이라는 실용적 결론, 계속 오르면 "선생은 길수록 좋다"는
  더 단순하고 더 센 결론.
- **P4.** ⚠ **반례 예측** — 회전각과 노름팽창도 사다리에서 단조인데 **방향이 H와 반대**다
  (오래 학습한 선생이 *더* 흔들린다: 59.2°→65.4°). 그런데 그 더 흔들리는 선생이 학생 AA를
  올렸다. 따라서 "안정적인 선생 → 안정적인 학생"은 **이미 기각**. 만약 학생 AA가 선생
  회전각과 양의 상관을 보이면, 그건 H가 아니라 **아직 이름 없는 제3의 축**이다.

**H를 죽이는 조건:** Sw/Sb ↔ (학생 clean, AA)가 단조가 아니거나, 부호가 P1과 반대이거나,
선생 정확도가 Sw/Sb보다 학생 성능을 더 잘 설명하면 → H 폐기, TIN 이동은 다른 걸로 설명해야 함.

---

## 2. Track A (본체) — TIN 선생 사다리 × 학생

**학생 config는 손대지 않는다.** `config/TinyImageNet/featdir_tin_100ep.yaml`이 이미
요구사항과 정확히 일치: L2 타깃(`featdir_rawteacher: True`, `featdir_rawstudent: True`,
`student_norm: False`) + **헤드 학습 없음**(`featdir_freeze_head: True`). 바꾸는 것은
**선생 경로 두 줄뿐**:

```yaml
checkpoint          : TinyImageNet/checkpoint/<TEACHER>/clean_last.pkl
finetune_checkpoint : TinyImageNet/checkpoint/<TEACHER>/clean_last.pkl
```

### 2.1 선생 준비

이미 있는 것: `clean_80ep`(65.97), 그리고 새 200ep 선생.
사다리를 만들려면 최소 4점이 필요하다. **OneCycle이라 중간 에폭 체크포인트를 쓰면 안 된다**
(LR이 아직 안 식었으므로 "40에폭 선생"이 아니다). 각 점은 **자기 학습 런**이어야 한다.

`config/TinyImageNet/clean_200ep.yaml`을 복사해 `epochs`만 바꿔 만들 것:

| 선생 | config | 상태 | 우선순위 |
|---|---|---|---|
| **400ep** | `clean_400ep.yaml` | 새로 | **1순위 — NRR 봉우리 탐색** |
| 800ep | `clean_800ep.yaml` | 새로 | 2순위 (400ep에서도 안 꺾이면) |
| 40ep | `clean_40ep.yaml` | 새로 | 3순위 (아래쪽 앵커) |
| 80ep / 100ep / 200ep | 있음 | ✅ 학생까지 완료 | — |

자연 학습이라 PGD 내부 루프가 없어 싸다. 위쪽 끝이 미지라 거기부터 채운다.

### 2.2 학생 런

선생 4개 × 학생 1런. 다른 건 전부 고정.

```bash
for T in clean_400ep clean_800ep clean_40ep; do   # 80/100/200은 이미 끝남
  sed -e "s|clean_80ep/clean_last.pkl|${T}/clean_last.pkl|g" \
      config/TinyImageNet/featdir_tin_100ep.yaml \
      > config/TinyImageNet/tladder_${T}.yaml
  python main.py --config_name tladder_${T}.yaml --dataset TinyImageNet --seed 0
done
```

`aa: True`이므로 AA까지 나온다. **`interval: 5` 유지, 시드 0 고정.**

### 2.3 ⚠ 교락 하나 — 알고 가야 함

선생을 바꾸면 **앵커 타깃과 학생 초기값이 동시에** 바뀐다(같은 체크포인트를 둘 다 쓰므로).
방법의 정의상 한 덩어리라 이대로가 기본 비교 단위지만, 분리하려면 셀 하나 더:

| 셀 | `checkpoint`(타깃) | `finetune_checkpoint`(초기값) |
|---|---|---|
| 분리 | `clean_400ep` | `clean_80ep` |

이게 400ep 학생과 비슷하게 나오면 **타깃 기하**가 원인, 80ep 학생과 비슷하면 **초기값**이 원인.
Track A가 신호를 보이면 그때 돌리면 된다.

---

## 3. Track B — 선생별 기하 (학생 런과 짝지을 x축)

각 선생에 대해 §0 표와 같은 숫자를 뽑는다. 스크립트는 이쪽에 있음:

```bash
# CIFAR100용으로 짜여 있으니 TEACHERS/CK/dataset 세 군데만 TinyImageNet으로 바꿀 것
python scripts/diag_teacher_collapse.py
```

뽑을 것: clean, ‖Φ‖, norm CV, eff.rank(unit), **Sw/Sb**, 공격 회전각, 노름팽창.
학생 표와 합쳐 **가로축 Sw/Sb, 세로축 (학생 clean, 학생 AA)** 산점도가 논문 그림 후보.

옵션: 학생 쪽 기하도 같이 재면 고리가 완성된다.
```bash
python scripts/diag_retention_within_family.py   # CELLS를 TIN 학생들로 교체
```
→ 학생 cos(Φ_s, Φ_t)가 선생 Sw/Sb와 어떻게 움직이는지. **선생이 수축할수록 학생이 더 잘
붙어야 한다**(타깃이 단순하니까)는 부수 예측이 여기서 검증됨.

---

## 4. Track C (선택) — CIFAR-100 거울

선생 사다리가 **이미 5개 다 디스크에 있다**(`CIFAR100/checkpoint/clean{,_100ep,_150ep,_200ep,_300ep}`).
학생 config는 L2 + 헤드 학습 없음으로 맞춰야 한다. `config/CIFAR100/l2_bestrecipe_angeps.yaml`
기준으로:

```yaml
featdir_freeze_head : True     # 추가 — 헤드 학습 없음
checkpoint          : CIFAR100/checkpoint/<TEACHER>/clean_last.pkl
finetune_checkpoint : CIFAR100/checkpoint/<TEACHER>/clean_last.pkl
```

⚠ **미해결 하나**: `featdir_champ200_freezehead.yaml`은 `featdir_freeze_head: True`와 함께
`feat_scale: 11.23`을 준다(≈ 티처 ‖Φ‖). 그건 정규화된 헤드(`student_norm: True`) 파이프라인용
스케일이다. L2 셀은 `student_norm: False`라 헤드가 raw 피처를 읽으므로 `feat_scale`이 필요한지
불명. **첫 셀에서 epoch 0 clean accuracy를 확인할 것** — 티처와 비슷하면(≈77) 정상, 1%대로
떨어지면 스케일 문제이므로 `feat_scale`을 붙여야 한다.

Track A가 우선. C는 결과가 재현되는지 보는 용도.

---

## 5. 보고할 것

셀당 한 줄이면 충분:

```
teacher=<name>  T.clean=..  T.Sw/Sb=..  T.angle=..  |  S.clean=..  S.PGD20=..  S.CW=..  S.AA=..  S.NRR=..
```

그리고 P1~P4 각각에 대해 **맞았는지 틀렸는지 한 줄씩.** 틀린 게 있으면 그게 제일 중요한 정보다.

---

## 6. 이게 맞으면 논문에 뭐가 생기나

**선생 학습 길이 = 트레이드오프 다이얼.** 공짜(자연 학습), config 변경 없음, 그리고
**로버스트 선생 KD 계열엔 없는 노브** — 저쪽 선생의 기하는 적대적 학습이 정해버리므로
따로 돌릴 수 없다.

⚠ 문구 주의: **"로버스트 선생은 불필요하다"로 쓰면 안 된다**(랭킹 주장이라 즉사).
쓸 수 있는 형태는 **"선생 강도/기하 축이 자연학습 비용으로 열린다"** — 우리 방법의 성질에
대한 진술이지 남의 방법을 깎는 게 아니다.
