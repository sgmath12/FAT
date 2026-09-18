# 리뷰어 설득을 위한 증거 정리 (2026-09-18)

목적은 점수 자체를 예측하거나 문장을 과장하는 것이 아니라, 거절의 핵심 근거를 해소하고 채택할 과학적 이유를 명확히 하는 것이다. 아래 계획은 새 학습을 실행하라는 명령이나 진행 상태가 아니다. 이번 작업에서는 원고와 기존 결과만 정리했다.

## 1. 논문의 중심 주장

자연 티처의 clean target에 대한 직접 회귀는 별도 label objective 없이 adversarial training의 supervision이 될 수 있다. CFA는 이를 clean feature target과 inherited classifier로 구현한다. Teacher의 perturbed-input 행동을 모방하지 않고 clean target에 대한 student의 안정성을 학습한다.

Feature가 모든 logit objective보다 우월하다는 주장은 하지 않는다. Logit MSE가 잘 작동한다는 관찰은 clean-target regression이라는 설명을 뒷받침한다. Sole objective라는 단순성이 다양한 조건에서 작동한다는 것이 중요한 경험적 기여다.

고정 anchor 부등식은 이 설계가 어떤 양을 제어하는지 설명하는 보조 결과다. Triangle inequality에 기초하고 logit norm regression에도 적용된다. 이것을 새로운 강건성 보증이나 feature-specific 이론으로 제시하지 않는다.

## 2. 이번에 원고에 반영한 증거와 수정

### 누락되어 있던 stacked logit MSE

| 조건 | Clean | APGD-CE/APGD-T |
|---|---:|---:|
| Logit MSE, fixed head | 60.36 | 28.02 |
| Feature regression, fixed head | 60.42 | 28.42 |

- CIFAR-100 / ResNet-18 / 100 epochs / WA + AWP / uniform training epsilon 8.8/255 / evaluation epsilon 8/255 / seed 0.
- 두 objective 모두 출력 좌표별 squared error를 합하고 batch 평균을 취한다. 각 objective를 inner attack과 outer update에 사용한다.
- 동일 teacher, frozen inherited head, AdamW 0.021, one-cycle schedule, WA 시작 20%, AWP gamma 0.005 / warmup 10.
- 이는 sensitivity allocation을 포함한 최종 CFA와의 비교가 아니다. 올바른 feature 대조군은 uniform-radius CFA 60.42 / 28.42다.
- 원고 Section 5.4와 Appendix `tab:regression_stack`에 추가했다. 0.40 AA 차이를 유의한 feature 우월성으로 해석하지 않는다.

확인한 원본 (연구 저장소 루트 `D:/research/FAT` 기준):

- `results/CIFAR100/ResNet18/logitmse_stack_100ep/2609100235.log` (last_clean_acc / last_aa_acc)
- `results/CIFAR100/ResNet18/ladder_p0_wa_awp_fh_100ep/2609062124.log`
- `config/CIFAR100/logitmse_stack_100ep.yaml`
- `config/CIFAR100/ladder_p0_wa_awp_fh_100ep.yaml`
- `methods.py`: `train_logit_mse`, `_DistillStack`

### 주장과 근거의 일치

- Introduction은 feature-exclusive superiority 대신 clean-target regression의 sufficiency를 중심에 둔다.
- Contributions에 teacher clean accuracy와 student clean accuracy의 불일치 및 radius 분포를 고정한 signal control을 드러냈다.
- 본문 proposition 해석에서 norm-based logit regression에도 적용됨을 명시했다.
- Appendix의 '기존 방법 robustness는 label term에서만 온다'는 입증되지 않은 귀속을 제거했다.
- Teacher-only KL 대조군을 '최선의 logit 방법'처럼 부르던 표현을 제거했다. Logit MSE와 구분한다.
- 두 seed의 관측 차이를 다른 실험의 유의성 기준처럼 사용하던 문장을 정리했다.
- 초록, 성능 수치, Figure 2/3 데이터는 변경하지 않았다. ARREST 비교를 유지했다.

## 3. 5 -> 6: 실제로 점수를 막는 우려 해소

### A. 실험 조건과 비교 범위의 일관성

완료: base 및 WA/AWP 하의 logit MSE 대조군을 모두 제시. Feature만의 필연적 우위는 주장하지 않음.

남은 확인: 최종 adaptive-radius recipe에서 logit MSE를 평가한 완료 결과는 이번 점검에서 찾지 못했다. 필요하면 logit objective 자체의 gradient로 radius를 배정하고 step size도 동일 multiplier로 조정해야 한다. Feature용 radius를 그대로 주면 다른 질문이 된다. 성능이 비슷해도 regression이라는 중심 결론은 유지된다.

### B. 가까운 비교에서 반복성 확인

현재 final CFA는 seed 0/1, 많은 대조군은 단일 run이다. 0.3~0.4 AA 차이를 강한 우월성으로 제시하지 않는다.

학습을 추가한다면: 먼저 핵심 CIFAR-100 비교 하나를 정해 같은 seed 목록, teacher checkpoint, 평가 프로토콜로 반복한다. 예: uniform-radius feature vs logit MSE, 또는 비슷한 clean operating point의 CFA vs ARREST. 모든 ablation을 무작정 반복하는 것보다 주장에 직접 연결되는 pair가 우선이다. 평균/표준편차와 seed별 차이를 보고하며, 소수 seed에서 유의성을 과장하지 않는다.

### C. 기하 분석의 대상

현재 주된 anchor-vs-CE 기하 진단은 earlier trained-head models다. 본문에서 scope를 공개했지만 final method의 설명으로 사용하려면 frozen-head 최종모델과 대응 CE에서 같은 데이터/공격/BN mode로 재측정해야 한다.

측정 후보: clean/adversarial within-between scatter, sample-to-own centroid angle, nearest-centroid gap. 기존 normalized diagnostics의 정의를 유지하고 raw training loss와 혼동하지 않는다. 결과가 유지되지 않으면 그 분석을 최종 recipe의 설명으로 쓰지 않는다.

### D. 전체 AutoAttack

확인 완료: CIFAR-100 final-recipe seed-1, 10000 test images, epsilon 8/255에서 2-attack 28.73 / 4-attack 28.73. 추가 FAB-T/Square 성공 0. 원본 `logs/aa_full_20260906.log`.

모든 데이터셋과 모든 주요 checkpoint의 four-attack 실행으로 일반화하지 않는다. 사용자가 다른 완료 결과를 보유하면 해당 로그/체크포인트를 연결해 범위를 확장하면 된다. 동일 검증을 무조건 다시 실행할 필요는 없다.

## 4. 6 -> 8: 왜 이 발견이 중요한지 설득

### A. 최소 설계가 여러 조건에서 작동한다

별도 label objective나 temperature search 없이 robust backbone을 학습한다는 실용적 가치가 핵심이다. 더 복잡한 정리를 만드는 것이 필수 조건은 아니다. Base regime, optimizer/schedule control, WA/AWP, 여러 dataset/architecture가 이를 지지하도록 묶는다. 단순함 자체보다 유효 범위를 입증하는 것이 중요하다.

### B. Teacher 선택에 실제 의사결정 지침을 준다

이미 있는 관찰: CIFAR-100 checkpoint sweep에서 teacher clean은 상승하지만 student clean은 하락하며 robustness는 상승한다. Tiny-ImageNet의 80/200-epoch teacher 비교도 같은 방향이다.

더 강한 증거가 필요하다면: 같은 checkpoint 비교를 student seed별로 반복하거나 별도의 natural-teacher trajectory에서 확인한다. 이것은 재현성 질문이다. Geometry가 원인이라는 더 강한 주장을 하려면 initialization/head와 target을 분리한 통제가 추가로 필요하다. 두 질문을 혼동하지 않는다.

### C. Adaptive radius에서 중요한 것은 분포가 아니라 배정 신호다

이미 exact multiplier multiset을 보존한 difficulty-order control이 있어, 단순히 여러 epsilon을 섞어서 생기는 효과와 구분할 수 있다. 새로운 휴리스틱을 더 늘리기보다 이 통제의 의미를 명확히 전달한다. 추가 확인은 같은 paired seeds에서 효과의 방향이 유지되는지에 집중한다.

## 5. 실행 순서 제안

1. 기존 완료 결과를 먼저 연결: WRN CIFAR-100, 타 주요 checkpoint의 full AA, final-model geometry 결과의 존재 확인.
2. 핵심 비교 pair 하나를 정해 반복성 보강. 어떤 우려를 해결할지 정한 뒤 GPU 작업을 시작.
3. Teacher-checkpoint 발견의 독립 반복을 통해 practical selection insight의 재현성 보강.

큰 성능 차이가 나오도록 조건을 고르지 않는다. 가까운 대조군을 삭제하지 않는다. Negative/null 결과는 주장의 범위를 정하는 정보로 사용한다. 어떤 점수 조합이나 채택도 이 계획으로 보장할 수 없다.
