# 제출 2. 모델 확장 및 비교 보고서

## 프로젝트 개요

본 보고서는 제출 1에서 설계한 구독 해지 후보 분류 모델을 확장하여, 복수의 머신러닝 모델을 체계적으로 비교·분석한 결과를 정리한 문서이다. 데이터는 `mock_data_2.csv` (20,000건)로 교체되었으며, 피처 엔지니어링을 통해 12개의 파생 변수를 추가로 설계하였다. 비교 대상 모델은 **Logistic Regression**, **Random Forest**, **CatBoost** 세 가지이며, 각 모델의 선택 근거와 성능 차이의 원인을 중심으로 분석하였다.

---

## 1. 데이터 변경 및 피처 엔지니어링

### 1-1. 데이터 변경 사항

제출 1에서 사용한 `mock_data.csv` (10,000건)에서 `mock_data_2.csv` (20,000건)으로 데이터를 확장하였다. 변수 구성은 아래와 같이 변경되었다.

| 항목 | 제출 1 (`mock_data.csv`) | 제출 2 (`mock_data_2.csv`) |
| --- | --- | --- |
| 데이터 크기 | 10,000건 | 20,000건 |
| 타깃 분포 (0 : 1) | 2,058 : 7,942 (21% : 79%) | 6,000 : 14,000 (30% : 70%) |
| 주요 변수 | 수치형 중심 (이용시간, 비용 등) | 범주형 + 순서형 + 수치형 혼합 |

새로운 데이터는 범주형·순서형 변수를 포함하며, 클래스 불균형이 다소 완화(21% → 30%)되어 있다. 그러나 여전히 해지 후보(클래스 0)가 소수 클래스이므로 이를 고려한 모델링이 필요하다.

### 1-2. 주요 변수 설명

| 변수명 | 타입 | 설명 | 사용자 관점 해석 |
| --- | --- | --- | --- |
| `subscription_type` | 범주형 | 구독 서비스 유형 (Education, Music, Fitness 등) | 서비스 카테고리별 이탈 패턴 차이 반영 |
| `monthly_cost` | 연속형 | 월 구독료 (원) | 비용 부담 수준 |
| `use_frequency` | 순서형 문자 | 이용 빈도 (rare / monthly / weekly / frequent) | 서비스 활용 빈도 |
| `last_use_recency` | 순서형 문자 | 최근 사용 시점 (>30d / 7-30d / 1-7d / <1d) | 최근 활동성 지표 |
| `perceived_necessity` | 순서형 정수 | 사용자가 느끼는 필요도 (1~5) | 주관적 유지 의향 |
| `cost_burden` | 순서형 정수 | 비용 부담감 (1~5) | 가격 민감도 |
| `would_rebuy` | 순서형 정수 | 재구독 의향 (1~5) | 미래 유지 가능성 |
| `replacement_available` | 이진형 | 대체 서비스 존재 여부 (0/1) | 이탈 가능성 촉진 요인 |
| `billing_cycle` | 이진형 | 청구 주기 (0=월간, 1=연간) | 연간 약정 구독 여부 판단 |
| `remaining_months` | 연속형 | 현재 시점 기준 잔여 구독 개월 수 | 즉시 해지 가능 여부 판단 |
| `discount_amount` | 연속형 | 월 기준 할인·환급 금액 (제휴 카드 혜택, 포인트 환급 등) | 명목 비용과 실질 부담의 괴리 측정 |
| `target` | 이진형 | 예측 대상 (0=해지 후보, 1=유지 후보) | - |

### 1-3. 피처 엔지니어링

범주형 변수를 수치형으로 변환하고, 사용자의 구독 유지 가치를 종합적으로 반영하는 파생 변수를 설계하였다.

#### 순서형 인코딩

```python
use_frequency_map  = {"rare": 1, "monthly": 2, "weekly": 3, "frequent": 4}
last_use_recency_map = {">30d": 1, "7-30d": 2, "1-7d": 3, "<1d": 4}
```

숫자 크기에 의미적 순서가 있는 변수이므로 One-Hot 대신 순서형 인코딩을 적용하였다.

#### 파생 변수 목록

| 파생 변수명 | 계산식 | 설계 의도 |
| --- | --- | --- |
| `value_gap` | `use_frequency_score + last_use_recency_score + perceived_necessity - cost_burden` | 사용 가치에서 비용 부담을 뺀 순수 유지 가치 |
| `rebuy_satisfaction_gap` | `effective_monthly_cost - perceived_necessity` | 실질 비용 대비 만족도 격차 |
| `cost_to_necessity_ratio` | `effective_monthly_cost × (perceived_necessity + ε)` | 필요도를 반영한 실질 비용 절대값 |
| `log_monthly_cost` | `log(1 + effective_monthly_cost)` | 실질 비용 오른쪽 꼬리 분포 완화 |
| `monthly_cost_z` | `effective_monthly_cost` Z-score 표준화 | 스케일 통일 |
| `cost_burden_x_replacement` | `cost_burden × replacement_available` | 비용 부담 + 대체재 존재 시 이탈 위험 증폭 |
| `necessity_x_recency` | `perceived_necessity × last_use_recency_score` | 필요도와 최근성의 상호작용 |
| `frequency_x_rebuy` | `use_frequency_score × would_rebuy` | 이용 빈도와 재구독 의향 상호작용 |
| `is_high_cost` | `effective_monthly_cost > 75th percentile` → 1 | 실질 고비용 구독 여부 이진 플래그 |
| `has_churn_signal` | `(freq ≤ 2) & (recency ≤ 2) & (cost_burden ≥ 4)` → 1 | 규칙 기반 이탈 신호 복합 플래그 |
| `is_deferred` | `(billing_cycle == 1) & (remaining_months > 3)` → 1 | 연간 구독 중 잔여 기간 초과로 처리 보류 대상 플래그 |
| `effective_monthly_cost` | `max(0, monthly_cost - discount_amount)` | 할인·환급 차감 후 실질 월 결제 금액 |
| `is_zero_cost` | `effective_monthly_cost == 0` → 1 | 실질 결제 금액이 0인 구독 여부 플래그 |

`value_gap`과 `has_churn_signal`은 도메인 지식을 결합한 핵심 파생 변수로, 이탈 후보의 패턴(저빈도 + 고비용 + 낮은 만족도)을 직접 수치화한 것이다.

### 1-4. 연간 구독 처리 방안

기존 데이터 구조는 월간 구독만을 전제하고 있어, 연간 구독이 혼재될 경우 `monthly_cost` 기반의 모든 비용 피처에 왜곡이 발생할 수 있다. 이를 해소하기 위해 두 가지 처리 방안을 적용한다.

#### 방안 1. 월간/연간 플래그 기반 월환산 비용 처리

`billing_cycle` 플래그를 기준으로, 연간 구독의 경우 `annual_cost / 12`를 `monthly_cost`에 덮어써서 모든 비용 피처를 월 단위 기준으로 정규화한다.

```python
df["monthly_cost"] = df.apply(
    lambda row: row["annual_cost"] / 12 if row["billing_cycle"] == 1 else row["monthly_cost"],
    axis=1
)
```

이 처리는 이후 `rebuy_satisfaction_gap`, `cost_to_necessity_ratio`, `log_monthly_cost`, `is_high_cost` 등 `monthly_cost`를 입력으로 사용하는 모든 파생 변수에 자동으로 반영된다. 또한 `billing_cycle` 자체도 독립 피처로 모델에 포함한다. 연간 약정 구독은 단기 해지 가능성이 구조적으로 낮으므로, 이 플래그는 유지 여부 분류에 유의미한 신호가 될 수 있다.

#### 방안 2. 잔여 기간 3개월 초과 시 처리 보류

연간 구독 중 잔여 기간이 3개월을 초과하는 경우, 해지 후보 분류 대상에서 제외하여 예측을 보류한다. 아직 약정이 충분히 남아 있는 구독은 현실적으로 즉시 해지가 어렵기 때문에, 이 시점에서 해지 추천을 제공하는 것은 실용성이 낮다.

```python
DEFERRAL_THRESHOLD_MONTHS = 3

df["is_deferred"] = (
    (df["billing_cycle"] == 1) & (df["remaining_months"] > DEFERRAL_THRESHOLD_MONTHS)
).astype(int)

df_active   = df[df["is_deferred"] == 0]  # 예측 대상
df_deferred = df[df["is_deferred"] == 1]  # 처리 보류 (갱신 시점에 재평가)
```

이후 모든 모델 학습 및 평가는 `df_active` 기준으로 수행한다. 보류된 구독은 갱신 시점(잔여 3개월 이하 도달)에 재평가 대상으로 전환된다.

#### 처리 흐름 요약

| 구독 유형 | 잔여 기간 | 처리 방식 |
| --- | --- | --- |
| 월간 구독 | — | 방안 1 적용 후 즉시 예측 대상 |
| 연간 구독 | ≤ 3개월 | 방안 1으로 월환산 후 예측 대상 |
| 연간 구독 | > 3개월 | 방안 2에 따라 처리 보류 (`is_deferred = 1`) |

### 1-5. 실질 결제 금액 처리

#### 배경

`monthly_cost`는 구독 서비스의 명목 청구 금액이다. 그러나 제휴 카드 할인, 통신사 결합 혜택, 포인트 전액 환급 등으로 인해 실제 사용자의 지갑에서 지출되는 금액이 0원이 되는 경우가 존재한다. 이 경우 명목 비용만으로 "비용 부담이 크다"고 판단하면 유지 가치가 충분한 구독을 해지 후보로 과분류할 위험이 있다.

#### 처리 방식

총 결제 금액에서 할인·환급 금액을 차감한 **실질 결제 금액(`effective_monthly_cost`)**을 산출하고, 이를 이후 모든 비용 관련 피처의 기준값으로 사용한다.

```python
df["effective_monthly_cost"] = (df["monthly_cost"] - df["discount_amount"]).clip(lower=0)
df["is_zero_cost"] = (df["effective_monthly_cost"] == 0).astype(int)
```

`clip(lower=0)`은 환급액이 구독료를 초과하는 엣지 케이스(예: 가입 프로모션 과환급)를 0으로 처리하여 음수 비용이 발생하지 않도록 한다.

#### 피처 파이프라인 적용 순서

비용 관련 파생 변수는 아래 순서로 `effective_monthly_cost`를 최종 입력값으로 사용한다.

```
[원본] monthly_cost
    → (방안 1) billing_cycle == 1이면 annual_cost / 12로 월환산
    → (본 처리) effective_monthly_cost = max(0, monthly_cost - discount_amount)
    → 이후 모든 비용 파생 변수 산출에 effective_monthly_cost 사용
```

#### `is_zero_cost` 플래그의 해석

실질 결제 금액이 0인 구독(`is_zero_cost = 1`)은 비용 측면의 해지 동기가 사라진 상태다. 이 경우 모델이 비용 기반 피처에서 해지 신호를 포착하기 어려우며, 유지 여부 판단은 사용 빈도·최근성·필요도 같은 활용 기반 피처에 의존하게 된다.

| 구독 상태 | 실질 비용 | 주요 해지 판단 근거 |
| --- | --- | --- |
| 실질 결제 발생 | `effective_monthly_cost > 0` | 비용 + 활용 피처 종합 |
| 전액 할인·환급 | `effective_monthly_cost == 0` | 활용 피처 중심 (빈도, 최근성, 필요도) |

다만 `is_zero_cost` 구독이라도 이용 빈도가 매우 낮고 대체재가 존재한다면 여전히 해지 후보가 될 수 있으므로, 플래그를 독립 피처로 모델에 포함하여 맥락을 학습하도록 한다.

---

## 2. 모델 선택 및 선택 근거

### 2-1. 모델 파이프라인 구성

세 모델 모두 동일한 데이터 분할 조건 (`test_size=0.2, stratify=y, random_state=42`)을 적용하여 공정한 비교 환경을 구성하였다.

| 피처 그룹 | 전처리 방법 | 해당 변수 예시 |
| --- | --- | --- |
| 명목형 (nominal) | OneHotEncoder | `subscription_type`, `use_frequency`, `last_use_recency` |
| 순서형·연속형 (ordinal/continuous) | StandardScaler | `perceived_necessity`, `value_gap`, `log_monthly_cost` 등 |
| 이진형 (binary) | passthrough (변환 없음) | `replacement_available`, `is_high_cost`, `has_churn_signal` |

### 2-2. 모델별 선택 근거

#### ① Logistic Regression (기준 모델)

- 선형 분류 경계를 학습하는 가장 단순한 모델로, 1차 실행의 연장선상에서 기준선(baseline)을 제공한다.
- `class_weight="balanced"` 설정으로 클래스 불균형을 보정하였다.
- 피처 계수(coefficient)를 통해 변수의 방향성과 크기를 직접 해석할 수 있어, 다른 모델의 성능과 비교하는 참조점으로 활용한다.
- **한계:** 변수 간 비선형 상호작용(예: 비용이 높아도 빈도가 높으면 유지)을 포착하지 못한다.

#### ② Random Forest (비선형 앙상블)

- 다수의 Decision Tree를 병렬로 학습하는 배깅(Bagging) 기반 앙상블 모델이다.
- 비선형 결정 경계를 학습할 수 있으며, 피처 중요도(feature importance)를 통해 어떤 변수가 분류에 기여하는지 파악할 수 있다.
- `class_weight="balanced_subsample"` 적용으로 소수 클래스(해지 후보) 탐지를 강화하였다.
- 스케일링 없이 원시 수치값을 그대로 입력해도 안정적으로 학습되는 장점이 있다.
- **선택 이유:** Logistic Regression 대비 구독 이탈의 복잡한 조합 패턴(예: rare + >30d + cost_burden ≥ 4)을 더 잘 포착할 수 있다고 판단하였다.

#### ③ CatBoost (범주형 특화 그래디언트 부스팅)

- Yandex에서 개발한 그래디언트 부스팅 계열 모델로, 범주형 변수를 별도 인코딩 없이 내부적으로 처리하는 것이 핵심 강점이다.
- `subscription_type`, `use_frequency`, `last_use_recency` 등 범주형 변수를 `cat_features`로 직접 지정하였다.
- 순서형 부스팅(Ordered Boosting) 방식을 통해 과적합을 억제하면서도 높은 성능을 달성한다.
- **선택 이유:** 이번 데이터에는 의미 있는 범주형 변수가 다수 포함되어 있어, 이를 가장 효과적으로 활용할 수 있는 모델로 판단하였다.

---

## 3. 모델 성능 비교

### 3-1. 성능 지표 비교표

> 평가 기준: 테스트셋 4,000건 (클래스 0: 1,200건, 클래스 1: 2,800건)
> 

| 모델 | Accuracy | ROC-AUC | Precision (0) | Recall (0) | F1 (0) | Precision (1) | Recall (1) | F1 (1) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.6783 | 0.6753 | 0.4688 | 0.5450 | 0.5040 | 0.7904 | 0.7354 | 0.7619 |
| **Random Forest** | **0.7695** | **0.8122** | **0.6273** | 0.5708 | **0.5977** | **0.8229** | **0.8546** | **0.8385** |
| CatBoost | 0.7350 | **0.8184** | 0.5467 | **0.6833** | 0.6074 | 0.8480 | 0.7571 | 0.8000 |

### 3-2. 혼동 행렬 비교

**Logistic Regression**

| 실제 \ 예측 | 해지 후보 예측(0) | 유지 후보 예측(1) |
| --- | --- | --- |
| --- | :---: | :---: |
| 실제 해지 후보(0) | 654 | 546 |
| 실제 유지 후보(1) | 741 | 2,059 |

**Random Forest**

| 실제 \ 예측 | 해지 후보 예측(0) | 유지 후보 예측(1) |
| --- | --- | --- |
| --- | :---: | :---: |
| 실제 해지 후보(0) | 685 | 515 |
| 실제 유지 후보(1) | 407 | 2,393 |

**CatBoost**

| 실제 \ 예측 | 해지 후보 예측(0) | 유지 후보 예측(1) |
| --- | --- | --- |
| --- | :---: | :---: |
| 실제 해지 후보(0) | 820 | 380 |
| 실제 유지 후보(1) | 680 | 2,120 |

---

## 4. 결과 해석 및 모델 비교 분석

### 4-1. 왜 Logistic Regression의 성능이 가장 낮은가?

Logistic Regression은 선형 결정 경계만을 학습하므로, 이번 데이터와 같이 범주형 변수와 파생 변수가 복합적으로 얽힌 구조를 충분히 표현하지 못한다. ROC-AUC가 0.6753으로 다른 모델 대비 약 0.14 낮으며, 클래스 0에 대한 Recall(0.5450)이 현저히 낮아 해지 후보를 절반 가까이 놓치고 있다.

또한 제출 1(mock_data.csv)에서는 성능이 매우 높았던 반면, 이번 데이터에서는 크게 하락하였다. 이는 이전 데이터가 선형 분리가 쉬운 구조였고, `mock_data_2.csv`는 변수 간 상호작용이 더 복잡하게 설계되어 있음을 시사한다. 즉, **선형 모델의 한계가 데이터 복잡도 증가와 함께 명확하게 드러난 결과**이다.

### 4-2. Random Forest vs CatBoost — 무엇을 기준으로 선택해야 하는가?

두 모델은 서로 다른 장단점을 가진다.

**Random Forest**는 전체 Accuracy (0.7695)와 유지 후보(1) F1 (0.8385) 기준에서 가장 우수하다. 실제 유지 후보를 잘못 해지로 분류하는 오류(False Positive, 2행 1열)가 407건으로 가장 적어, **유지 후보를 잘못 해지시키는 리스크를 최소화**하는 데 유리하다.

**CatBoost**는 ROC-AUC (0.8184)와 해지 후보(0) Recall (0.6833)이 가장 높다. 실제 해지 후보를 찾아내는 능력(820건 적중)이 세 모델 중 가장 뛰어나, **불필요한 구독을 빠짐없이 탐지하는 목적**에 더 적합하다.

### 4-3. 핵심 평가 관점으로의 해석

이 프로젝트의 본질적 목적은 **사용자가 불필요하게 지속 결제 중인 구독을 식별**하는 것이다. 따라서 실제 해지 후보를 놓치는 것(False Negative)이 유지 후보를 잘못 해지로 분류하는 것(False Positive)보다 더 큰 손실이다.

이 관점에서 클래스 0 Recall이 가장 높은 **CatBoost가 현재 목적에 더 부합하는 모델**이다. 다만, Accuracy와 유지 후보 분류 안정성 측면에서는 Random Forest가 더 균형 잡혀 있다.

| 평가 목적 | 추천 모델 | 근거 |
| --- | --- | --- |
| 해지 후보를 빠짐없이 탐지 | **CatBoost** | Recall (0) = 0.6833, ROC-AUC = 0.8184 |
| 전반적인 분류 균형 | **Random Forest** | Accuracy = 0.7695, F1 (1) = 0.8385 |
| 해석 가능성 확보 | Logistic Regression | 계수 분석 가능 |

### 4-4. 피처 엔지니어링의 효과

`value_gap`, `has_churn_signal`, `frequency_x_rebuy` 등 도메인 기반 파생 변수들은 단순 원본 변수만으로는 표현하기 어려운 복합적 이탈 패턴을 수치화한다. 특히 `has_churn_signal` (저빈도 + 고비용 + 고부담 복합 플래그)은 규칙 기반 이탈 탐지와 머신러닝 학습을 연결하는 브리지 역할을 한다.

---

## 5. 현재 한계 및 다음 단계

### 5-1. 한계 분석

- **클래스 불균형 미해결:** 해지 후보(0)의 Recall이 세 모델 모두 0.7 미만으로, 여전히 해지 후보를 충분히 탐지하지 못하고 있다. SMOTE, class_weight 조정, 임계값 튜닝 등의 추가 대응이 필요하다.
- **합성 데이터 한계:** 현실에서 발생하는 계절성 이용 패턴, 가족 공유 구독, 프로모션 할인 기간 등의 요소가 반영되지 않아 실제 환경에서의 일반화 성능은 검증되지 않았다.
- **CatBoost Accuracy vs Recall 트레이드오프:** CatBoost는 Recall은 높으나 Precision (0.5467)이 상대적으로 낮아, 실제로는 유지해도 좋은 구독까지 해지 후보로 과분류하는 경향이 있다.
- **연간 구독 처리 데이터 미확보:** `billing_cycle`과 `remaining_months`는 현재 합성 데이터에 포함되어 있지 않아 별도 생성이 필요하다. 보류 처리(`is_deferred = 1`) 이후 `df_active`의 타깃 불균형 비율(현재 유지:해지 ≈ 7:3)이 변화할 수 있으므로, 보류 적용 후 분포를 재확인해야 한다.
- **보류 구독의 재평가 로직 미구현:** 잔여 기간 3개월 초과로 보류된 구독은 갱신 시점에 재평가 대상으로 전환되어야 하나, 현재 파이프라인에는 이 흐름이 포함되지 않았다.
- **할인·환급 데이터 미확보:** `discount_amount`는 현재 합성 데이터에 포함되어 있지 않아 별도 생성이 필요하다. 또한 제휴 카드 혜택은 카드사·구독사 약관에 따라 조건이 달라지며, 혜택 만료 시 실질 비용이 급격히 증가할 수 있으므로 정적인 값이 아닌 시점 종속 변수로 설계해야 한다.
- **`is_zero_cost` 구독의 분류 편향 위험:** 실질 결제 금액이 0인 구독은 비용 기반 해지 신호가 사라져 모델이 활용 피처에만 의존하게 된다. 학습 데이터 내 `is_zero_cost = 1` 비율이 낮을 경우 이 유형의 패턴을 충분히 학습하지 못할 수 있다.

### 5-2. 3차 제출 개선 방향

1. **임계값(Threshold) 조정:** 기본 0.5 기준을 0.35~0.4 수준으로 낮춰 해지 후보 Recall을 의도적으로 높이는 전략을 실험한다.
2. **SMOTE 오버샘플링:** 학습 데이터에서 소수 클래스(0)를 합성 증강하여 불균형 문제를 완화한다.
3. **하이퍼파라미터 튜닝 (CatBoost / RF):** `depth`, `learning_rate`, `n_estimators`, `min_samples_leaf` 등에 대해 GridSearch 또는 Optuna 기반 베이지안 최적화를 수행한다.
4. **피처 중요도 분석:** Random Forest의 feature importance 시각화를 통해 기여도 낮은 변수를 제거하고 모델 단순화를 시도한다.

---

## 결론

이번 2차 제출에서는 데이터를 20,000건으로 확대하고, 연간 구독 처리 방안을 포함한 총 13개의 피처 엔지니어링 변수를 설계한 후, Logistic Regression, Random Forest, CatBoost 세 가지 모델을 비교하였다. Logistic Regression은 데이터 복잡도 증가로 인해 한계를 명확히 드러냈으며, Random Forest와 CatBoost는 비선형 학습 능력을 통해 유의미한 성능 향상을 달성하였다.

연간 구독에 대해서는 두 가지 처리 방안을 도입하였다. 첫째, 연간 구독의 `monthly_cost`를 `annual_cost / 12`로 정규화하여 모든 비용 기반 피처의 기준을 월 단위로 통일하였다. 둘째, 잔여 기간이 3개월을 초과하는 연간 구독은 즉시 해지 판단이 실용적이지 않으므로 예측 대상에서 제외하고 갱신 시점에 재평가하는 보류 로직을 설계하였다.

또한 제휴 카드 할인·전액 환급 등으로 실제 지출이 발생하지 않는 구독에 대해서는, 명목 청구 금액에서 할인액을 차감한 `effective_monthly_cost`를 도입하였다. 이를 통해 모든 비용 파생 변수가 실질 부담 기준으로 산출되며, `is_zero_cost` 플래그로 비용 동기가 사라진 구독을 별도로 식별하여 활용 기반 피처 위주의 판단이 가능하도록 설계하였다.

본 프로젝트의 핵심 목적인 해지 후보 탐지 관점에서는 CatBoost (Recall: 0.6833, ROC-AUC: 0.8184)가 현재 단계에서 가장 적합한 모델로 판단된다. 다음 단계에서는 연간 구독 변수를 실제 합성 데이터에 반영하고, 임계값 조정과 하이퍼파라미터 최적화를 통해 이탈 탐지 성능을 추가적으로 개선할 계획이다.