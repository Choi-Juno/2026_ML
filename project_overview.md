# 피처 엔지니어링 & 모델 비교 프로젝트 개요

이 문서는 사용자 관점에서 "이 구독을 계속 유지할 필요가 있을까?"를 판단하기 위한 churn 분류 프로젝트의 핵심 내용을 한눈에 보기 쉽게 정리한 요약본이다.

---

## 1. 문제 정의서

### 해결할 문제
사용자는 여러 구독 서비스를 동시에 이용하지만, 실제로는 거의 사용하지 않는 구독도 자동 결제로 계속 유지하는 경우가 많다.  
따라서 이용 패턴과 비용 대비 효율을 바탕으로, 현재 유지 중인 구독이 정말 필요한지 판단할 수 있는 기준이 필요하다.

### 예측 목표
- 각 구독을 `해지 후보`와 `유지 후보`로 이진 분류
- `0`: 해지 후보, 유지 필요성이 낮은 구독
- `1`: 유지 후보, 계속 유지할 가능성이 높은 구독
- 사용자가 구독 정리 우선순위를 정할 수 있도록 지원

## 2. 주요 변수

### 원본 변수

| 변수명 | 유형 | 의미 |
|---|---|---|
| `subscription_type` | 명목형 | 구독 서비스 종류 (Education, Music, Fitness 등 6종) |
| `monthly_cost` | 연속형 | 월 구독료 |
| `use_frequency` | 순서형 | 이용 빈도 (rare / monthly / weekly / frequent) |
| `last_use_recency` | 순서형 | 마지막 이용 시점 (>30d / 7-30d / 1-7d / <1d) |
| `perceived_necessity` | 순서형 | 체감 필요도 (1–5) |
| `cost_burden` | 순서형 | 비용 부담감 (1–5) |
| `would_rebuy` | 순서형 | 재구독 의향 (1–5) |
| `replacement_available` | 이진형 | 대체 서비스 존재 여부 |

### 파생 변수 (피처 엔지니어링)

| 변수명 | 산출식 | 의미 |
|---|---|---|
| `use_frequency_score` | 빈도 → 1–4 정수 매핑 | 이용 빈도 수치화 |
| `last_use_recency_score` | 최근성 → 1–4 정수 매핑 | 최근 이용 시점 수치화 |
| `value_gap` | `freq_score + recency_score + necessity - cost_burden` | 사용 가치 대비 비용 부담 |
| `rebuy_satisfaction_gap` | `monthly_cost - perceived_necessity` | 비용 대비 만족 간극 |
| `cost_to_necessity_ratio` | `monthly_cost × (necessity + ε)` | 필요도 가중 비용 |
| `log_monthly_cost` | `log1p(monthly_cost)` | 비용 분포 정규화 |
| `cost_burden_x_replacement` | `cost_burden × replacement_available` | 대체재 있을 때의 비용 부담 |
| `necessity_x_recency` | `perceived_necessity × recency_score` | 필요도·최근성 교호 효과 |
| `frequency_x_rebuy` | `freq_score × would_rebuy` | 이용 빈도·재구독 의향 교호 효과 |
| `is_high_cost` | 월비용 상위 25% 여부 | 고비용 구독 플래그 |
| `has_churn_signal` | 빈도 ≤2 & 최근성 ≤2 & 부담 ≥4 | 규칙 기반 이탈 신호 |

## 3. 데이터 & 실험 설정

- 데이터: `mock_data_2.csv`, 총 `20,000`건
- 타깃 분포: 유지(1) `14,000` / 해지(0) `6,000` (약 7:3 불균형)
- 분할: 학습 `80%` / 테스트 `20%` (stratify 적용)
- 모델: `Logistic Regression` / `Random Forest` / `CatBoost` 비교

## 4. 성능 지표 비교

| 모델 | Accuracy | ROC-AUC | Precision (0) | Recall (0) | F1 (0) |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | `0.6783` | `0.6753` | `0.4688` | `0.5450` | `0.5040` |
| Random Forest | `0.7695` | `0.8122` | `0.6273` | `0.5708` | `0.5977` |
| CatBoost | `0.7350` | `0.8184` | `0.5467` | `0.6833` | `0.6074` |

- **Random Forest**는 전반적 정확도와 F1 균형이 가장 우수하다.
- **CatBoost**는 ROC-AUC와 해지 후보 Recall이 가장 높아 실제 해지 후보를 놓칠 확률이 가장 낮다.
- **Logistic Regression**은 두 모델 대비 성능이 낮으나, 선형 기준선으로서의 의미를 유지한다.

## 5. 현재 한계 정리

1. 실제 사용자 로그가 아닌 합성 데이터 기반 결과다.
2. 해지 후보(클래스 0) Recall이 최대 0.68 수준에 머물러 실사용 기준으로는 추가 개선이 필요하다.
3. 만족도, 번들 할인, 가족 공유 여부 등 실제 유지 판단 요인이 아직 포함되지 않았다.
4. 하이퍼파라미터 튜닝 및 임계값 최적화가 수행되지 않았다.
