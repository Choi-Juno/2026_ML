# 제출 1. 초기 설계 및 1차 실행 보고서

## 프로젝트 개요
본 보고서는 사용자의 서비스 이용 패턴을 바탕으로 다음 달 구독 유지 여부를 예측하는 초기 머신러닝 모델을 설계하고, 1차 실행 결과를 정리한 문서이다.  
예측 대상은 이진 분류 문제이며, `0`은 해지(Churn), `1`은 유지(Retain)를 의미한다.

---

## 1. 문제 정의서

### 해결할 문제
구독 서비스에서는 사용자의 이탈 가능성을 사전에 예측하는 것이 중요하다.  
해지 가능성이 높은 사용자를 조기에 식별할 수 있다면, 할인 쿠폰 제공, 맞춤형 리마인드, 프로모션 제안 등과 같은 이탈 방어 전략을 선제적으로 수행할 수 있다.

### 예측 목표
- 다음 달 구독 유지 여부를 예측하는 이진 분류 모델 구축
- 클래스 `0(해지)` 고객을 우선적으로 잘 찾아내는 모델 확보
- 향후 마케팅 방어 전략의 기초 데이터로 활용

### 비즈니스 관점의 핵심
본 과제에서 중요한 것은 단순 정확도보다 실제 해지 고객을 놓치지 않고 식별하는 능력이다.  
따라서 `Recall`과 `F1-Score`를 함께 해석하며, 해지 고객 선별 관점에서 모델 성능을 평가한다.

---

## 2. 데이터 설명

### 데이터 개요
- 데이터 파일: `mock_data.csv`
- 데이터 크기: `5,000`건
- 타깃 분포: 유지 `3,971`, 해지 `1,029`
- 데이터 성격: 실제 서비스 로그가 아닌, 초기 모델 검증을 위한 합성 데이터

### 주요 변수
| 변수명 | 설명 | 해석 포인트 |
|---|---|---|
| `Monthly_Fee` | 월 구독료 | 해지군에서 상대적으로 높은 경향 |
| `Access_Days` | 월 접속 일수 | 해지군에서 낮은 경향 |
| `Usage_Time` | 월 누적 이용 시간 | 해지군에서 짧은 경향 |
| `Content_Count` | 콘텐츠 소비량 | 이용도와 충성도 판단 보조 |
| `Customer_Inquiry` | 고객센터 문의 횟수 | 해지군에서 높은 경향 |
| `Target` | 예측 대상 | `0=해지`, `1=유지` |

### 파생 변수 생성
모델이 원본 변수만으로 판단하지 않도록, 이용 행태를 더 분명하게 드러내는 파생 변수를 추가하였다.

- `Inquiry_Rate = Customer_Inquiry / (Access_Days + 1)`
- `Value_Score = Usage_Time / Monthly_Fee`

### 데이터 준비 코드
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

df = pd.read_csv("mock_data.csv")

df["Inquiry_Rate"] = df["Customer_Inquiry"] / (df["Access_Days"] + 1)
df["Value_Score"] = df["Usage_Time"] / df["Monthly_Fee"]

display(df.head())
print("데이터 크기:", df.shape)
print(df["Target"].value_counts())
```

---

## 3. 기본 모델 실행

### 모델 선택
초기 기준 모델로 `Logistic Regression`을 사용하였다.  
로지스틱 회귀는 구조가 단순하고 해석이 용이하여, 초기 성능 확인과 기준선 설정에 적합한 모델이다.

### 전처리 및 학습 절차
1. `Target`을 제외한 나머지 변수를 입력값 `X`로 구성
2. `train_test_split(..., test_size=0.2, stratify=y)`로 학습/평가 데이터 분리
3. `StandardScaler`로 수치형 변수 스케일 정규화
4. 로지스틱 회귀 모델 학습
5. 테스트셋 기준 예측값과 예측 확률 산출

### 모델 실행 코드
```python
X = df.drop("Target", axis=1)
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]
```

### 실행 결과 요약
- 테스트 데이터는 총 `1,000`건으로 평가
- 혼동 행렬 기준 오분류는 `2건`만 발생
- 기준 모델임에도 매우 높은 분류 성능이 확인됨

---

## 4. 성능 지표 해석

### 평가 코드
```python
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=0)
recall = recall_score(y_test, y_pred, pos_label=0)
f1 = f1_score(y_test, y_pred, pos_label=0)
roc_auc = roc_auc_score(y_test, y_proba)

print("=== 로지스틱 회귀 초기 모델 평가 결과 ===")
print(f"정확도 (Accuracy)  : {accuracy:.4f}")
print(f"정밀도 (Precision) (클래스 0 기준) : {precision:.4f}")
print(f"재현율 (Recall)    (클래스 0 기준) : {recall:.4f}")
print(f"F1-점수 (F1-Score) (클래스 0 기준) : {f1:.4f}")
print(f"ROC-AUC 점수      : {roc_auc:.4f}")

print("\n[분류 보고서 (Classification Report)]")
print(classification_report(y_test, y_pred))
```

### 평가 결과
| 지표 | 값 | 해석 |
|---|---:|---|
| Accuracy | `0.9980` | 전체 예측의 정답 비율이 매우 높음 |
| Precision (`0` 기준) | `0.9951` | 해지로 예측한 고객 대부분이 실제 해지 고객 |
| Recall (`0` 기준) | `0.9951` | 실제 해지 고객을 거의 놓치지 않음 |
| F1-Score (`0` 기준) | `0.9951` | 정밀도와 재현율이 균형 있게 높음 |
| ROC-AUC | `0.99997` | 두 클래스를 거의 완벽하게 분리 |

### ROC-AUC 해석 보완
노트북 출력에는 `ROC-AUC 점수 : 1.0000`으로 표시되지만, 이는 소수 넷째 자리에서 반올림된 결과이다.  
실제 계산값은 약 `0.99997`이며, 완전히 `1.0`인 것은 아니다.

### 혼동 행렬 해석
실제 테스트 결과의 혼동 행렬은 다음과 같습니다.

| 실제 \\ 예측 | 해지 예측(0) | 유지 예측(1) |
|---|---:|---:|
| 실제 해지(0) | 205 | 1 |
| 실제 유지(1) | 1 | 793 |

해지 고객 206명 중 205명을 정확히 분류하였고, 유지 고객 794명 중 793명을 정확히 분류하였다.  
즉, 초기 기준 모델임에도 해지 탐지 성능이 매우 우수하게 나타났다.

### 성능이 지나치게 높은 이유
현재 사용한 데이터는 실제 운영 데이터가 아닌 합성 데이터이므로, 변수 간 경향이 비교적 뚜렷하게 설계되어 있다.  
예를 들어 해지 고객은 접속 일수와 사용 시간이 낮고, 문의 횟수는 높게 분포하므로 모델이 클래스 경계를 쉽게 학습할 수 있다.  
따라서 본 결과는 모델의 성능뿐 아니라 데이터의 단순성도 크게 반영한 결과로 해석할 필요가 있다.

---

## 5. 현재 한계 정리

### 한계 1. 합성 데이터 기반 평가
현재 사용한 `mock_data.csv`는 실제 서비스 로그가 아니라 규칙성이 강한 합성 데이터이다.  
따라서 현실 데이터에서 발생할 수 있는 결측치, 이상치, 예외 행동, 비정형 패턴이 충분히 반영되지 않는다.

### 한계 2. 기준 모델 수준의 실험
이번 단계는 초기 기준선을 확인하기 위한 1차 실행이다.  
로지스틱 회귀만으로는 변수 간 비선형 관계나 복잡한 상호작용을 충분히 반영하기 어렵다.

### 한계 3. 클래스 불균형 존재
유지 고객이 약 80%, 해지 고객이 약 20%인 불균형 데이터이므로, 정확도만으로 모델 성능을 판단할 경우 해석상 착시가 발생할 수 있다.  
이 때문에 해지 클래스 기준 `Recall`, `Precision`, `F1-Score`를 함께 확인하였다.

### 향후 개선 방향
1. `Decision Tree`, `Random Forest`, `XGBoost` 등 비선형 모델과 성능 비교
2. 교차 검증 기반 하이퍼파라미터 튜닝 수행
3. 실제 서비스 로그 또는 더 현실적인 시뮬레이션 데이터 확보
4. ROC 곡선, PR 곡선, 피처 중요도까지 포함한 심화 분석 진행

---

## 결론
이번 1차 실행에서는 구독 해지 예측 문제를 명확히 정의하고, 합성 데이터를 기반으로 로지스틱 회귀 기준 모델을 구축하였다.  
해당 모델은 높은 정확도와 재현율을 보였으며, 특히 해지 고객 탐지 성능이 매우 우수하게 나타났다.

다만 현재 결과는 합성 데이터의 단순성이 크게 반영된 값이므로, 다음 단계에서는 더 현실적인 데이터와 다양한 모델을 비교하여 일반화 성능을 검증할 필요가 있다.
