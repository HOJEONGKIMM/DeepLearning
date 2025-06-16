# DeepLearning

# ex01_유방암_데이터_분류(이진분류)

## 사용한 주요 라이브러리
- **NumPy**: 수치 계산용 라이브러리. 시계열 데이터를 다룰 때 필수.
- **Pandas**: 데이터프레임 형태의 데이터 조작.
- **Matplotlib / Seaborn**: 시각화 라이브러리.
- **TensorFlow / Keras**: 딥러닝 모델 구축 및 학습.
- **Scikit-learn**: 데이터 분할(train/test), 정규화, 평가 지표 계산 등에 사용.

---

## 주요 모델 및 아키텍처

| 모델명       | 설명 |
|--------------|------|
| **LSTM**     | 장기 의존성을 처리하는 순환 신경망(RNN) 구조 |
| **GRU**      | LSTM보다 구조가 간단하면서도 성능은 유사 |
| **1D CNN**   | Conv1D 기반의 모델. 특징 추출에 강함 |
| **BiLSTM**   | 양방향 LSTM. 과거와 미래 시점을 동시에 반영 |
| **CNN-LSTM** | CNN으로 특성 추출 → LSTM으로 시계열 처리 |
| **Transformer** | self-attention 기반. 최근 시계열에서도 각광받음 |

---

## 데이터 전처리
- **정규화**: `MinMaxScaler`로 0~1 범위로 스케일링
- **슬라이딩 윈도우**: 시계열 윈도우 구성으로 `x`와 `y` 생성
- **데이터 분할**: 학습 / 검증 / 테스트 데이터셋 분리

---

## 모델 학습 예시 코드
```python
from keras.models import Sequential
from keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(units=64, input_shape=(timesteps, features)))
model.add(Dense(1))  # 출력층
model.compile(optimizer='adam', loss='mse')
```

---

## 평가 지표
- **RMSE (Root Mean Squared Error)**: 예측의 정확도 측정
- **R² Score**: 결정계수, 회귀 적합도 평가

---

## 정리
- 다양한 딥러닝 기반 시계열 예측 모델을 비교 실험
- 전처리부터 모델 학습, 평가까지 일관된 구조
- 시계열에 적합한 모델 선택에 도움을 주는 실험 설계


---


# ex04_개_고양이_분류하기

## 실험 환경 및 공통 하이퍼파라미터 설정

### 데이터 전처리
- 데이터 정규화: `MinMaxScaler` 사용
- 윈도우 크기: `window_size = 30` (과거 30일 데이터를 사용)
- 타겟: 종가(Close)
- 데이터 분할: 학습 70% / 검증 20% / 테스트 10%

---

## 공통 하이퍼파라미터
| 항목         | 값           |
|--------------|--------------|
| Epoch 수     | 100          |
| Batch Size   | 32           |
| Optimizer    | Adam         |
| Learning Rate | 0.001       |
| 손실 함수     | MSE          |
| 평가지표      | RMSE         |

---

## 모델별 구성 요약

### 1. LSTM
- `LSTM(64) → Dense(1)`
- 장기 의존성에 강함

### 2. GRU
- `GRU(64) → Dense(1)`
- 계산 효율성 우수

### 3. 1D CNN
- `Conv1D(64, kernel_size=3) → MaxPooling1D → Flatten → Dense(50) → Dense(1)`
- 패턴 감지 능력 우수

### 4. Bi-LSTM
- `Bidirectional(LSTM(64)) → Dense(1)`
- 과거 + 미래 정보 활용

### 5. CNN-LSTM
- `TimeDistributed(Conv1D(...)) → LSTM(...)`
- 시계열 + 패턴 특징 결합

### 6. Transformer
- `MultiHeadAttention + PositionalEncoding` 등 직접 구현
- 최신 Self-Attention 기반

---

## 요약
- 동일한 구조, 동일한 목적(종가 예측)으로 모델 훈련
- 하이퍼파라미터 통일로 공정한 성능 비교
- 다양한 모델의 예측 성능(RMSE) 비교 실험
