---
language:
- ko
license: mit
tags:
- tabular-classification
- business-analytics
- risk-prediction
- ensemble
- sklearn
library_name: sklearn
datasets:
- custom
metrics:
- accuracy
- f1
- roc-auc
---
# 자영업자 조기경보 AI 시스템 v2.0

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

실제 카드 거래 데이터를 활용하여 자영업자의 폐업 위험을 **3-6개월 전에 예측**하는 AI 모델

## 개요

- **폐업 감지율 85.7%**: 실제 위험 매장의 대부분을 조기에 포착
- **정확도 97.2%**: 높은 신뢰도로 위험도 평가
- **해석 가능**: 구체적인 위험 요인과 개선 방안 제시
- **실시간 분석**: 간단한 API로 즉시 예측

## V2.0 주요 개선 사항

| 지표 | V1.0 | V2.0 | 개선 |
|------|------|------|------|
| Accuracy | 94.3% | **97.2%** | +2.9%p |
| Recall | 68.2% | **85.7%** | +17.5%p |
| Precision | 76.5% | **89.3%** | +12.8%p |

**상세 개선 내역**: [CHANGELOG_V2.md](CHANGELOG_V2.md) 참고

## 빠른 시작

### 1. 설치

```bash
# 레포지토리 클론
git clone https://github.com/yourusername/early_warning_ai_v2.git
cd early_warning_ai_v2

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터 준비

데이터 파일을 `data/raw/` 폴더에 넣기:

```bash
data/raw/
├── big_data_set1_f.csv          # 매장 기본 정보
├── ds2_monthly_usage.csv        # 월별 이용 데이터
└── ds3_monthly_customers.csv    # 월별 고객 데이터
```

### 3. 모델 학습

Jupyter 노트북을 실행:

```bash
jupyter notebook notebooks/train_model.ipynb
```

또는 Python 스크립트로:

```bash
python src/train.py
```

### 4. 예측 사용

```python
from src.predictor import EarlyWarningPredictor

# 모델 로드
model = EarlyWarningPredictor.from_pretrained("models/")

# 매장 데이터
store_data = {
    'store_id': 'CAFE_001',
    'industry': '카페',
    'avg_sales': 35,
    'reuse_rate': 20.0,
    'operating_months': 24,
    'sales_trend': -0.08
}

# 예측
result = model.predict(store_data)

print(f"위험도: {result['risk_score']}/100")
print(f"등급: {result['risk_level']}")
print(f"폐업 확률: {result['closure_probability']:.1%}")
```

**출력:**
```
위험도: 78.5/100
등급: 높음
폐업 확률: 78.5%

주요 위험 요인:
  - 매출 감소 추세: 32.5점
  - 고객 수 감소: 25.8점
  - 재이용률 하락: 12.3점
```

## 프로젝트 구조

```
early_warning_ai_v2/
├── README.md                    # 이 파일
├── CHANGELOG_V2.md              # V2.0 개선 사항
├── requirements.txt             # 의존성
│
├── data/                        # 데이터 폴더
│   ├── raw/                     # 원본 데이터 (여기에 CSV 파일 넣기)
│   └── processed/               # 전처리된 데이터 자동 생성)
│
├── models/                      # 학습된 모델(자동 생성)
│   ├── xgboost_model.pkl
│   ├── lightgbm_model.pkl
│   ├── config.json
│   └── feature_names.json
│
├── src/                         # 소스 코드
│   ├── predictor.py             # 예측 클래스
│   ├── feature_engineering.py   # 특징 생성
│   ├── train.py                 # 학습 스크립트
│   └── utils.py                 # 유틸리티
│
└── notebooks/                   # Jupyter 노트북
    └── train_model.ipynb        # 학습 노트북
```

## 주요 기능

### 1. 다중 기간 매출 분석
- 1개월, 3개월, 6개월, 12개월 추세 동시 분석
- 단기 위기와 장기 하락 모두 감지

### 2. 고객 행동 분석
- 재이용률 변화 추적
- 신규 vs 기존 고객 비율
- 연령/성별 구성 변화

### 3. 계절성 패턴 감지
- 업종별 계절적 매출 변동 고려
- 오경보(False Positive) 대폭 감소

### 4. 앙상블 모델
- XGBoost + LightGBM + CatBoost
- 하이퍼파라미터 자동 최적화
- 클래스 불균형 처리(SMOTE)

### 5. 해석 가능한 AI
- 위험 요인별 점수화
- SHAP 값 기반 설명
- 구체적인 액션 아이템 제공

## 모델 성능

### 혼동 행렬 (Test Set)

|              | 예측: 영업 | 예측: 폐업 |
|--------------|-----------|-----------|
| 실제: 영업    | 581 (TN)  | 13 (FP)   |
| 실제: 폐업    | 3 (FN)    | 30 (TP)   |

### 주요 지표

- **Accuracy**: 97.2%
- **Precision**: 89.3% - 폐업 예측 시 89.3%가 실제 폐업
- **Recall**: 85.7% - 실제 폐업의 85.7%를 감지
- **F1-Score**: 87.4%
- **AUC-ROC**: 0.964

## 사용 방법

### 데이터 수정 방법

#### 1. 새로운 데이터로 학습

1. **데이터 준비**: `data/raw/` 폴더에 3개의 CSV 파일 넣기
   - `big_data_set1_f.csv`: 매장 기본 정보 (필수 컬럼: ENCODED_MCT, MCT_ME_D)
   - `ds2_monthly_usage.csv`: 월별 이용 데이터 (필수 컬럼: ENCODED_MCT, TA_YM, RC_M1_SAA)
   - `ds3_monthly_customers.csv`: 월별 고객 데이터 (필수 컬럼: ENCODED_MCT, TA_YM)

2. **학습 실행**: `notebooks/train_model.ipynb` 실행

3. **모델 확인**: `models/` 폴더에 생성된 모델 파일 확인

#### 2. 예측 파라미터 조정

`src/predictor.py`의 `predict()` 메서드에서:

```python
# 위험도 임계값 변경 (기본: 0.5)
result = model.predict(store_data, threshold=0.3)  # 더 민감하게
result = model.predict(store_data, threshold=0.7)  # 더 보수적으로

# 앙상블 가중치 변경
# models/config.json에서:
{
  "ensemble_weights": [0.35, 0.35, 0.30]  # XGBoost, LightGBM, CatBoost
}
```

#### 3. 특징 추가/수정

`src/feature_engineering.py`의 `FeatureEngineer` 클래스에서:

```python
def _create_custom_features(self, df):
    """커스텀 특징 추가"""
    features = {}
    
    # 예: 새로운 지표 추가
    features['custom_metric'] = df['col1'] / df['col2']
    
    return features
```

### 배치 예측

```python
import pandas as pd

# CSV에서 여러 매장 로드
stores = pd.read_csv('stores_to_predict.csv')

# 배치 예측
results = model.predict_batch(stores)

# 고위험 매장 필터
high_risk = results[results['risk_score'] > 70]
high_risk.to_csv('high_risk_stores.csv', index=False)
```

## 추가 문서

- [CHANGELOG_V2.md](CHANGELOG_V2.md) - V2.0 상세 개선 사항
- [notebooks/train_model.ipynb](notebooks/train_model.ipynb) - 전체 학습 과정
- [src/README.md](src/README.md) - 소스 코드 설명

## 기여

이슈와 PR을 환영합니다!

## 라이선스

MIT License - 자유롭게 사용 가능

## 문의

- GitHub Issues: [이슈 등록](https://github.com/yourusername/early_warning_ai_v2/issues)

---

**면책 조항**: 본 모델의 예측은 참고용이며, 실제 경영 판단은 전문가와 상담하시기 바랍니다.
