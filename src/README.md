# 소스 코드 설명

## 파일 구조

```
src/
├── predictor.py              # 예측 클래스
├── feature_engineering.py    # 특징 생성
├── train.py                  # 학습 스크립트
└── README.md                 # 이 파일
```

---

## 각 파일 설명

### 1. `predictor.py` - 예측 클래스

**용도**: 학습된 모델을 로드하고 예측을 수행하는 메인 클래스

**주요 클래스**: `EarlyWarningPredictor`

**주요 메서드**:

```python
# 모델 로드 (허깅페이스 스타일)
model = EarlyWarningPredictor.from_pretrained("models/")

# 단일 예측
result = model.predict(store_data)

# 배치 예측
results = model.predict_batch(stores_df)

# 예측 설명
explanation = model.explain(store_data)

# 모델 정보
info = model.get_model_info()
```

**반환 값**:
```python
{
    'risk_score': 78.5,           # 0-100점 위험도
    'risk_level': '높음',          # 낮음/보통/높음
    'closure_probability': 0.785, # 폐업 확률
    'risk_factors': {...},        # 위험 요인별 점수
    'action_items': [...]         # 권장 조치
}
```

**수정 방법**:

```python
# 1. 위험도 임계값 변경
def predict(self, store_data, threshold=0.5):  # 기본값 변경
    ...

# 2. 앙상블 가중치 조정
# models/config.json 파일에서:
{
    "ensemble_weights": [0.6, 0.4]  # XGBoost 60%, LightGBM 40%
}

# 3. 위험 등급 기준 변경
if risk_score < 40:  # 기존 30에서 40으로
    risk_level = '낮음'
```

---

### 2. `feature_engineering.py` - 특징 생성

**용도**: 원본 데이터에서 47개의 특징을 자동으로 생성

**주요 클래스**: `FeatureEngineer`

**생성되는 특징**:

#### 매출 관련 (15개)
- `sales_avg_1m`, `sales_avg_3m`, `sales_avg_6m`, `sales_avg_12m`
- `sales_recent_vs_previous`, `sales_mom_change`, `sales_yoy_change`
- `sales_max`, `sales_min`, `sales_range`

#### 고객 관련 (12개)
- `customer_reuse_rate`, `customer_reuse_trend`
- `customer_new_rate`
- 연령/성별별 고객 비율 (10개)

#### 운영 관련 (8개)
- `operation_months`, `operation_avg_amount`
- `operation_cancel_rate`, `operation_delivery_rate`

#### 트렌드 (5개)
- `trend_slope`, `trend_r2`, `trend_direction`
- `trend_consecutive_down`, `trend_consecutive_up`

#### 변동성 (4개)
- `volatility_cv`, `volatility_std`, `volatility_mad`, `volatility_recent_std`

#### 계절성 (2개)
- `seasonality_detected`, `seasonality_strength`

#### 맥락 (1개)
- `context_industry`

**사용 예시**:

```python
from feature_engineering import FeatureEngineer

engineer = FeatureEngineer()

features = engineer.create_features(
    store_data={'industry': '카페', 'location': '서울'},
    monthly_usage=usage_df,
    monthly_customers=customer_df
)
```

**새로운 특징 추가 방법**:

```python
class FeatureEngineer:
    def _create_custom_features(self, df):
        """커스텀 특징 추가"""
        features = {}
        
        # 예: 성장률 지표
        if 'RC_M1_SAA' in df.columns and len(df) >= 6:
            recent_3m = df['RC_M1_SAA'].tail(3).mean()
            past_3m = df['RC_M1_SAA'].head(3).mean()
            features['growth_rate'] = (recent_3m / past_3m - 1) * 100
        
        return features
    
    def create_features(self, store_data, monthly_usage, monthly_customers):
        features = {}
        
        # 기존 특징들...
        features.update(self._create_sales_features(monthly_usage))
        features.update(self._create_customer_features(monthly_customers))
        
        # 새로운 커스텀 특징 추가
        features.update(self._create_custom_features(monthly_usage))
        
        return pd.DataFrame([features])
```

---

### 3. `train.py` - 학습 스크립트

**용도**: 커맨드라인에서 모델을 학습하는 스크립트

**사용법**:

```bash
# 기본 사용
python src/train.py

# 옵션 지정
python src/train.py --data data/raw --output models/ --max-stores 1000

# 도움말
python src/train.py --help
```

**파라미터**:
- `--data`: 데이터 디렉토리 경로 (기본: `data/raw`)
- `--output`: 모델 저장 경로 (기본: `models`)
- `--max-stores`: 테스트용 최대 매장 수 (선택사항)

**주요 함수**:

```python
def load_data(data_dir)
    """데이터 로드"""

def create_features(df_store, df_usage, df_customer)
    """특징 생성"""

def preprocess_data(X, y)
    """전처리 및 분할"""

def apply_smote(X_train, y_train)
    """SMOTE 적용"""

def train_models(X_train, y_train)
    """모델 학습"""

def evaluate_models(xgb_model, lgb_model, X_test, y_test)
    """평가"""

def save_models(...)
    """모델 저장"""
```

**수정 방법**:

```python
# 1. 모델 하이퍼파라미터 변경
def train_models(X_train, y_train):
    xgb_model = xgb.XGBClassifier(
        max_depth=8,           # 6에서 8로 증가
        learning_rate=0.05,    # 0.1에서 0.05로 감소
        n_estimators=300,      # 200에서 300으로 증가
        # ...
    )

# 2. 앙상블 가중치 변경
def evaluate_models(...):
    ensemble_pred = 0.6 * xgb_pred + 0.4 * lgb_pred  # 기존 0.5, 0.5

# 3. 데이터 분할 비율 변경
def preprocess_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, ...  # 0.25에서 0.2로
    )
```

---

## 주요 수정 시나리오

### 시나리오 1: 새로운 데이터로 학습

**1단계**: 데이터 준비
```bash
# data/raw/에 CSV 파일 3개 배치
data/raw/
├── big_data_set1_f.csv
├── ds2_monthly_usage.csv
└── ds3_monthly_customers.csv
```

**2단계**: 학습 실행
```bash
python src/train.py
```

**3단계**: 예측 사용
```python
from src.predictor import EarlyWarningPredictor
model = EarlyWarningPredictor.from_pretrained("models/")
```

### 시나리오 2: 모델 성능 개선

**방법 1**: 특징 추가
```python
# feature_engineering.py에 새로운 특징 추가
def _create_custom_features(self, df):
    # 새로운 지표 계산
    pass
```

**방법 2**: 하이퍼파라미터 튜닝
```python
# train.py에서 파라미터 조정
xgb_model = xgb.XGBClassifier(
    max_depth=8,
    learning_rate=0.05,
    ...
)
```

**방법 3**: 앙상블 가중치 조정
```python
# models/config.json 수정
{
    "ensemble_weights": [0.6, 0.4]
}
```

### 시나리오 3: 예측 임계값 조정

**더 민감하게 (조기 경보 강화)**:
```python
result = model.predict(store_data, threshold=0.3)
# 폐업 확률 30% 이상이면 위험으로 판단
```

**더 보수적으로**:
```python
result = model.predict(store_data, threshold=0.7)
# 폐업 확률 70% 이상이어야 위험으로 판단
```

---

## 참고 자료

- XGBoost 문서: https://xgboost.readthedocs.io/
- LightGBM 문서: https://lightgbm.readthedocs.io/
- SMOTE 설명: https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
