# 자영업 조기경보 AI v2.0

## 프로젝트 구조

```
early_warning_ai_v2/
│
├── README.md                    # 메인 가이드
├── CHANGELOG_V2.md              # V2.0 개선 사항
├── requirements.txt             # 의존성
├── LICENSE                      # ⚖MIT 라이선스
├── .gitignore                   # Git 제외 파일
│
├── data/                        # 데이터 폴더
│   ├── README.md                # 데이터 준비 가이드
│   ├── raw/                     # 여기에 CSV 파일 넣기
│   │   └── .gitkeep
│   └── processed/               # (자동 생성)
│
├── models/                      # 학습된 모델 (자동 생성)
│
├── src/                         # 소스 코드
│   ├── README.md                # 코드 설명
│   ├── predictor.py             # 예측 클래스 (허깅페이스 스타일)
│   ├── feature_engineering.py   # 특징 생성 (47개)
│   └── train.py                 # 학습 스크립트
│
└── notebooks/                   # Jupyter 노트북
    └── train_model.ipynb        # 전체 학습 과정
```

---

## 주요 특징

### 1. 깔끔한 구조
- **필수 파일만 포함**: 실제로 필요한 코드와 문서만
- **명확한 디렉토리**: 각 폴더의 용도가 분명함
- **상세한 가이드**: 모든 폴더에 README.md 포함

### 2. 실용적인 설계
- **데이터 분리**: `data/raw/`에 CSV만 넣으면 됨
- **모듈화**: 각 기능이 독립적인 파일로 분리
- **확장 가능**: 새로운 특징이나 모델 추가 쉬움

### 3. 완벽한 문서화
- **README.md**: 전체 프로젝트 개요
- **CHANGELOG_V2.md**: V2.0 상세 개선 사항
- **src/README.md**: 소스 코드 설명 및 수정 방법
- **data/README.md**: 데이터 준비 가이드

---

## 빠른 시작

### 1. 설치
```bash
cd early_warning_ai_v2
pip install -r requirements.txt
```

### 2. 데이터 준비
`data/raw/` 폴더에 3개의 CSV 파일 넣기:
- `big_data_set1_f.csv`
- `ds2_monthly_usage.csv`
- `ds3_monthly_customers.csv`

### 3. 학습
```bash
# Jupyter 노트북으로
jupyter notebook notebooks/train_model.ipynb

# 또는 스크립트로
python src/train.py
```

### 4. 예측
```python
from src.predictor import EarlyWarningPredictor

model = EarlyWarningPredictor.from_pretrained("models/")
result = model.predict(store_data)
```

---

## 주요 파일 설명

### README.md
- 프로젝트 전체 개요
- 빠른 시작 가이드
- 사용 방법
- 프로젝트 구조

### CHANGELOG_V2.md
- V1.0 → V2.0 모든 개선 사항
- 성능 비교표
- 실제 개선 사례
- 구조 변경 내역

### src/predictor.py
- 허깅페이스 스타일 API
- `from_pretrained()` 메서드
- 단일/배치 예측
- 위험 요인 분석

### src/feature_engineering.py
- 47개 특징 자동 생성
- 매출, 고객, 운영, 트렌드, 변동성, 계절성
- 확장 가능한 설계

### src/train.py
- 전체 학습 파이프라인
- 커맨드라인 인터페이스
- SMOTE 클래스 불균형 처리
- 자동 평가 및 저장

### notebooks/train_model.ipynb
- 전체 학습 과정 시각화
- EDA (탐색적 데이터 분석)
- 단계별 설명
- 성능 평가 및 분석

---

## 데이터 수정 방법

### 새로운 데이터로 학습

**1단계**: `data/raw/`에 CSV 파일 3개 배치

**2단계**: 학습 실행
```bash
python src/train.py
```

**3단계**: 생성된 모델 확인
```bash
ls models/
# xgboost_model.pkl, lightgbm_model.pkl, config.json 등
```

### 파라미터 조정

#### 예측 임계값 변경
```python
# src/predictor.py의 predict() 메서드에서
result = model.predict(store_data, threshold=0.3)  # 더 민감하게
```

#### 앙상블 가중치 변경
```json
// models/config.json에서
{
    "ensemble_weights": [0.6, 0.4]  // XGBoost 60%, LightGBM 40%
}
```

#### 특징 추가
```python
# src/feature_engineering.py의 FeatureEngineer 클래스에
def _create_custom_features(self, df):
    features = {}
    # 새로운 특징 추가
    features['new_metric'] = df['col1'] / df['col2']
    return features
```

---

## V2.0 핵심 개선

### 1. 특징 강화 (20개 → 47개)
- 다중 기간 추세 분석
- 계절성 패턴 감지
- 고객 행동 변화 추적

### 2. 클래스 불균형 해결
- SMOTE 적용
- Recall +17.5%p 향상

### 3. 모델 최적화
- XGBoost + LightGBM 앙상블
- 하이퍼파라미터 자동 튜닝

### 4. 성능 향상
| 지표 | V1.0 | V2.0 | 개선 |
|------|------|------|------|
| Accuracy | 94.3% | 97.2% | +2.9%p |
| Recall | 68.2% | 85.7% | +17.5%p |
| Precision | 76.5% | 89.3% | +12.8%p |

---

## 문서 위치

- **전체 가이드**: `README.md`
- **개선 사항**: `CHANGELOG_V2.md`
- **코드 설명**: `src/README.md`
- **데이터 가이드**: `data/README.md`
- **학습 과정**: `notebooks/train_model.ipynb`

---

## 사용 팁

### 첫 학습 시
1. 샘플 데이터로 테스트 (빠름)
2. 전체 데이터로 학습 (정확함)
3. 성능 평가 후 파라미터 조정

### 성능 개선 시
1. **특징 추가**: `feature_engineering.py` 수정
2. **하이퍼파라미터**: `train.py`에서 조정
3. **앙상블 가중치**: `config.json` 수정

### 배포 시
1. `models/` 폴더 전체 복사
2. `src/predictor.py`만 사용
3. API 서버 구축 (FastAPI 추천)

---

## 주의사항

### 데이터 준비
- 3개의 CSV 파일 모두 필수
- 최소 3개월 이상의 월별 데이터
- 폐업 매장 1-5% 권장

### 모델 사용
- 예측은 참고용, 실제 판단은 전문가와 상담
- 주기적 재학습 권장 (3-6개월)
- 업종별 차이 고려

---

## 기여 및 문의

- GitHub Issues: 버그 리포트, 기능 제안
- Pull Request: 코드 개선, 문서 수정 환영

---

## 라이선스

MIT License - 자유롭게 사용, 수정, 배포 가능
