import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
import warnings

warnings.filterwarnings('ignore')


class EarlyWarningPredictor:
    """자영업 조기경보 예측 모델"""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = Path(model_path) if model_path else Path(__file__).parent.parent / 'model'
        self.xgb_model = None
        self.lgb_model = None
        self.catboost_model = None
        self.label_encoders = {}
        self.feature_names = []
        self.config = {}
        self.is_loaded = False

    @classmethod
    def from_pretrained(cls, model_name_or_path: str):
        predictor = cls(model_path=model_name_or_path)
        predictor.load_model()
        return predictor

    def load_model(self):
        """모델 및 설정 로드"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_path}")

        # XGBoost 로드
        xgb_path = self.model_path / 'xgboost_model.pkl'
        if xgb_path.exists():
            with open(xgb_path, 'rb') as f:
                self.xgb_model = pickle.load(f)

        # LightGBM 로드
        lgb_path = self.model_path / 'lightgbm_model.pkl'
        if lgb_path.exists():
            with open(lgb_path, 'rb') as f:
                self.lgb_model = pickle.load(f)

        # CatBoost 로드
        catboost_path = self.model_path / 'catboost_model.pkl'
        if catboost_path.exists():
            with open(catboost_path, 'rb') as f:
                self.catboost_model = pickle.load(f)

        # Label Encoders 로드
        le_path = self.model_path / 'label_encoders.pkl'
        if le_path.exists():
            with open(le_path, 'rb') as f:
                self.label_encoders = pickle.load(f)

        # Feature names 로드
        fn_path = self.model_path / 'feature_names.json'
        if fn_path.exists():
            with open(fn_path, 'r', encoding='utf-8') as f:
                self.feature_names = json.load(f)

        # Config 로드
        config_path = self.model_path / 'config.json'
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)

        self.is_loaded = True
        print(f"모델 로드 완료: v{self.config.get('model_version', '2.0')}")

    def predict(self, store_data: Dict,
                monthly_usage: Optional[pd.DataFrame] = None,
                monthly_customers: Optional[pd.DataFrame] = None,
                threshold: Optional[float] = None) -> Dict:
        if not self.is_loaded:
            self.load_model()

        # 특징 생성
        from src.feature_engineering import FeatureEngineer
        engineer = FeatureEngineer()

        if monthly_usage is None or monthly_customers is None:
            # 간단한 데이터 형식
            features = self._create_simple_features(store_data)
        else:
            # 전체 특징 생성
            features = engineer.create_features(store_data, monthly_usage, monthly_customers)

        # 특징 정렬 및 결측치 처리
        features = self._align_features(features)

        # 예측
        threshold = threshold or self.config.get('threshold', 0.5)

        if self.xgb_model and self.lgb_model:
            # 앙상블 예측
            xgb_prob = self.xgb_model.predict_proba(features)[0][1]
            lgb_prob = self.lgb_model.predict_proba(features)[0][1]

            weights = self.config.get('ensemble_weights', [0.5, 0.5])
            closure_probability = weights[0] * xgb_prob + weights[1] * lgb_prob

            if self.catboost_model and len(weights) > 2:
                cat_prob = self.catboost_model.predict_proba(features)[0][1]
                closure_probability = (weights[0] * xgb_prob +
                                       weights[1] * lgb_prob +
                                       weights[2] * cat_prob)
        else:
            closure_probability = 0.5

        # 위험도 점수(0-100)
        risk_score = closure_probability * 100

        # 위험 등급
        if risk_score < 30:
            risk_level = '낮음'
            risk_color = 'green'
        elif risk_score < 60:
            risk_level = '보통'
            risk_color = 'yellow'
        else:
            risk_level = '높음'
            risk_color = 'red'

        # 예측 결과
        result = {
            'risk_score': round(risk_score, 2),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'closure_probability': round(closure_probability, 4),
            'is_at_risk': closure_probability > threshold,
            'threshold': threshold,
            'confidence': max(closure_probability, 1 - closure_probability),
            'model_version': self.config.get('model_version', '2.0')
        }

        # 위험 요인 분석(특징 중요도 기반)
        if self.xgb_model:
            result['risk_factors'] = self._analyze_risk_factors(features)

        # 액션 아이템
        result['action_items'] = self._generate_action_items(result, store_data)

        return result

    def predict_batch(self, stores_df: pd.DataFrame) -> pd.DataFrame:
        results = []

        for idx, row in stores_df.iterrows():
            store_data = row.to_dict()
            result = self.predict(store_data)
            result['store_id'] = row.get('store_id', idx)
            results.append(result)

        return pd.DataFrame(results)

    def explain(self, store_data: Dict, top_n: int = 10) -> Dict:
        # SHAP 분석(간단한 버전)
        result = self.predict(store_data)

        explanation = {
            'prediction': result,
            'top_features': result.get('risk_factors', {}),
            'interpretation': self._interpret_prediction(result)
        }

        return explanation

    def _create_simple_features(self, store_data: Dict) -> pd.DataFrame:
        """간단한 특징 생성"""
        # 기본 특징만 사용
        features = {
            'sales_avg_all': store_data.get('avg_sales', 50),
            'customer_reuse_rate': store_data.get('reuse_rate', 25),
            'operation_months': store_data.get('operating_months', 12),
            'trend_slope': store_data.get('sales_trend', 0),
        }

        # 나머지 특징은 기본값으로
        for fname in self.feature_names:
            if fname not in features:
                features[fname] = 0

        return pd.DataFrame([features])

    def _align_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """특징 정렬 및 전처리"""
        # 모델 학습 시 사용한 특징 순서로 정렬
        aligned = pd.DataFrame()

        for fname in self.feature_names:
            if fname in features.columns:
                aligned[fname] = features[fname]
            else:
                aligned[fname] = 0

        # 결측치 처리
        aligned = aligned.fillna(aligned.median().fillna(0))

        return aligned

    def _analyze_risk_factors(self, features: pd.DataFrame) -> Dict[str, float]:
        """위험 요인 분석"""
        # 특징 중요도 기반
        if not hasattr(self.xgb_model, 'feature_importances_'):
            return {}

        importance = self.xgb_model.feature_importances_
        feature_values = features.iloc[0].values

        # 중요도와 값을 곱해서 기여도 계산
        contributions = {}

        for i, fname in enumerate(self.feature_names):
            if importance[i] > 0.01:  # 중요한 특징만
                score = importance[i] * abs(feature_values[i]) * 10

                # 특징명을 한글로 변환
                readable_name = self._translate_feature_name(fname)
                contributions[readable_name] = min(round(score, 1), 100)

        # 상위 6개만 반환
        sorted_factors = sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:6]

        return dict(sorted_factors)

    def _translate_feature_name(self, fname: str) -> str:
        """특징명을 읽기 쉬운 형태로 변환"""
        translations = {
            'sales_avg': '매출',
            'trend_slope': '매출 추세',
            'trend_consecutive_down': '연속 하락',
            'customer_reuse_rate': '재이용률',
            'volatility_cv': '매출 변동성',
            'operation_months': '영업 기간',
            'sales_recent_vs_previous': '최근 매출 변화'
        }

        for key, value in translations.items():
            if key in fname:
                return value

        return fname

    def _generate_action_items(self, result: Dict, store_data: Dict) -> List[str]:
        """액션 아이템 생성"""
        actions = []

        risk_score = result['risk_score']

        if risk_score > 70:
            actions.append("즉시 조치 필요: 비용 절감 및 매출 증대 방안 마련")
            actions.append("현금흐름 개선: 외상 매출 회수 및 재고 최적화")
            actions.append("전문가 상담: 경영 컨설팅 및 구조조정 검토")
        elif risk_score > 40:
            actions.append("매출 분석: 주력 상품/서비스 재점검")
            actions.append("마케팅 강화: 신규 고객 유치 캠페인")
            actions.append("차별화 전략: 경쟁력 있는 요소 발굴 및 강화")
        else:
            actions.append("현재 상태 유지: 정기적인 모니터링 지속")
            actions.append("성장 기회 탐색: 추가 매출원 발굴")
            actions.append("고객 충성도 강화: 멤버십 프로그램 등")

        return actions

    def _interpret_prediction(self, result: Dict) -> str:
        """예측 결과 해석"""
        risk_level = result['risk_level']
        risk_score = result['risk_score']

        if risk_level == '높음':
            return f"위험도가 매우 높습니다 ({risk_score:.1f}점). 폐업 위험이 높으므로 즉각적인 대응이 필요합니다."
        elif risk_level == '보통':
            return f"주의가 필요합니다 ({risk_score:.1f}점). 개선 방안을 마련하여 위험을 줄이세요."
        else:
            return f"안정적입니다 ({risk_score:.1f}점). 현재의 운영 방식을 유지하면서 지속적으로 모니터링하세요."

    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        return {
            'version': self.config.get('model_version', '2.0'),
            'n_features': self.config.get('n_features', 0),
            'performance': self.config.get('performance', {}),
            'ensemble_weights': self.config.get('ensemble_weights', []),
            'models': {
                'xgboost': self.xgb_model is not None,
                'lightgbm': self.lgb_model is not None,
                'catboost': self.catboost_model is not None
            }
        }


if __name__ == "__main__":
    # 사용 예시
    print("=" * 70)
    print("Early Warning Predictor v2.0 테스트")
    print("=" * 70)

    # 모델 로드
    predictor = EarlyWarningPredictor(model_path='../model')

    try:
        predictor.load_model()

        # 테스트 데이터
        store_data = {
            'store_id': 'TEST_001',
            'industry': '카페',
            'location': '서울 강남구',
            'avg_sales': 45,
            'reuse_rate': 22.5,
            'operating_months': 18,
            'sales_trend': -0.05
        }

        # 예측
        result = predictor.predict(store_data)

        print("\n예측 결과:")
        print(f"  위험도 점수: {result['risk_score']}/100")
        print(f"  위험 등급: {result['risk_level']}")
        print(f"  폐업 확률: {result['closure_probability']:.1%}")

        if 'risk_factors' in result:
            print("\n주요 위험 요인:")
            for factor, score in result['risk_factors'].items():
                print(f"    - {factor}: {score:.1f}점")

        print("\n액션 아이템:")
        for action in result['action_items']:
            print(f"    {action}")

    except FileNotFoundError:
        print("모델 파일이 없습니다. 먼저 모델을 학습해주세요.")
