import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


def safe_numeric_convert(series, default_value=0):
    """안전하게 숫자로 변환"""
    try:
        converted = pd.to_numeric(series, errors='coerce')
        converted = converted.replace(-999999.9, np.nan)
        if converted.isna().all():
            return default_value
        return converted.mean()
    except:
        return default_value


class FeatureEngineer:
    """특징 생성 클래스"""

    def __init__(self, include_weather: bool = False):
        self.include_weather = include_weather

    def create_features(self, store_data: Dict, monthly_usage: pd.DataFrame,
                        monthly_customers: pd.DataFrame) -> pd.DataFrame:
        features = {}

        # 1. 매출 관련 특징
        sales_features = self._create_sales_features(monthly_usage)
        features.update(sales_features)

        # 2. 고객 관련 특징
        customer_features = self._create_customer_features(monthly_customers)
        features.update(customer_features)

        # 3. 운영 관련 특징
        operation_features = self._create_operation_features(monthly_usage)
        features.update(operation_features)

        # 4. 트렌드 특징
        trend_features = self._create_trend_features(monthly_usage)
        features.update(trend_features)

        # 5. 변동성 특징
        volatility_features = self._create_volatility_features(monthly_usage)
        features.update(volatility_features)

        # 6. 계절성 특징
        seasonality_features = self._create_seasonality_features(monthly_usage)
        features.update(seasonality_features)

        # 7. 맥락 특징
        context_features = self._create_context_features(store_data, monthly_usage)
        features.update(context_features)

        return pd.DataFrame([features])

    def _create_sales_features(self, df: pd.DataFrame) -> Dict:
        """매출 관련 특징 생성(15개)"""
        features = {}

        if len(df) == 0:
            return self._get_default_sales_features()

        # 매출 구간 매핑
        sales_map = {
            '1_0-25%': 25,
            '2_25-50%': 37.5,
            '3_25-50%': 37.5,
            '4_50-75%': 62.5,
            '5_75-100%': 87.5,
            '6_100%+': 100
        }

        if 'RC_M1_SAA' in df.columns:
            sales = df['RC_M1_SAA'].map(sales_map).fillna(50)
        else:
            sales = pd.Series([50] * len(df))

        # 다중 기간 평균
        features['sales_avg_1m'] = sales.tail(1).mean() if len(sales) >= 1 else 50
        features['sales_avg_3m'] = sales.tail(3).mean() if len(sales) >= 3 else 50
        features['sales_avg_6m'] = sales.tail(6).mean() if len(sales) >= 6 else 50
        features['sales_avg_12m'] = sales.mean()

        # 최근 vs 이전
        if len(sales) >= 6:
            recent = sales.tail(3).mean()
            previous = sales.tail(6).head(3).mean()
            features['sales_recent_vs_previous'] = (recent / previous - 1) * 100 if previous > 0 else 0
        else:
            features['sales_recent_vs_previous'] = 0

        # 전월 대비, 전년 대비
        if len(sales) >= 2:
            features['sales_mom_change'] = (sales.iloc[-1] / sales.iloc[-2] - 1) * 100 if sales.iloc[-2] > 0 else 0
        else:
            features['sales_mom_change'] = 0

        if len(sales) >= 13:
            features['sales_yoy_change'] = (sales.iloc[-1] / sales.iloc[-13] - 1) * 100 if sales.iloc[-13] > 0 else 0
        else:
            features['sales_yoy_change'] = 0

        # 최대, 최소, 범위
        features['sales_max'] = sales.max()
        features['sales_min'] = sales.min()
        features['sales_range'] = features['sales_max'] - features['sales_min']

        # 최근 3개월 평균 vs 전체 평균
        if len(sales) >= 3:
            recent_avg = sales.tail(3).mean()
            total_avg = sales.mean()
            features['sales_recent_vs_total'] = (recent_avg / total_avg - 1) * 100 if total_avg > 0 else 0
        else:
            features['sales_recent_vs_total'] = 0

        # 최근 매출이 평균보다 낮은지
        features['sales_below_avg'] = 1 if features['sales_avg_3m'] < features['sales_avg_12m'] else 0

        # 최근 매출 추세(최근 3개월)
        if len(sales) >= 3:
            recent_sales = sales.tail(3).values
            if len(recent_sales) >= 2:
                slope = (recent_sales[-1] - recent_sales[0]) / len(recent_sales)
                features['sales_recent_trend'] = slope
            else:
                features['sales_recent_trend'] = 0
        else:
            features['sales_recent_trend'] = 0

        return features

    def _create_customer_features(self, df: pd.DataFrame) -> Dict:
        """고객 관련 특징 생성 (12개)"""
        features = {}

        if len(df) == 0:
            return self._get_default_customer_features()

        # 재이용률 - 안전한 변환
        if 'MCT_UE_CLN_REU_RAT' in df.columns:
            try:
                reuse_rate = pd.to_numeric(df['MCT_UE_CLN_REU_RAT'], errors='coerce').replace(-999999.9, np.nan)
                features['customer_reuse_rate'] = reuse_rate.mean() if not reuse_rate.isna().all() else 25.0
                features['customer_reuse_rate_last'] = reuse_rate.iloc[-1] if len(reuse_rate) > 0 and pd.notna(
                    reuse_rate.iloc[-1]) else features['customer_reuse_rate']

                # 재이용률 추세
                if len(reuse_rate) >= 6:
                    recent = reuse_rate.tail(3).mean()
                    previous = reuse_rate.tail(6).head(3).mean()
                    if pd.notna(recent) and pd.notna(previous) and previous > 0:
                        features['customer_reuse_trend'] = (recent / previous - 1) * 100
                    else:
                        features['customer_reuse_trend'] = 0
                else:
                    features['customer_reuse_trend'] = 0
            except:
                features['customer_reuse_rate'] = 25.0
                features['customer_reuse_rate_last'] = 25.0
                features['customer_reuse_trend'] = 0
        else:
            features['customer_reuse_rate'] = 25.0
            features['customer_reuse_rate_last'] = 25.0
            features['customer_reuse_trend'] = 0

        # 신규 고객 비율 - 안전한 변환
        if 'MCT_UE_CLN_NEW_RAT' in df.columns:
            features['customer_new_rate'] = safe_numeric_convert(df['MCT_UE_CLN_NEW_RAT'], 30.0)
        else:
            features['customer_new_rate'] = 30.0

        # 연령대별 고객 비율 (남성) - 안전한 변환
        age_columns_male = ['M12_MAL_1020_RAT', 'M12_MAL_30_RAT', 'M12_MAL_40_RAT',
                            'M12_MAL_50_RAT', 'M12_MAL_60_RAT']
        for col in age_columns_male:
            if col in df.columns:
                features[f'customer_{col.lower()}'] = safe_numeric_convert(df[col], 10.0)
            else:
                features[f'customer_{col.lower()}'] = 10.0

        # 연령대별 고객 비율 (여성) - 안전한 변환
        age_columns_female = ['M12_FME_1020_RAT', 'M12_FME_30_RAT', 'M12_FME_40_RAT',
                              'M12_FME_50_RAT', 'M12_FME_60_RAT']
        for col in age_columns_female:
            if col in df.columns:
                features[f'customer_{col.lower()}'] = safe_numeric_convert(df[col], 10.0)
            else:
                features[f'customer_{col.lower()}'] = 10.0

        return features

    def _create_operation_features(self, df: pd.DataFrame) -> Dict:
        """운영 관련 특징 생성(8개)"""
        features = {}

        if len(df) == 0:
            return self._get_default_operation_features()

        # 영업 개월 수
        if 'MCT_OPE_MS_CN' in df.columns:
            ope_months_map = {
                '1_0-25%': 3,
                '2_25-50%': 9,
                '3_25-50%': 9,
                '4_50-75%': 18,
                '5_75-100%': 30,
                '6_100%+': 48
            }
            ope_numeric = df['MCT_OPE_MS_CN'].map(ope_months_map).fillna(12)
            features['operation_months'] = ope_numeric.iloc[-1] if len(ope_numeric) > 0 else 12
            features['operation_months_avg'] = ope_numeric.mean()
        else:
            features['operation_months'] = 12
            features['operation_months_avg'] = 12

        # 평균 이용 금액
        if 'RC_M1_AV_NP_AT' in df.columns:
            avg_amount_map = {
                '1_0-25%': 15000,
                '2_25-50%': 30000,
                '3_25-50%': 30000,
                '4_50-75%': 45000,
                '5_75-100%': 60000,
                '6_100%+': 80000
            }
            avg_amount = df['RC_M1_AV_NP_AT'].map(avg_amount_map).fillna(30000)
            features['operation_avg_amount'] = avg_amount.mean()
            features['operation_avg_amount_last'] = avg_amount.iloc[-1] if len(avg_amount) > 0 else features[
                'operation_avg_amount']
        else:
            features['operation_avg_amount'] = 30000
            features['operation_avg_amount_last'] = 30000

        # 승인 취소율 - 안전한 변환
        if 'APV_CE_RAT' in df.columns:
            features['operation_cancel_rate'] = safe_numeric_convert(df['APV_CE_RAT'], 5.0)
        else:
            features['operation_cancel_rate'] = 5.0

        # 배달 매출 비율 - 안전한 변환
        if 'DLV_SAA_RAT' in df.columns:
            features['operation_delivery_rate'] = safe_numeric_convert(df['DLV_SAA_RAT'], 20.0)
        else:
            features['operation_delivery_rate'] = 20.0

        return features

    def _create_trend_features(self, df: pd.DataFrame) -> Dict:
        """트렌드 특징 생성(5개)"""
        features = {}

        if len(df) < 3:
            return self._get_default_trend_features()

        # 매출 구간 매핑
        sales_map = {
            '1_0-25%': 25,
            '2_25-50%': 37.5,
            '3_25-50%': 37.5,
            '4_50-75%': 62.5,
            '5_75-100%': 87.5,
            '6_100%+': 100
        }

        if 'RC_M1_SAA' in df.columns:
            sales = df['RC_M1_SAA'].map(sales_map).fillna(50).values
        else:
            sales = np.array([50] * len(df))

        # 선형 회귀
        X = np.arange(len(sales))
        if len(sales) >= 2 and not np.all(np.isnan(sales)):
            valid_mask = ~np.isnan(sales)
            if valid_mask.sum() >= 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(X[valid_mask], sales[valid_mask])
                features['trend_slope'] = slope
                features['trend_r2'] = r_value ** 2
                features['trend_direction'] = 1 if slope > 0 else -1 if slope < 0 else 0
            else:
                features['trend_slope'] = 0
                features['trend_r2'] = 0
                features['trend_direction'] = 0
        else:
            features['trend_slope'] = 0
            features['trend_r2'] = 0
            features['trend_direction'] = 0

        # 연속 하락/상승 개월 수
        consecutive_down = 0
        consecutive_up = 0
        for i in range(len(sales) - 1, 0, -1):
            if not np.isnan(sales[i]) and not np.isnan(sales[i - 1]):
                if sales[i] < sales[i - 1]:
                    consecutive_down += 1
                else:
                    break

        for i in range(len(sales) - 1, 0, -1):
            if not np.isnan(sales[i]) and not np.isnan(sales[i - 1]):
                if sales[i] > sales[i - 1]:
                    consecutive_up += 1
                else:
                    break

        features['trend_consecutive_down'] = consecutive_down
        features['trend_consecutive_up'] = consecutive_up

        return features

    def _create_volatility_features(self, df: pd.DataFrame) -> Dict:
        """변동성 특징 생성(4개)"""
        features = {}

        if len(df) < 2:
            return self._get_default_volatility_features()

        # 매출 구간 매핑
        sales_map = {
            '1_0-25%': 25,
            '2_25-50%': 37.5,
            '3_25-50%': 37.5,
            '4_50-75%': 62.5,
            '5_75-100%': 87.5,
            '6_100%+': 100
        }

        if 'RC_M1_SAA' in df.columns:
            sales = df['RC_M1_SAA'].map(sales_map).fillna(50)
        else:
            sales = pd.Series([50] * len(df))

        # 변동계수(CV)
        mean_sales = sales.mean()
        std_sales = sales.std()
        features['volatility_cv'] = (std_sales / mean_sales * 100) if mean_sales > 0 else 0

        # 표준편차
        features['volatility_std'] = std_sales

        # MAD(Mean Absolute Deviation)
        features['volatility_mad'] = (sales - mean_sales).abs().mean()

        # 최근 3개월 변동성
        if len(sales) >= 3:
            recent_std = sales.tail(3).std()
            features['volatility_recent_std'] = recent_std if not np.isnan(recent_std) else 0
        else:
            features['volatility_recent_std'] = 0

        return features

    def _create_seasonality_features(self, df: pd.DataFrame) -> Dict:
        """계절성 특징 생성(2개)"""
        features = {}

        if len(df) < 12:
            features['seasonality_detected'] = 0
            features['seasonality_strength'] = 0
            return features

        # 매출 구간 매핑
        sales_map = {
            '1_0-25%': 25,
            '2_25-50%': 37.5,
            '3_25-50%': 37.5,
            '4_50-75%': 62.5,
            '5_75-100%': 87.5,
            '6_100%+': 100
        }

        if 'RC_M1_SAA' in df.columns:
            sales = df['RC_M1_SAA'].map(sales_map).fillna(50).values
        else:
            sales = np.array([50] * len(df))

        # 간단한 계절성 감지(최대-최소 차이)
        max_sales = np.nanmax(sales)
        min_sales = np.nanmin(sales)
        mean_sales = np.nanmean(sales)

        if mean_sales > 0:
            seasonality_strength = (max_sales - min_sales) / mean_sales * 100
            features['seasonality_strength'] = seasonality_strength
            features['seasonality_detected'] = 1 if seasonality_strength > 30 else 0
        else:
            features['seasonality_strength'] = 0
            features['seasonality_detected'] = 0

        return features

    def _create_context_features(self, store_data: Dict, df: pd.DataFrame) -> Dict:
        """맥락 특징 생성(1개)"""
        features = {}

        # 업종
        features['context_industry'] = store_data.get('industry', '기타')

        return features

    # 기본값 반환 함수들
    def _get_default_sales_features(self) -> Dict:
        """기본 매출 특징"""
        return {
            'sales_avg_1m': 50, 'sales_avg_3m': 50, 'sales_avg_6m': 50, 'sales_avg_12m': 50,
            'sales_recent_vs_previous': 0, 'sales_mom_change': 0, 'sales_yoy_change': 0,
            'sales_max': 50, 'sales_min': 50, 'sales_range': 0,
            'sales_recent_vs_total': 0, 'sales_below_avg': 0, 'sales_recent_trend': 0
        }

    def _get_default_customer_features(self) -> Dict:
        """기본 고객 특징"""
        features = {
            'customer_reuse_rate': 25.0,
            'customer_reuse_rate_last': 25.0,
            'customer_reuse_trend': 0,
            'customer_new_rate': 30.0
        }
        # 연령대별 기본값
        for age in ['1020', '30', '40', '50', '60']:
            features[f'customer_m12_mal_{age}_rat'] = 10.0
            features[f'customer_m12_fme_{age}_rat'] = 10.0
        return features

    def _get_default_operation_features(self) -> Dict:
        """기본 운영 특징"""
        return {
            'operation_months': 12,
            'operation_months_avg': 12,
            'operation_avg_amount': 30000,
            'operation_avg_amount_last': 30000,
            'operation_cancel_rate': 5.0,
            'operation_delivery_rate': 20.0
        }

    def _get_default_trend_features(self) -> Dict:
        """기본 트렌드 특징"""
        return {
            'trend_slope': 0,
            'trend_r2': 0,
            'trend_direction': 0,
            'trend_consecutive_down': 0,
            'trend_consecutive_up': 0
        }

    def _get_default_volatility_features(self) -> Dict:
        """기본 변동성 특징"""
        return {
            'volatility_cv': 0,
            'volatility_std': 0,
            'volatility_mad': 0,
            'volatility_recent_std': 0
        }
