import pandas as pd
import numpy as np
import pickle
import json
import argparse
from pathlib import Path
import sys

from feature_engineering import FeatureEngineer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE


def load_data(data_dir):
    """데이터 로드"""
    print("데이터 로드 중...")

    df_store = pd.read_csv(f'{data_dir}/big_data_set1_f.csv',
                           encoding='cp949', on_bad_lines='skip')
    df_usage = pd.read_csv(f'{data_dir}/ds2_monthly_usage.csv',
                           encoding='cp949', on_bad_lines='skip')
    df_customer = pd.read_csv(f'{data_dir}/ds3_monthly_customers.csv',
                              encoding='cp949', on_bad_lines='skip')

    print(f"매장 정보: {df_store.shape}")
    print(f"이용 데이터: {df_usage.shape}")
    print(f"고객 데이터: {df_customer.shape}")

    return df_store, df_usage, df_customer


def create_features(df_store, df_usage, df_customer, max_stores=None):
    """특징 생성"""
    print("\n특징 생성 중...")

    engineer = FeatureEngineer(include_weather=False)

    all_features = []
    all_targets = []

    store_ids = df_store['ENCODED_MCT'].unique()
    if max_stores:
        store_ids = store_ids[:max_stores]

    for idx, store_id in enumerate(store_ids):
        store_info = df_store[df_store['ENCODED_MCT'] == store_id].iloc[0]
        usage_data = df_usage[df_usage['ENCODED_MCT'] == store_id]
        customer_data = df_customer[df_customer['ENCODED_MCT'] == store_id]

        # 최소 3개월 데이터 필요
        if len(usage_data) >= 3:
            store_data = {
                'industry': store_info['HPSN_MCT_BZN_CD_NM'] if pd.notna(store_info['HPSN_MCT_BZN_CD_NM']) else '기타',
                'location': store_info['MCT_SIGUNGU_NM']
            }

            features = engineer.create_features(store_data, usage_data, customer_data)
            target = 1 if pd.notna(store_info['MCT_ME_D']) else 0

            all_features.append(features)
            all_targets.append(target)

        if (idx + 1) % 500 == 0:
            print(f"  처리 중... {idx + 1}/{len(store_ids)}")

    X = pd.concat(all_features, ignore_index=True)
    y = pd.Series(all_targets)

    print(f"총 샘플: {len(X)}, 특징 수: {X.shape[1]}")
    print(f"폐업 비율: {y.mean():.2%} ({y.sum()}개)")

    return X, y


def preprocess_data(X, y):
    """데이터 전처리"""
    print("\n데이터 전처리 중...")

    # 카테고리 변수 인코딩
    label_encoders = {}
    if 'context_industry' in X.columns:
        le = LabelEncoder()
        X['context_industry'] = le.fit_transform(X['context_industry'].astype(str))
        label_encoders['context_industry'] = le

    # 결측치 처리
    X = X.fillna(X.median())

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train 폐업: {y_train.mean():.2%}, Test 폐업: {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test, label_encoders


def apply_smote(X_train, y_train):
    """SMOTE 적용"""
    print("\n클래스 불균형 처리(SMOTE)...")

    min_samples = min(y_train.sum(), len(y_train) - y_train.sum())
    k_neighbors = min(5, min_samples - 1)

    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    print(f"SMOTE 후: 영업 {(y_train_balanced == 0).sum()}개, 폐업 {(y_train_balanced == 1).sum()}개")

    return X_train_balanced, y_train_balanced


def train_models(X_train, y_train):
    """모델 학습"""
    print("\n모델 학습 중...")

    # XGBoost
    print("  - XGBoost 학습...")
    xgb_model = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)

    # LightGBM
    print("  - LightGBM 학습...")
    lgb_model = lgb.LGBMClassifier(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)

    print("모델 학습 완료")

    return xgb_model, lgb_model


def evaluate_models(xgb_model, lgb_model, X_test, y_test):
    """모델 평가"""
    print("\n모델 평가 중...")

    # 예측
    xgb_pred = xgb_model.predict_proba(X_test)[:, 1]
    lgb_pred = lgb_model.predict_proba(X_test)[:, 1]

    # 앙상블
    ensemble_pred = 0.5 * xgb_pred + 0.5 * lgb_pred
    ensemble_pred_binary = (ensemble_pred > 0.5).astype(int)

    # 평가 지표
    accuracy = accuracy_score(y_test, ensemble_pred_binary)
    precision = precision_score(y_test, ensemble_pred_binary, zero_division=0)
    recall = recall_score(y_test, ensemble_pred_binary, zero_division=0)
    f1 = f1_score(y_test, ensemble_pred_binary, zero_division=0)
    auc = roc_auc_score(y_test, ensemble_pred)

    print("\n" + "=" * 70)
    print("모델 성능 (Test Set)")
    print("=" * 70)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.1f}%)")
    print(f"Precision: {precision:.4f} ({precision * 100:.1f}%)")
    print(f"Recall:    {recall:.4f} ({recall * 100:.1f}%)")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    print("=" * 70)

    # 혼동 행렬
    cm = confusion_matrix(y_test, ensemble_pred_binary)
    print(f"\n혼동 행렬:")
    print(f"  TN: {cm[0, 0]}, FP: {cm[0, 1]}")
    print(f"  FN: {cm[1, 0]}, TP: {cm[1, 1]}")

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc_roc': float(auc)
    }


def save_models(xgb_model, lgb_model, X, label_encoders, performance, output_dir):
    """모델 저장"""
    print(f"\n모델 저장 중... ({output_dir})")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 모델 저장
    with open(output_path / 'xgboost_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)

    with open(output_path / 'lightgbm_model.pkl', 'wb') as f:
        pickle.dump(lgb_model, f)

    with open(output_path / 'label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)

    # 특징 이름 저장
    feature_names = list(X.columns)
    with open(output_path / 'feature_names.json', 'w', encoding='utf-8') as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)

    # 설정 저장
    config = {
        'model_version': '2.0',
        'ensemble_weights': [0.5, 0.5],
        'threshold': 0.5,
        'n_features': len(feature_names),
        'performance': performance
    }

    with open(output_path / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("모델 저장 완료")
    print(f"  - {output_path / 'xgboost_model.pkl'}")
    print(f"  - {output_path / 'lightgbm_model.pkl'}")
    print(f"  - {output_path / 'config.json'}")


def main():
    parser = argparse.ArgumentParser(description='자영업 조기경보 모델 학습')
    parser.add_argument('--data', type=str, default='data/raw',
                        help='데이터 디렉토리 경로')
    parser.add_argument('--output', type=str, default='models',
                        help='모델 저장 경로')
    parser.add_argument('--max-stores', type=int, default=None,
                        help='최대 매장 수 (테스트용)')

    args = parser.parse_args()

    print("=" * 70)
    print("자영업 조기경보 모델 v2.0 학습")
    print("=" * 70)

    # 1. 데이터 로드
    df_store, df_usage, df_customer = load_data(args.data)

    # 2. 특징 생성
    X, y = create_features(df_store, df_usage, df_customer, args.max_stores)

    # 3. 전처리
    X_train, X_test, y_train, y_test, label_encoders = preprocess_data(X, y)

    # 4. SMOTE
    X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)

    # 5. 모델 학습
    xgb_model, lgb_model = train_models(X_train_balanced, y_train_balanced)

    # 6. 평가
    performance = evaluate_models(xgb_model, lgb_model, X_test, y_test)

    # 7. 저장
    save_models(xgb_model, lgb_model, X, label_encoders, performance, args.output)

    print("\n" + "=" * 70)
    print("학습 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
