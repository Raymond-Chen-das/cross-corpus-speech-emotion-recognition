"""
Phase 3: Baseline ML 訓練腳本

從已萃取的 MFCC .npy (94, 39) 計算統計量摘要 (234D)，
使用 StratifiedGroupKFold (group=speaker_id) 訓練 SVM / RF / XGBoost。

用法：
    cd src && python run_baseline_ml.py
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[WARNING] xgboost not installed, skipping XGBoost")


# === 路徑設定 ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = PROJECT_ROOT / "data" / "metadata.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "baseline_ml"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# === 實驗參數 ===
N_SPLITS = 5
RANDOM_STATE = 42


# === 特徵工程：MFCC 統計量摘要 ===

def mfcc_to_stats(mfcc: np.ndarray) -> np.ndarray:
    """將 MFCC 時間序列 (n_frames, 39) 壓縮為 234 維統計量向量。

    對時間軸 (axis=0) 計算 6 種統計量 × 39 維 = 234 維：
    mean, std, max, min, skewness, kurtosis

    Args:
        mfcc: shape (n_frames, 39)

    Returns:
        shape (234,)
    """
    return np.concatenate([
        np.mean(mfcc, axis=0),      # 39
        np.std(mfcc, axis=0),       # 39
        np.max(mfcc, axis=0),       # 39
        np.min(mfcc, axis=0),       # 39
        skew(mfcc, axis=0),         # 39
        kurtosis(mfcc, axis=0),     # 39
    ])  # total: 234


def load_features(df: pd.DataFrame, project_root: Path) -> np.ndarray:
    """載入所有 MFCC .npy 並轉為統計量特徵矩陣。

    先批次載入所有 .npy 為 3D 陣列，再用 numpy 向量化計算統計量，
    比逐檔計算 scipy.stats 快 10 倍以上。

    Args:
        df: metadata DataFrame，需有 mfcc_path 欄位
        project_root: 專案根目錄

    Returns:
        shape (n_samples, 234)
    """
    # 批次載入為 3D 陣列 (n_samples, 94, 39)
    all_mfcc = np.stack([
        np.load(project_root / p)
        for p in tqdm(df["mfcc_path"], desc="Loading .npy files")
    ])
    print(f"  Loaded: {all_mfcc.shape}")

    # 向量化計算 6 種統計量 (沿 axis=1 = 時間軸)
    feat_mean = np.mean(all_mfcc, axis=1)       # (N, 39)
    feat_std = np.std(all_mfcc, axis=1)          # (N, 39)
    feat_max = np.max(all_mfcc, axis=1)          # (N, 39)
    feat_min = np.min(all_mfcc, axis=1)          # (N, 39)
    feat_skew = skew(all_mfcc, axis=1)           # (N, 39)
    feat_kurt = kurtosis(all_mfcc, axis=1)       # (N, 39)

    features = np.hstack([feat_mean, feat_std, feat_max, feat_min, feat_skew, feat_kurt])

    # 靜音/補零段的 skew/kurtosis 可能產生 NaN，以 0 填補
    nan_count = np.isnan(features).sum()
    if nan_count > 0:
        print(f"  [WARNING] {nan_count} NaN values replaced with 0 (from constant segments)")
        np.nan_to_num(features, copy=False, nan=0.0)

    return features


# === 模型定義 ===

def get_models() -> dict:
    """回傳要訓練的模型字典。"""
    models = {
        "SVM": SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="mlogloss",
        )
    return models


# === 主程式 ===

def main() -> None:
    # 載入 metadata
    df = pd.read_csv(METADATA_PATH)
    print(f"Samples: {len(df):,}, Emotions: {sorted(df['emotion'].unique())}")
    print(f"Datasets: {df['dataset'].value_counts().to_dict()}")
    print(f"Speakers: {df['speaker_id'].nunique()}")

    # 載入特徵
    X = load_features(df, PROJECT_ROOT)
    print(f"\nFeature matrix: {X.shape}  (samples, 234D stats)")

    # 標籤編碼
    le = LabelEncoder()
    y = le.fit_transform(df["emotion"])
    classes = le.classes_.tolist()
    print(f"Classes: {classes}")

    # Group = speaker_id
    groups = df["speaker_id"].values

    # StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # 訓練與評估
    models = get_models()
    all_results = {}

    for model_name, model_template in models.items():
        print(f"\n{'='*60}")
        print(f"  {model_name}")
        print(f"{'='*60}")

        fold_metrics = []
        fold_cms = []
        fold_reports = []
        t_start = time.time()

        for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # 標準化（以 train fold 計算 scaler）
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # 訓練
            from sklearn.base import clone
            model = clone(model_template)
            model.fit(X_train, y_train)

            # 預測
            y_pred = model.predict(X_test)

            # 指標
            acc = accuracy_score(y_test, y_pred)
            f1_w = f1_score(y_test, y_pred, average="weighted")
            f1_macro = f1_score(y_test, y_pred, average="macro")
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)

            fold_metrics.append({
                "fold": fold_idx + 1,
                "accuracy": acc,
                "f1_weighted": f1_w,
                "f1_macro": f1_macro,
                "test_samples": len(y_test),
                "train_samples": len(y_train),
                "test_speakers": len(set(groups[test_idx])),
                "train_speakers": len(set(groups[train_idx])),
            })
            fold_cms.append(cm.tolist())
            fold_reports.append(report)

            print(f"  Fold {fold_idx+1}: acc={acc:.4f}, F1(w)={f1_w:.4f}, "
                  f"F1(macro)={f1_macro:.4f}, "
                  f"test={len(y_test)}, speakers(train/test)="
                  f"{len(set(groups[train_idx]))}/{len(set(groups[test_idx]))}")

        elapsed = time.time() - t_start

        # 匯總
        metrics_df = pd.DataFrame(fold_metrics)
        summary = {
            "accuracy_mean": metrics_df["accuracy"].mean(),
            "accuracy_std": metrics_df["accuracy"].std(),
            "f1_weighted_mean": metrics_df["f1_weighted"].mean(),
            "f1_weighted_std": metrics_df["f1_weighted"].std(),
            "f1_macro_mean": metrics_df["f1_macro"].mean(),
            "f1_macro_std": metrics_df["f1_macro"].std(),
            "total_time_sec": round(elapsed, 1),
        }

        print(f"\n  Summary: acc={summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}")
        print(f"           F1(w)={summary['f1_weighted_mean']:.4f} +/- {summary['f1_weighted_std']:.4f}")
        print(f"           F1(macro)={summary['f1_macro_mean']:.4f} +/- {summary['f1_macro_std']:.4f}")
        print(f"           Time: {elapsed:.1f}s")

        # 儲存結果
        all_results[model_name] = {
            "summary": summary,
            "folds": fold_metrics,
            "confusion_matrices": fold_cms,
            "classification_reports": fold_reports,
            "classes": classes,
        }

    # 寫出 JSON
    output_path = RESULTS_DIR / "baseline_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path.relative_to(PROJECT_ROOT)}")

    # 寫出摘要表
    summary_rows = []
    for name, res in all_results.items():
        s = res["summary"]
        summary_rows.append({
            "Model": name,
            "Accuracy": f"{s['accuracy_mean']:.4f} +/- {s['accuracy_std']:.4f}",
            "F1 (weighted)": f"{s['f1_weighted_mean']:.4f} +/- {s['f1_weighted_std']:.4f}",
            "F1 (macro)": f"{s['f1_macro_mean']:.4f} +/- {s['f1_macro_std']:.4f}",
            "Time (s)": s["total_time_sec"],
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = RESULTS_DIR / "baseline_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary saved to {summary_csv.relative_to(PROJECT_ROOT)}")
    print(f"\n{summary_df.to_string(index=False)}")

    # 驗證：確認每個 fold 的 train/test speaker 沒有交集
    print("\n[Speaker Leakage Check]")
    for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
        train_speakers = set(groups[train_idx])
        test_speakers = set(groups[test_idx])
        overlap = train_speakers & test_speakers
        status = "OK" if len(overlap) == 0 else f"LEAK: {overlap}"
        print(f"  Fold {fold_idx+1}: {status}")


if __name__ == "__main__":
    main()
