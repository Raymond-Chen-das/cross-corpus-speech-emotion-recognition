"""
批次音訊前處理腳本

篩選 6 類情緒，將原始音訊統一為 16kHz mono + peak normalize。
不做 pad/truncate，保留完整長度。跳過已存在的檔案，支援中斷後繼續。
"""

from pathlib import Path

import pandas as pd
import numpy as np
import soundfile as sf
from tqdm import tqdm

from audio_utils import preprocess_audio, save_audio, TARGET_SR

# === 路徑設定 ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_METADATA = PROJECT_ROOT / "data" / "metadata.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "audio_16k"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === 目標 6 類情緒 ===
TARGET_EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad"]


def main() -> None:
    """主程式：篩選情緒 → 批次前處理 → 更新 metadata。"""
    # 載入 metadata
    df = pd.read_csv(RAW_METADATA)
    print(f"原始 metadata：{len(df):,} 筆，{df['emotion'].nunique()} 類情緒")

    # 篩選 6 類情緒
    removed = df[~df["emotion"].isin(TARGET_EMOTIONS)]
    if len(removed) > 0:
        print(f"移除情緒：{removed['emotion'].value_counts().to_dict()}")
        print(f"移除筆數：{len(removed):,}")
    df = df[df["emotion"].isin(TARGET_EMOTIONS)].reset_index(drop=True)
    print(f"篩選後：{len(df):,} 筆，{df['emotion'].nunique()} 類情緒")

    # 批次前處理（跳過已存在的檔案）
    processed_paths: list[str | None] = []
    errors: list[tuple[str, str]] = []
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        full_path = PROJECT_ROOT / row["filepath"]
        dataset_name = row["dataset"]
        output_filename = f"{dataset_name}_{full_path.name}"
        output_path = OUTPUT_DIR / output_filename

        # 跳過已存在
        if output_path.exists():
            processed_paths.append(output_path.relative_to(PROJECT_ROOT).as_posix())
            skipped += 1
            continue

        try:
            audio, sr = preprocess_audio(full_path, TARGET_SR)
            save_audio(audio, output_path, sr)
            processed_paths.append(output_path.relative_to(PROJECT_ROOT).as_posix())
        except Exception as e:
            errors.append((row["filepath"], str(e)))
            processed_paths.append(None)

    print(f"\nDone: {len(df) - len(errors) - skipped:,} new, {skipped:,} skipped, {len(errors)} errors")

    # 更新 metadata
    df["processed_path"] = processed_paths
    failed = df["processed_path"].isna().sum()
    if failed > 0:
        print(f"Removing {failed} failed samples")
        df = df.dropna(subset=["processed_path"]).reset_index(drop=True)

    df.to_csv(RAW_METADATA, index=False)
    print(f"metadata updated: {len(df):,} samples")
    print(f"columns: {df.columns.tolist()}")

    # 驗證抽樣
    print("\nVerification (10 random samples):")
    sample = df.sample(10, random_state=42)
    for _, row in sample.iterrows():
        full_path = PROJECT_ROOT / row["processed_path"]
        info = sf.info(str(full_path))
        print(f"  [{row['dataset']:10s}] sr={info.samplerate} Hz, "
              f"dur={info.duration:.2f}s, emotion={row['emotion']}")


if __name__ == "__main__":
    main()
