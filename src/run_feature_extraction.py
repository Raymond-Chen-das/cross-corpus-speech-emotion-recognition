"""
批次特徵萃取腳本（獨立執行版）

跳過已存在的 .npy 檔案，支援中斷後繼續。
"""

from pathlib import Path
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm

from feature_utils import extract_mfcc, extract_melspectrogram

# === 路徑設定 ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MFCC_DIR = PROJECT_ROOT / "data" / "features" / "mfcc"
MELSPEC_DIR = PROJECT_ROOT / "data" / "features" / "melspec"

MFCC_DIR.mkdir(parents=True, exist_ok=True)
MELSPEC_DIR.mkdir(parents=True, exist_ok=True)

# === 載入 metadata ===
df = pd.read_csv(PROJECT_ROOT / "data" / "metadata.csv")
print(f"Total samples: {len(df):,}")

# === 批次萃取 ===
mfcc_paths = []
melspec_paths = []
errors = []
skipped = 0

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Feature extraction"):
    stem = Path(row["processed_path"]).stem
    mfcc_path = MFCC_DIR / f"{stem}.npy"
    melspec_path = MELSPEC_DIR / f"{stem}.npy"

    # 跳過已存在的檔案
    if mfcc_path.exists() and melspec_path.exists():
        mfcc_paths.append(mfcc_path.relative_to(PROJECT_ROOT).as_posix())
        melspec_paths.append(melspec_path.relative_to(PROJECT_ROOT).as_posix())
        skipped += 1
        continue

    try:
        audio_path = PROJECT_ROOT / row["processed_path"]
        audio, sr = librosa.load(str(audio_path), sr=None)

        mfcc = extract_mfcc(audio, sr)
        np.save(mfcc_path, mfcc)

        mel_spec = extract_melspectrogram(audio, sr)
        np.save(melspec_path, mel_spec)

        mfcc_paths.append(mfcc_path.relative_to(PROJECT_ROOT).as_posix())
        melspec_paths.append(melspec_path.relative_to(PROJECT_ROOT).as_posix())

    except Exception as e:
        errors.append((row["processed_path"], str(e)))
        mfcc_paths.append(None)
        melspec_paths.append(None)

print(f"\nDone: {len(df) - len(errors) - skipped:,} new, {skipped:,} skipped, {len(errors)} errors")

# === 更新 metadata ===
df["mfcc_path"] = mfcc_paths
df["melspec_path"] = melspec_paths

failed = df["mfcc_path"].isna().sum()
if failed > 0:
    print(f"Removing {failed} failed samples")
    df = df.dropna(subset=["mfcc_path", "melspec_path"]).reset_index(drop=True)

df.to_csv(PROJECT_ROOT / "data" / "metadata.csv", index=False)
print(f"metadata updated: {len(df):,} samples, columns: {df.columns.tolist()}")

# === 驗證 ===
print("\nShape verification (10 random samples):")
sample = df.sample(10, random_state=42)
for _, row in sample.iterrows():
    mfcc = np.load(PROJECT_ROOT / row["mfcc_path"])
    melspec = np.load(PROJECT_ROOT / row["melspec_path"])
    print(f"  [{row['dataset']:10s}] {row['emotion']:8s} MFCC={mfcc.shape}, Mel-spec={melspec.shape}")
