"""
音訊前處理工具模組

提供音訊載入、重新取樣、正規化等前處理函式。
前處理階段只做 resample + mono + peak normalize，不做 pad/truncate。
固定長度的截斷/補零留到特徵萃取階段（feature_utils.py）處理。
"""

from pathlib import Path

import numpy as np
import librosa
import soundfile as sf


# === 預設參數 ===
TARGET_SR: int = 16000  # 目標取樣率（Hz）


def load_audio(filepath: str | Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """載入音訊檔案並重新取樣為目標取樣率。

    使用 librosa 載入，自動轉換為 mono 並 resample。

    Args:
        filepath: 音訊檔案路徑
        target_sr: 目標取樣率（預設 16000 Hz）

    Returns:
        (audio, sr) — 音訊 numpy 陣列與取樣率
    """
    audio, sr = librosa.load(str(filepath), sr=target_sr, mono=True)
    return audio, sr


def peak_normalize(audio: np.ndarray) -> np.ndarray:
    """將音訊振幅歸一化到 [-1, 1]（Peak Normalization）。

    若音訊為全零（靜音），則直接回傳不處理。

    Args:
        audio: 音訊 numpy 陣列

    Returns:
        歸一化後的音訊陣列
    """
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio


def preprocess_audio(filepath: str | Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """完整的音訊前處理流程：載入 → resample → mono → peak normalize。

    不做 pad/truncate，保留完整長度。

    Args:
        filepath: 原始音訊檔案路徑
        target_sr: 目標取樣率（預設 16000 Hz）

    Returns:
        (audio, sr) — 前處理後的音訊陣列與取樣率
    """
    audio, sr = load_audio(filepath, target_sr)
    audio = peak_normalize(audio)
    return audio, sr


def save_audio(audio: np.ndarray, filepath: str | Path, sr: int = TARGET_SR) -> None:
    """將音訊陣列儲存為 WAV 檔案。

    Args:
        audio: 音訊 numpy 陣列
        filepath: 輸出檔案路徑
        sr: 取樣率
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(filepath), audio, sr)


def batch_preprocess(
    filepaths: list[str],
    output_dir: str | Path,
    target_sr: int = TARGET_SR,
) -> list[str]:
    """批次前處理音訊檔案：resample + normalize，存到指定目錄。

    輸出檔案命名規則：使用原始的 dataset + 檔名，確保唯一性。
    例如：RAVDESS_03-01-06-01-02-01-12.wav

    Args:
        filepaths: 原始音訊檔案的相對路徑列表（相對於專案根目錄）
        output_dir: 輸出目錄路徑
        target_sr: 目標取樣率

    Returns:
        輸出檔案的相對路徑列表（相對於專案根目錄）
    """
    from tqdm import tqdm

    project_root = Path(__file__).resolve().parent.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[str] = []

    for fp in tqdm(filepaths, desc="前處理音訊"):
        full_path = project_root / fp

        # 產生唯一的輸出檔名：dataset_originalname.wav
        # 從路徑中提取 dataset 名稱（data/raw/RAVDESS/... → RAVDESS）
        parts = Path(fp).parts
        # parts 類似 ('data', 'raw', 'RAVDESS', ...)
        dataset_name = parts[2] if len(parts) > 2 else "unknown"
        output_filename = f"{dataset_name}_{full_path.name}"
        output_path = output_dir / output_filename

        # 前處理並儲存
        audio, sr = preprocess_audio(full_path, target_sr)
        save_audio(audio, output_path, sr)

        # 存相對路徑
        rel_path = output_path.relative_to(project_root).as_posix()
        output_paths.append(rel_path)

    return output_paths
