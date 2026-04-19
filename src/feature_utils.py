"""
特徵萃取工具模組

提供 MFCC 和 Mel-spectrogram 的萃取函式。
萃取時才做 3 秒的 pad/truncate（取中間段），前處理後的完整音訊作為輸入。

參數設定（已定案）：
- sr = 16000 Hz
- MFCC: n_mfcc=13, + delta + delta-delta = 39 維
- Mel-spectrogram: n_mels=128, n_fft=2048, hop_length=512 → shape (128, 94)
"""

from pathlib import Path

import numpy as np
import librosa


# === 特徵萃取參數 ===
TARGET_SR: int = 16000
TARGET_DURATION: float = 3.0  # 目標長度（秒）
TARGET_SAMPLES: int = int(TARGET_SR * TARGET_DURATION)  # 48000 samples

# MFCC 參數
N_MFCC: int = 13

# Mel-spectrogram 參數
N_MELS: int = 128
N_FFT: int = 2048
HOP_LENGTH: int = 512


def pad_or_truncate(audio: np.ndarray, target_length: int) -> np.ndarray:
    """將音訊截斷或補零至指定長度。

    截斷策略：取中間段（去除頭尾），保留最有意義的部分。
    補零策略：左右對稱補零。

    Args:
        audio: 音訊 numpy 陣列
        target_length: 目標樣本數

    Returns:
        固定長度的音訊陣列
    """
    length = len(audio)

    if length > target_length:
        # 截斷：取中間段
        start = (length - target_length) // 2
        audio = audio[start:start + target_length]
    elif length < target_length:
        # 補零：左右對稱
        pad_total = target_length - length
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        audio = np.pad(audio, (pad_left, pad_right), mode="constant", constant_values=0)

    return audio


def extract_mfcc(
    audio: np.ndarray,
    sr: int = TARGET_SR,
    n_mfcc: int = N_MFCC,
    target_duration: float = TARGET_DURATION,
) -> np.ndarray:
    """萃取 MFCC 特徵（含 delta 和 delta-delta）。

    先將音訊截斷/補零至固定長度，再萃取 MFCC。
    輸出 shape: (n_frames, 39)，其中 39 = 13 MFCC + 13 delta + 13 delta-delta。

    Args:
        audio: 前處理後的音訊陣列（完整長度）
        sr: 取樣率
        n_mfcc: MFCC 維度數
        target_duration: 目標時長（秒）

    Returns:
        MFCC 特徵矩陣，shape (n_frames, 39)
    """
    target_samples = int(sr * target_duration)
    audio = pad_or_truncate(audio, target_samples)

    # 萃取 MFCC: shape (n_mfcc, n_frames)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    # 計算 delta 和 delta-delta
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # 堆疊: shape (39, n_frames) → 轉置為 (n_frames, 39)
    features = np.vstack([mfcc, delta, delta2]).T

    return features


def extract_mfcc_summary(
    audio: np.ndarray,
    sr: int = TARGET_SR,
    n_mfcc: int = N_MFCC,
    target_duration: float = TARGET_DURATION,
) -> np.ndarray:
    """萃取 MFCC 統計量特徵（適用於傳統 ML）。

    對每個 MFCC 維度（含 delta/delta-delta）計算 mean 和 std，
    產出固定長度的特徵向量。

    Args:
        audio: 前處理後的音訊陣列
        sr: 取樣率
        n_mfcc: MFCC 維度數
        target_duration: 目標時長（秒）

    Returns:
        MFCC 統計量向量，shape (78,) = 39 mean + 39 std
    """
    mfcc_seq = extract_mfcc(audio, sr, n_mfcc, target_duration)
    # mfcc_seq shape: (n_frames, 39)
    mean = np.mean(mfcc_seq, axis=0)  # (39,)
    std = np.std(mfcc_seq, axis=0)    # (39,)
    return np.concatenate([mean, std])  # (78,)


def extract_melspectrogram(
    audio: np.ndarray,
    sr: int = TARGET_SR,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    target_duration: float = TARGET_DURATION,
) -> np.ndarray:
    """萃取 Log Mel-spectrogram（適用於 CNN 輸入）。

    先將音訊截斷/補零至固定長度，再萃取 Mel-spectrogram 並轉換為 dB scale。
    輸出 shape: (n_mels, n_frames) = (128, 94)（3 秒 @ hop_length=512）。

    Args:
        audio: 前處理後的音訊陣列（完整長度）
        sr: 取樣率
        n_mels: Mel 濾波器數量
        n_fft: FFT 視窗大小
        hop_length: 跳躍步長
        target_duration: 目標時長（秒）

    Returns:
        Log Mel-spectrogram 矩陣，shape (n_mels, n_frames)
    """
    target_samples = int(sr * target_duration)
    audio = pad_or_truncate(audio, target_samples)

    # 萃取 Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,
    )

    # 轉換為 log scale（dB）
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    return log_mel_spec
