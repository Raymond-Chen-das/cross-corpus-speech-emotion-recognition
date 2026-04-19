"""
資料載入與 metadata 建構工具模組

掃描四個子資料集（RAVDESS, CREMA-D, TESS, SAVEE），
解析檔案名稱中的情緒標籤與說話者 ID，產出統一的 metadata.csv。
"""

import re
from pathlib import Path

import pandas as pd

# === 專案根目錄（自動偵測） ===
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"

# === RAVDESS 情緒代碼對照表 ===
RAVDESS_EMOTION_MAP: dict[str, str] = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}

# === CREMA-D 情緒縮寫對照表 ===
CREMAD_EMOTION_MAP: dict[str, str] = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

# === TESS 情緒標籤對照表（處理 'ps' = surprise） ===
TESS_EMOTION_MAP: dict[str, str] = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "ps": "surprise",
}

# === SAVEE 情緒前綴對照表 ===
SAVEE_EMOTION_MAP: dict[str, str] = {
    "a": "angry",
    "d": "disgust",
    "f": "fear",
    "h": "happy",
    "n": "neutral",
    "sa": "sad",
    "su": "surprise",
}


def parse_ravdess(raw_dir: Path) -> list[dict[str, str]]:
    """解析 RAVDESS 資料集。

    檔案格式：03-01-06-01-02-01-12.wav
    - 第 3 段（index 2）為情緒代碼
    - 資料夾名稱 Actor_XX 為說話者 ID

    Args:
        raw_dir: 原始資料根目錄（data/raw/）

    Returns:
        解析後的記錄列表
    """
    ravdess_dir = raw_dir / "RAVDESS" / "audio_speech_actors_01-24"
    records: list[dict[str, str]] = []

    for actor_dir in sorted(ravdess_dir.iterdir()):
        if not actor_dir.is_dir() or not actor_dir.name.startswith("Actor_"):
            continue

        # 提取說話者編號（例如 Actor_01 → 01）
        speaker_num = actor_dir.name.split("_")[1]
        speaker_id = f"RAVDESS_{speaker_num}"

        for wav_file in sorted(actor_dir.glob("*.wav")):
            parts = wav_file.stem.split("-")
            emotion_code = parts[2]  # 第 3 段為情緒代碼
            emotion = RAVDESS_EMOTION_MAP.get(emotion_code, "unknown")

            records.append({
                "filepath": wav_file.relative_to(PROJECT_ROOT).as_posix(),
                "emotion": emotion,
                "dataset": "RAVDESS",
                "speaker_id": speaker_id,
            })

    return records


def parse_cremad(raw_dir: Path) -> list[dict[str, str]]:
    """解析 CREMA-D 資料集。

    檔案格式：1001_DFA_ANG_XX.wav
    - 第 1 段為說話者 ID
    - 第 3 段為情緒縮寫

    Args:
        raw_dir: 原始資料根目錄（data/raw/）

    Returns:
        解析後的記錄列表
    """
    cremad_dir = raw_dir / "CREMA-D" / "AudioWAV"
    records: list[dict[str, str]] = []

    for wav_file in sorted(cremad_dir.glob("*.wav")):
        parts = wav_file.stem.split("_")
        speaker_id = f"CREMAD_{parts[0]}"
        emotion_code = parts[2]  # 第 3 段為情緒縮寫
        emotion = CREMAD_EMOTION_MAP.get(emotion_code, "unknown")

        records.append({
            "filepath": wav_file.relative_to(PROJECT_ROOT).as_posix(),
            "emotion": emotion,
            "dataset": "CREMA-D",
            "speaker_id": speaker_id,
        })

    return records


def parse_tess(raw_dir: Path) -> list[dict[str, str]]:
    """解析 TESS 資料集。

    檔案格式：OAF_back_angry.wav
    - 第 1 段為說話者（OAF 或 YAF）
    - 最後一段為情緒標籤（'ps' 代表 surprise）

    ⚠️ TESS 資料夾中有巢狀重複的子目錄，需跳過以避免重複計算。

    Args:
        raw_dir: 原始資料根目錄（data/raw/）

    Returns:
        解析後的記錄列表
    """
    tess_root = raw_dir / "TESS" / "TESS Toronto emotional speech set data"
    records: list[dict[str, str]] = []

    for emotion_folder in sorted(tess_root.iterdir()):
        if not emotion_folder.is_dir():
            continue
        # 跳過巢狀重複資料夾
        if emotion_folder.name == "TESS Toronto emotional speech set data":
            continue

        for wav_file in sorted(emotion_folder.glob("*.wav")):
            parts = wav_file.stem.split("_")
            speaker = parts[0].upper()  # OAF 或 YAF
            # 修正命名不一致（如 OA → OAF）
            if speaker == "OA":
                speaker = "OAF"
            speaker_id = f"TESS_{speaker}"
            emotion_raw = parts[-1].lower()  # 最後一段為情緒
            emotion = TESS_EMOTION_MAP.get(emotion_raw, "unknown")

            records.append({
                "filepath": wav_file.relative_to(PROJECT_ROOT).as_posix(),
                "emotion": emotion,
                "dataset": "TESS",
                "speaker_id": speaker_id,
            })

    return records


def parse_savee(raw_dir: Path) -> list[dict[str, str]]:
    """解析 SAVEE 資料集。

    檔案格式：DC_a01.wav
    - 第 1 段為說話者（DC/JE/JK/KL）
    - 第 2 段的字母前綴為情緒代碼（a/d/f/h/n/sa/su）

    Args:
        raw_dir: 原始資料根目錄（data/raw/）

    Returns:
        解析後的記錄列表
    """
    savee_dir = raw_dir / "SAVEE" / "ALL"
    records: list[dict[str, str]] = []

    for wav_file in sorted(savee_dir.glob("*.wav")):
        parts = wav_file.stem.split("_")
        speaker_id = f"SAVEE_{parts[0]}"

        # 用正則表達式分離情緒前綴與數字編號
        match = re.match(r"^([a-z]+)(\d+)$", parts[1])
        if match:
            emotion_prefix = match.group(1)
            emotion = SAVEE_EMOTION_MAP.get(emotion_prefix, "unknown")
        else:
            emotion = "unknown"

        records.append({
            "filepath": wav_file.relative_to(PROJECT_ROOT).as_posix(),
            "emotion": emotion,
            "dataset": "SAVEE",
            "speaker_id": speaker_id,
        })

    return records


def build_metadata() -> pd.DataFrame:
    """合併四個資料集的解析結果，建構統一的 metadata DataFrame。

    Returns:
        包含 filepath, emotion, dataset, speaker_id 欄位的 DataFrame
    """
    print("正在解析 RAVDESS...")
    ravdess = parse_ravdess(RAW_DATA_DIR)
    print(f"  → {len(ravdess)} 筆")

    print("正在解析 CREMA-D...")
    cremad = parse_cremad(RAW_DATA_DIR)
    print(f"  → {len(cremad)} 筆")

    print("正在解析 TESS...")
    tess = parse_tess(RAW_DATA_DIR)
    print(f"  → {len(tess)} 筆")

    print("正在解析 SAVEE...")
    savee = parse_savee(RAW_DATA_DIR)
    print(f"  → {len(savee)} 筆")

    # 合併所有記錄
    all_records = ravdess + cremad + tess + savee
    df = pd.DataFrame(all_records)

    return df


def print_summary(df: pd.DataFrame) -> None:
    """印出資料集統計摘要。

    Args:
        df: metadata DataFrame
    """
    print("\n" + "=" * 60)
    print("[統計] 資料集統計摘要")
    print("=" * 60)

    print(f"\n總檔案數：{len(df)}")

    # 各資料集樣本數
    print("\n【各資料集樣本數】")
    dataset_counts = df["dataset"].value_counts().sort_index()
    for dataset, count in dataset_counts.items():
        print(f"  {dataset:10s}: {count:>6,}")

    # 各情緒類別樣本數
    print("\n【各情緒類別樣本數】")
    emotion_counts = df["emotion"].value_counts().sort_index()
    for emotion, count in emotion_counts.items():
        print(f"  {emotion:10s}: {count:>6,}")

    # 各資料集的說話者數
    print("\n【各資料集說話者數】")
    for dataset in sorted(df["dataset"].unique()):
        n_speakers = df[df["dataset"] == dataset]["speaker_id"].nunique()
        print(f"  {dataset:10s}: {n_speakers:>6}")

    # 檢查是否有 unknown 標籤
    unknown_count = (df["emotion"] == "unknown").sum()
    if unknown_count > 0:
        print(f"\n[警告] 發現 {unknown_count} 筆無法辨識的情緒標籤！")
    else:
        print("\n[OK] 所有情緒標籤均成功解析，無 unknown。")

    # 檢查是否有重複 filepath
    dup_count = df["filepath"].duplicated().sum()
    if dup_count > 0:
        print(f"[警告] 發現 {dup_count} 筆重複的檔案路徑！")
    else:
        print("[OK] 無重複檔案路徑。")

    # 情緒 × 資料集 交叉表
    print("\n【情緒 × 資料集 交叉表】")
    cross_tab = pd.crosstab(df["emotion"], df["dataset"])
    print(cross_tab.to_string())


def main() -> None:
    """主程式：建構 metadata 並儲存為 CSV。"""
    df = build_metadata()

    # 儲存 metadata.csv
    output_path = PROJECT_ROOT / "data" / "metadata.csv"
    df.to_csv(output_path, index=False)
    print(f"\n[OK] metadata 已儲存至：{output_path}")

    # 印出統計摘要
    print_summary(df)


if __name__ == "__main__":
    main()
