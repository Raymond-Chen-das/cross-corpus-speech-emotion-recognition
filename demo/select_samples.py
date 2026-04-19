"""
從 fold-1 test set 中挑選高信心度正確預測樣本作為 Demo 保險樣本。

每個情緒挑 1 個 confidence 最高的正確預測。
輸出到 demo/samples/ 目錄。
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import librosa
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT_PATH = PROJECT_ROOT / 'models' / 'checkpoints' / 'wav2vec' / 'wav2vec_fold1_best.pt'
AUDIO_DIR = PROJECT_ROOT / 'data' / 'processed' / 'audio_16k'
METADATA_PATH = PROJECT_ROOT / 'data' / 'metadata.csv'
SAMPLES_DIR = Path(__file__).resolve().parent / 'samples'
SAMPLES_DIR.mkdir(exist_ok=True)

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
TARGET_SR = 16000
MAX_LENGTH_SAMPLES = 48000

DEVICE = torch.device('cpu')


def load_and_preprocess(path):
    audio, _ = librosa.load(path, sr=TARGET_SR)
    if len(audio) > MAX_LENGTH_SAMPLES:
        start = (len(audio) - MAX_LENGTH_SAMPLES) // 2
        audio = audio[start:start + MAX_LENGTH_SAMPLES]
    elif len(audio) < MAX_LENGTH_SAMPLES:
        pad_total = MAX_LENGTH_SAMPLES - len(audio)
        audio = np.pad(audio, (pad_total // 2, pad_total - pad_total // 2))
    return audio


def main():
    # 載入 metadata 並找出 fold-1 的 test 樣本（用 StratifiedGroupKFold 重建）
    df = pd.read_csv(METADATA_PATH)
    print(f'總樣本數: {len(df):,}')

    # 載入模型
    print('載入模型...')
    fe = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base')
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        'facebook/wav2vec2-base', num_labels=6, classifier_proj_size=256,
    )
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    # 從每個情緒隨機取 30 個樣本來篩選（避免跑全部 11K 樣本）
    candidates = []
    for emo in EMOTIONS:
        emo_df = df[df['emotion'] == emo]
        sample = emo_df.sample(n=min(30, len(emo_df)), random_state=42)
        candidates.append(sample)
    candidates = pd.concat(candidates).reset_index(drop=True)
    print(f'候選樣本: {len(candidates)}')

    # 逐一推論
    results = []
    for idx, row in tqdm(candidates.iterrows(), total=len(candidates), desc='推論中'):
        audio_path = AUDIO_DIR / Path(row['processed_path']).name
        if not audio_path.exists():
            continue

        audio = load_and_preprocess(str(audio_path))
        inputs = fe(audio, sampling_rate=TARGET_SR, return_tensors='pt', padding=False)

        with torch.no_grad():
            out = model(inputs['input_values'])
        probs = torch.softmax(out.logits[0], dim=0).numpy()

        pred_idx = int(probs.argmax())
        pred_emo = EMOTIONS[pred_idx]
        true_emo = row['emotion']
        confidence = float(probs[pred_idx])

        results.append({
            'audio_path': str(audio_path),
            'true_emotion': true_emo,
            'pred_emotion': pred_emo,
            'confidence': confidence,
            'correct': pred_emo == true_emo,
            'filename': audio_path.name,
        })

    results_df = pd.DataFrame(results)
    correct = results_df[results_df['correct']]
    print(f'\n正確預測: {len(correct)} / {len(results_df)}')

    # 每個情緒選 confidence 最高的 1 個
    selected = []
    for emo in EMOTIONS:
        emo_correct = correct[correct['true_emotion'] == emo].sort_values('confidence', ascending=False)
        if len(emo_correct) > 0:
            best = emo_correct.iloc[0]
            selected.append(best)
            print(f'  {emo:>8s}: {best["filename"]} (conf={best["confidence"]:.1%})')
        else:
            print(f'  {emo:>8s}: [無正確預測樣本]')

    # 複製到 demo/samples/
    for row in selected:
        src = Path(row['audio_path'])
        dst = SAMPLES_DIR / f'{row["true_emotion"]}_{src.name}'
        shutil.copy2(src, dst)
        print(f'  -> {dst.name}')

    print(f'\n完成！{len(selected)} 個樣本已存到 {SAMPLES_DIR}')


if __name__ == '__main__':
    main()
