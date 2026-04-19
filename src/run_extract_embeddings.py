"""
CNN / LSTM Embedding 萃取腳本（本地 CPU 執行）

從 fold-1 best checkpoint 萃取中間層 representation：
  - CNN: block4 → AdaptiveAvgPool → flatten → (N, 256)
  - LSTM: lstm → attention context → (N, 256)

wav2vec embedding 需在 Colab GPU 執行，見 notebooks/09a_extract_wav2vec_embeddings.ipynb。

用法：
    python src/run_extract_embeddings.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# === 路徑 ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = PROJECT_ROOT / 'data' / 'metadata.csv'
MELSPEC_DIR = PROJECT_ROOT / 'data' / 'features' / 'melspec'
MFCC_DIR = PROJECT_ROOT / 'data' / 'features' / 'mfcc'
CNN_CKPT = PROJECT_ROOT / 'models' / 'checkpoints' / 'cnn' / 'cnn_fold1_best.pt'
LSTM_CKPT = PROJECT_ROOT / 'models' / 'checkpoints' / 'lstm' / 'lstm_fold1_best.pt'
EMB_DIR = PROJECT_ROOT / 'data' / 'embeddings'
EMB_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 64
DEVICE = torch.device('cpu')


# === Dataset ===

class MelSpecDataset(Dataset):
    """載入 Mel-spectrogram .npy 檔案。"""

    def __init__(self, paths: list[str]):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        mel = np.load(self.paths[idx])[np.newaxis, :, :]  # (1, 128, 94)
        return torch.tensor(mel, dtype=torch.float32)


class MFCCDataset(Dataset):
    """載入 MFCC .npy 檔案。"""

    def __init__(self, paths: list[str]):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        mfcc = np.load(self.paths[idx])  # (94, 39)
        return torch.tensor(mfcc, dtype=torch.float32)


# === 萃取函式 ===

@torch.no_grad()
def extract_cnn_embeddings(model, dataloader: DataLoader) -> np.ndarray:
    """萃取 CNN block4 GAP 後的 embedding (N, 256)。"""
    model.eval()
    all_emb = []
    for X in tqdm(dataloader, desc='CNN embedding'):
        X = X.to(DEVICE)
        x = model.block1(X)
        x = model.block2(x)
        x = model.block3(x)
        x = model.block4(x)          # (batch, 256, 1, 1)
        emb = x.view(x.size(0), -1)  # (batch, 256)
        all_emb.append(emb.numpy())
    return np.concatenate(all_emb, axis=0)


@torch.no_grad()
def extract_lstm_embeddings(model, dataloader: DataLoader) -> np.ndarray:
    """萃取 LSTM attention context 向量 (N, 256)。"""
    model.eval()
    all_emb = []
    for X in tqdm(dataloader, desc='LSTM embedding'):
        X = X.to(DEVICE)
        h, _ = model.lstm(X)             # (batch, 94, 256)
        context, _ = model.attention(h)   # (batch, 256)
        all_emb.append(context.numpy())
    return np.concatenate(all_emb, axis=0)


# === 主程式 ===

def main():
    # 載入 metadata
    df = pd.read_csv(METADATA_PATH)
    print(f'Metadata: {len(df):,} samples')

    # 儲存 embedding_meta.csv（與 npy 行對齊）
    meta_path = EMB_DIR / 'embedding_meta.csv'
    if not meta_path.exists():
        df[['emotion', 'dataset', 'speaker_id']].to_csv(meta_path, index=False)
        print(f'Saved: {meta_path}')
    else:
        print(f'[SKIP] {meta_path.name} already exists')

    # 動態匯入 models（需要 src/ 在 sys.path）
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))
    from models import SERConvNet, SERBiLSTM

    # --- CNN ---
    cnn_out = EMB_DIR / 'cnn_embeddings.npy'
    if cnn_out.exists():
        print(f'[SKIP] {cnn_out.name} already exists')
    else:
        print('\n=== CNN Embedding ===')
        melspec_paths = [str(PROJECT_ROOT / p) for p in df['melspec_path']]
        cnn_ds = MelSpecDataset(melspec_paths)
        cnn_loader = DataLoader(cnn_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        model = SERConvNet(num_classes=6)
        model.load_state_dict(torch.load(CNN_CKPT, map_location='cpu', weights_only=True))
        model.to(DEVICE)

        emb = extract_cnn_embeddings(model, cnn_loader)
        np.save(cnn_out, emb)
        print(f'Saved: {cnn_out} — shape {emb.shape}')
        del model

    # --- LSTM ---
    lstm_out = EMB_DIR / 'lstm_embeddings.npy'
    if lstm_out.exists():
        print(f'[SKIP] {lstm_out.name} already exists')
    else:
        print('\n=== LSTM Embedding ===')
        mfcc_paths = [str(PROJECT_ROOT / p) for p in df['mfcc_path']]
        lstm_ds = MFCCDataset(mfcc_paths)
        lstm_loader = DataLoader(lstm_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        model = SERBiLSTM(input_dim=39, hidden_dim=128, num_layers=2, num_classes=6, dropout=0.3)
        model.load_state_dict(torch.load(LSTM_CKPT, map_location='cpu', weights_only=True))
        model.to(DEVICE)

        emb = extract_lstm_embeddings(model, lstm_loader)
        np.save(lstm_out, emb)
        print(f'Saved: {lstm_out} — shape {emb.shape}')
        del model

    print('\nDone! CNN/LSTM embeddings saved to data/embeddings/')


if __name__ == '__main__':
    main()
