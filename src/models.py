"""
DL 模型定義

CNN (Mel-spectrogram) 和 Bi-LSTM + Attention (MFCC) 的架構定義。
wav2vec2 使用 HuggingFace 預訓練模型，在 notebook 中直接建構。

所有模型供 Colab notebook 和 LOCO 跨語料庫實驗共用。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
#  CNN for Mel-spectrogram (1, 128, 94)
# ============================================================

class SERConvNet(nn.Module):
    """輕量 CNN — 4 個 conv block + global average pooling。

    輸入：(batch, 1, 128, 94) — Mel-spectrogram 灰階影像
    輸出：(batch, num_classes)
    """

    def __init__(self, num_classes: int = 6) -> None:
        super().__init__()

        # Block 1: (1, 128, 94) → (32, 64, 47)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
        )

        # Block 2: (32, 64, 47) → (64, 32, 23)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
        )

        # Block 3: (64, 32, 23) → (128, 16, 11)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
        )

        # Block 4: (128, 16, 11) → (256, 1, 1)
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.classifier(x)


# ============================================================
#  Self-Attention layer
# ============================================================

class Attention(nn.Module):
    """Additive self-attention（Bahdanau style）。

    輸入：(batch, seq_len, hidden_size)
    輸出：context (batch, hidden_size), weights (batch, seq_len)
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        score = self.v(torch.tanh(self.W(h)))       # (batch, seq, 1)
        alpha = F.softmax(score, dim=1)              # (batch, seq, 1)
        context = (alpha * h).sum(dim=1)             # (batch, hidden)
        return context, alpha.squeeze(-1)


# ============================================================
#  Bi-LSTM + Attention for MFCC (94, 39)
# ============================================================

class SERBiLSTM(nn.Module):
    """2 層 Bi-LSTM + Self-Attention。

    輸入：(batch, 94, 39) — MFCC 時間序列
    輸出：(batch, num_classes)
    """

    def __init__(
        self,
        input_dim: int = 39,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 6,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = Attention(hidden_dim * 2)  # bidirectional → 256
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)                          # (batch, 94, 256)
        context, _attn = self.attention(h)            # (batch, 256)
        return self.classifier(context)


# ============================================================
#  Early Stopping
# ============================================================

class EarlyStopping:
    """監控 validation loss，超過 patience 個 epoch 無改善則停止。"""

    def __init__(self, patience: int = 7, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: float | None = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> None:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

    def reset(self) -> None:
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
