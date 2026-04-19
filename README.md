# 語音情緒辨識（Speech Emotion Recognition）

> 跨語料庫遷移能力研究：Pre-trained Speech Model vs. 傳統特徵方法

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Gradio](https://img.shields.io/badge/Demo-Gradio-purple)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 專案概述

本專案系統性地比較六種機器學習／深度學習方法，在四個英語語音情緒資料集（共 11,318 筆、121 位說話者）上進行六類情緒辨識，並以 **Leave-One-Corpus-Out（LOCO）** 評估各模型的跨語料庫泛化能力。

**核心研究問題：** 特徵表徵能力（Feature Representation）對情緒辨識準確率的影響，是否大於模型架構本身？

**關鍵結論：** Fine-tuned **wav2vec 2.0** 在 5-fold 交叉驗證中達到 **73.9% 準確率**，LOCO 跨語料庫平均達 **52.2%**，均大幅優於傳統 MFCC 特徵方法（~28%）。

---

## 即時 Demo

```bash
cd demo
python app.py
# 開啟瀏覽器：http://127.0.0.1:7860
```

介面支援**上傳音訊檔**與**即時麥克風錄音**，輸出包含情緒類別、信心分數、機率分布圖與 Mel-Spectrogram 視覺化。

> 預錄保險樣本（`demo/samples/*.wav`）取自 TESS 資料集，採 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) 授權，僅供非商業展示用途。詳見 [LICENSE](LICENSE)。

---

## 關鍵結果

### In-Corpus（說話者獨立 5-fold 交叉驗證）

| 模型 | 特徵類型 | 準確率 | F1 Macro |
|------|----------|--------|----------|
| SVM | MFCC 統計特徵 | 49.2% | 48.9% |
| RandomForest | MFCC 統計特徵 | 46.5% | 45.3% |
| XGBoost | MFCC 統計特徵 | 48.7% | 48.1% |
| CNN | Mel-spectrogram | 58.6% | 58.3% |
| Bi-LSTM + Attention | MFCC 時序 | 49.4% | 49.1% |
| **wav2vec 2.0** | **Pre-trained embedding** | **73.9%** | **73.8%** |

### LOCO 跨語料庫泛化（zero-shot cross-corpus）

| 模型 | RAVDESS | CREMA-D | TESS | SAVEE | **平均** |
|------|---------|---------|------|-------|---------|
| CNN | 33.5% | 22.4% | 32.9% | 27.9% | 29.2% |
| Bi-LSTM | 28.3% | 29.2% | 25.7% | 26.9% | 27.5% |
| **wav2vec 2.0** | **64.6%** | **29.1%** | **62.5%** | **52.4%** | **52.2%** |

---

## 核心發現

1. **特徵表徵 > 模型架構：** SVM (49.2%) ≈ Bi-LSTM (49.4%)，兩者同用 MFCC 特徵時表現相近；改用 Mel-spectrogram，CNN 提升至 58.6%；改用 Pre-trained embedding，wav2vec 大幅躍升至 73.9%。特徵的選擇比模型架構影響更大。

2. **跨語料庫泛化是主要挑戰：** 所有模型在 LOCO 測試中準確率大幅下滑。CNN 從 58.6% 降至 29.2%（跌幅 ~30%）；即使是最強的 wav2vec 也從 73.9% 降至 52.2%（跌幅 ~22%），顯示預訓練特徵具備更好的跨域遷移能力。

3. **說話者獨立評估至關重要：** 採用 StratifiedGroupKFold（group=speaker\_id）確保同一說話者不同時出現在訓練集與測試集，避免高估模型泛化能力。

---

## 技術棧

| 類別 | 工具 |
|------|------|
| 語言 | Python 3.10+ |
| 深度學習 | PyTorch 2.x、HuggingFace Transformers |
| 音訊處理 | librosa、soundfile、audiomentations |
| 傳統 ML | scikit-learn、XGBoost |
| 視覺化 | Plotly（無 matplotlib） |
| Demo | Gradio |
| 降維 | UMAP |
| 實驗管理 | Jupyter Notebook |

---

## 專案結構

```
SER_project/
├── data/
│   ├── metadata.csv              # 主索引（11,318 筆，含特徵路徑）
│   ├── raw/                      # 原始音訊（.gitignore 排除）
│   ├── processed/audio_16k/      # 重採樣音訊（.gitignore 排除）
│   ├── features/                 # MFCC / Mel-spectrogram（.gitignore 排除）
│   └── embeddings/               # CNN / LSTM / wav2vec 嵌入向量（.gitignore 排除）
│
├── models/
│   ├── checkpoints/              # 訓練好的模型權重（.gitignore 排除）
│   └── (architecture defined in src/models.py)
│
├── notebooks/
│   ├── 01_eda.ipynb              # 探索性資料分析
│   ├── 02_preprocessing.ipynb   # 音訊前處理
│   ├── 03_feature_extraction.ipynb
│   ├── 04_baseline_ml.ipynb     # SVM / RF / XGBoost
│   ├── 05_train_cnn.ipynb       # CNN 訓練
│   ├── 06_train_lstm.ipynb      # Bi-LSTM 訓練
│   ├── 07_train_wav2vec.ipynb   # wav2vec 2.0 微調
│   ├── 08_cross_corpus_eval.ipynb  # LOCO 評估
│   ├── 09_embedding_analysis.ipynb # Embedding 視覺化分析
│   └── 10_results_summary.ipynb    # 結果總覽
│
├── src/
│   ├── audio_utils.py            # 音訊載入、重採樣、正規化
│   ├── data_utils.py             # Metadata 管理、K-Fold 分割
│   ├── feature_utils.py          # MFCC、Mel-spectrogram 萃取
│   ├── models.py                 # CNN、Bi-LSTM 架構定義
│   ├── plot_config.py            # 統一 Plotly 配色與版面
│   ├── run_preprocessing.py      # 批次音訊重採樣
│   ├── run_feature_extraction.py # 批次特徵萃取
│   ├── run_baseline_ml.py        # 批次訓練 SVM/RF/XGBoost
│   └── run_extract_embeddings.py # 批次萃取 DL 嵌入向量
│
├── demo/
│   ├── app.py                    # Gradio Demo 主程式
│   ├── select_samples.py         # 挑選高信心度保險樣本
│   └── samples/                  # 6 個預錄保險樣本（含於 repo）
│
├── results/
│   ├── baseline_ml/              # SVM/RF/XGBoost 結果 JSON
│   ├── cross_corpus/             # LOCO 結果 JSON（CNN/LSTM/wav2vec）
│   ├── wav2vec_folds/            # wav2vec 各 fold 結果
│   └── figures/                  # 圖表輸出（HTML + PNG）
│
├── requirements.txt
└── README.md
```

---

## 快速開始

### 1. 安裝相依套件

```bash
# 建立虛擬環境（建議）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安裝套件
pip install -r requirements.txt

# PyTorch CPU 版本（本地端 Demo 用）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 2. 取得模型權重

由於檔案過大（~370 MB），`models/checkpoints/wav2vec/wav2vec_fold1_best.pt` 未包含於本 repo。取得方式：

- **目前：** 請聯絡作者（見 repo issue 或 email）索取 fold-1 best checkpoint
- **未來計畫：** 上傳至 [HuggingFace Hub](https://huggingface.co/) 公開取用（規劃中）

檔案取得後請放置到：`models/checkpoints/wav2vec/wav2vec_fold1_best.pt`

### 3. 啟動 Demo

```bash
cd demo
python app.py
# 瀏覽器開啟 http://127.0.0.1:7860
```

### 4. 完整實驗複現

完整實驗需下載原始資料集並依序執行 `notebooks/01` → `10`（部分訓練步驟需 GPU，建議在 Google Colab 執行；所有 notebook 皆已支援 Colab 與本地環境自動偵測）。

---

## 資料集

本專案所有四個資料集統一取自 Kaggle：[Speech Emotion Recognition by Shivam Burnwal](https://www.kaggle.com/code/shivamburnwal/speech-emotion-recognition)

| 資料集 | 原始來源 | 筆數 | 說話者 |
|--------|----------|------|--------|
| RAVDESS | 加拿大 Ryerson 大學 | 1,056 | 24 |
| CREMA-D | 美國演員資料庫 | 7,442 | 91 |
| TESS | 多倫多大學 | 2,400 | 2 |
| SAVEE | 英國薩里大學 | 420 | 4 |

所有音訊統一重採樣至 16kHz、正規化至 [-1, 1]。

---

## 方法論

### 實驗設計

- **情緒類別（6 類）：** angry、disgust、fear、happy、neutral、sad
- **說話者獨立評估：** StratifiedGroupKFold（K=5，group=speaker\_id，stratify=emotion）
- **跨語料庫評估：** Leave-One-Corpus-Out（LOCO）

### 特徵萃取

| 特徵 | 維度 | 用於 |
|------|------|------|
| MFCC 統計特徵 | 234D（均值/標準差） | SVM / RF / XGBoost |
| MFCC 時序 | (94, 39) | Bi-LSTM |
| Mel-spectrogram | (128, 94) | CNN |
| wav2vec 2.0 embedding | 原始波形 → 預訓練特徵 | Fine-tuning |

### 模型架構

- **CNN：** 4 個卷積塊（32→64→128→256 filters）+ Global Average Pooling + 2 層分類器
- **Bi-LSTM：** 2 層雙向 LSTM（hidden=128）+ Bahdanau Self-Attention + 2 層分類器
- **wav2vec 2.0：** `facebook/wav2vec2-base`（95M 參數）+ Projection Head + Softmax 分類

---

## 限制

1. **Fold-1 Embedding 視覺化的方法論限制：** Embedding 分析採用 fold-1 best checkpoint 對全資料萃取，約 80% 樣本為該 fold 的訓練資料，可能使 cluster 品質有所膨脹。主要用於呈現三個模型間的相對差異。

2. **英語資料集限制：** 四個資料集均為英語，結果不能直接推論至中文或其他語言的情緒辨識。

3. **模型參數量不對等：** wav2vec 2.0（95M）遠大於 CNN（~1M）與 LSTM（~500K），比較的是「特徵表徵方式的遷移能力」而非「相同計算量下的效率」。

---

## 未來方向

1. **模型壓縮：** 透過 Knowledge Distillation 將 wav2vec 的知識蒸餾至輕量模型，降低部署成本
2. **多語言擴展：** 引入中文語音情緒資料集，測試模型的跨語言遷移能力
3. **Out-of-fold Embedding：** 採用嚴格的 out-of-fold prediction 產生 embedding，消除 fold-1 方法論限制

---

## 關於

東吳大學「深度學習創新與應用」 課程期末專題

> 本 repo 含完整實驗程式碼與 Gradio Demo；因版權限制，原始音訊資料與模型權重不隨 repo 提供。
