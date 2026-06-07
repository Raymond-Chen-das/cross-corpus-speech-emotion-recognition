---
title: 語音情緒辨識 Speech Emotion Recognition
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.9.0
app_file: app.py
pinned: false
license: mit
---

# 語音情緒辨識 Demo（Speech Emotion Recognition）

使用 fine-tuned **wav2vec 2.0** 即時辨識 6 類語音情緒：
憤怒、厭惡、恐懼、快樂、中性、悲傷。

- **上傳音訊** 或 **即時麥克風錄音**
- 輸出：情緒類別、各情緒信心分數、機率分布圖、Mel-Spectrogram
- 模型在 RAVDESS + CREMA-D + TESS + SAVEE 四個英語語料庫上以說話者獨立 5-fold 訓練，
  平均準確率 73.9%

> 模型來源由環境變數 `MODEL_ID` 指定（預設見 `app.py`）。
> 完整研究程式碼與報告：見專案 GitHub repo。

*東吳大學「深度學習創新與應用」期末專題*
