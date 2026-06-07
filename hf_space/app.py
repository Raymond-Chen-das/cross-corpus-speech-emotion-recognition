"""
SER Demo — 即時語音情緒辨識（Gradio 介面，Hugging Face Spaces 版）

與本地版的差異：
- 模型直接從 HF Model Hub 以 from_pretrained 載入（不再讀本地 .pt）
- theme 設定在 gr.Blocks（正確位置），launch() 交給 Spaces 處理 host/port

模型來源由環境變數 MODEL_ID 控制；若未設定，使用下方預設值。
"""

import os
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import plotly.graph_objects as go
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

# === 模型來源 ===
# 部署時：把預設值改成你的 "username/模型名"，或在 Space → Settings → Variables 設 MODEL_ID
MODEL_ID = os.environ.get("MODEL_ID", "RaymondChendas/wav2vec2-base-ser")

SAMPLES_DIR = Path(__file__).resolve().parent / 'samples'

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
EMOTION_LABELS_ZH = {
    'angry': '憤怒 Angry',
    'disgust': '厭惡 Disgust',
    'fear': '恐懼 Fear',
    'happy': '快樂 Happy',
    'neutral': '中性 Neutral',
    'sad': '悲傷 Sad',
}
EMOTION_COLORS = {
    'angry': '#EE6677',
    'disgust': '#AA3377',
    'fear': '#CCBB44',
    'happy': '#228833',
    'neutral': '#4477AA',
    'sad': '#66CCEE',
}

TARGET_SR = 16000
MAX_LENGTH_SEC = 3.0
MAX_LENGTH_SAMPLES = int(TARGET_SR * MAX_LENGTH_SEC)

DEVICE = torch.device('cpu')  # Spaces 免費 tier 為 CPU


# === 載入模型（啟動時一次性載入）===
def load_model():
    """從 HF Model Hub 載入 fine-tuned wav2vec2 模型。"""
    print(f'載入模型中：{MODEL_ID}')
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID)
    model.to(DEVICE)
    model.eval()
    print('模型載入完成！')
    return model, feature_extractor


MODEL, FEATURE_EXTRACTOR = load_model()


# === 音訊前處理 ===
def preprocess_audio(audio_path_or_tuple):
    """將輸入音訊轉換為模型所需格式。"""
    if isinstance(audio_path_or_tuple, tuple):
        sr, audio = audio_path_or_tuple
        audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio[:, 0]
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / max(abs(audio.max()), abs(audio.min()))
        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    else:
        audio, _ = librosa.load(audio_path_or_tuple, sr=TARGET_SR)

    # Pad / Truncate 到 3 秒（center crop + center pad）
    if len(audio) > MAX_LENGTH_SAMPLES:
        start = (len(audio) - MAX_LENGTH_SAMPLES) // 2
        audio = audio[start:start + MAX_LENGTH_SAMPLES]
    elif len(audio) < MAX_LENGTH_SAMPLES:
        pad_total = MAX_LENGTH_SAMPLES - len(audio)
        audio = np.pad(audio, (pad_total // 2, pad_total - pad_total // 2))

    return audio


# === 推論 ===
@torch.no_grad()
def predict_emotion(audio_input):
    """對輸入音訊進行情緒預測。"""
    if audio_input is None:
        return None, None, "請上傳音訊或使用麥克風錄音"

    audio = preprocess_audio(audio_input)

    inputs = FEATURE_EXTRACTOR(
        audio, sampling_rate=TARGET_SR,
        return_tensors='pt', padding=False,
    )
    input_values = inputs['input_values'].to(DEVICE)

    outputs = MODEL(input_values)
    logits = outputs.logits[0]
    probs = torch.softmax(logits, dim=0).cpu().numpy()

    pred_idx = int(probs.argmax())
    pred_emotion = EMOTIONS[pred_idx]
    confidence = float(probs[pred_idx])

    label_dict = {EMOTION_LABELS_ZH[e]: float(probs[i]) for i, e in enumerate(EMOTIONS)}

    fig = go.Figure(go.Bar(
        x=[EMOTION_LABELS_ZH[e] for e in EMOTIONS],
        y=[float(probs[i]) for i in range(len(EMOTIONS))],
        marker_color=[EMOTION_COLORS[e] for e in EMOTIONS],
        text=[f'{float(probs[i]):.1%}' for i in range(len(EMOTIONS))],
        textposition='outside',
    ))
    fig.update_layout(
        title=dict(
            text=f'預測結果：{EMOTION_LABELS_ZH[pred_emotion]}（信心 {confidence:.1%}）',
            font=dict(size=18),
        ),
        yaxis=dict(title='機率', range=[0, 1], tickformat='.0%'),
        xaxis=dict(title=''),
        template='plotly_white',
        height=400,
        margin=dict(t=60, b=40),
    )

    result_text = f"**{EMOTION_LABELS_ZH[pred_emotion]}** — 信心分數 {confidence:.1%}"

    return label_dict, fig, result_text


# === Mel-Spectrogram 視覺化 ===
def visualize_melspec(audio_input):
    """產生輸入音訊的 Mel-Spectrogram 視覺化。"""
    if audio_input is None:
        return None

    audio = preprocess_audio(audio_input)

    mel = librosa.feature.melspectrogram(
        y=audio, sr=TARGET_SR, n_mels=128, fmax=8000,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig = go.Figure(go.Heatmap(
        z=mel_db,
        colorscale='Viridis',
        colorbar=dict(title='dB'),
    ))
    fig.update_layout(
        title='Mel-Spectrogram',
        xaxis=dict(title='Time Frame'),
        yaxis=dict(title='Mel Band'),
        template='plotly_white',
        height=300,
        margin=dict(t=40, b=40),
    )
    return fig


# === 整合推論 + 視覺化 ===
def full_predict(audio_input):
    """合併情緒預測與 Mel-Spectrogram 視覺化。"""
    label_dict, prob_fig, result_text = predict_emotion(audio_input)
    mel_fig = visualize_melspec(audio_input)
    return label_dict, prob_fig, mel_fig, result_text


# === 收集預錄樣本 ===
def get_sample_paths():
    """取得 samples/ 中的預錄樣本路徑。"""
    if not SAMPLES_DIR.exists():
        return []
    return [str(p) for p in sorted(SAMPLES_DIR.glob('*.wav'))]


# === Gradio 介面 ===
def build_interface():
    """建構 Gradio Demo 介面。"""
    sample_paths = get_sample_paths()

    with gr.Blocks(
        title='SER Demo — 語音情緒辨識',
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            '# 語音情緒辨識 Demo\n'
            '使用 **wav2vec2** fine-tuned 模型即時辨識 6 類情緒：'
            '憤怒、厭惡、恐懼、快樂、中性、悲傷\n\n'
            '> 模型在 RAVDESS + CREMA-D + TESS + SAVEE 四個英語語料庫上訓練'
        )

        with gr.Tabs():
            with gr.Tab('上傳音訊檔'):
                with gr.Row():
                    with gr.Column(scale=1):
                        upload_input = gr.Audio(
                            label='上傳音訊（.wav / .mp3）',
                            type='filepath',
                        )
                        upload_btn = gr.Button('辨識情緒', variant='primary')

                        if sample_paths:
                            gr.Markdown('### 預錄保險樣本')
                            sample_dropdown = gr.Dropdown(
                                choices=sample_paths,
                                label='選擇預錄樣本',
                                interactive=True,
                            )
                            sample_dropdown.change(
                                fn=lambda x: x,
                                inputs=sample_dropdown,
                                outputs=upload_input,
                            )

                    with gr.Column(scale=2):
                        upload_result = gr.Markdown('等待輸入...')
                        upload_labels = gr.Label(label='情緒機率分布', num_top_classes=6)
                        upload_prob_plot = gr.Plot(label='機率分布圖')
                        upload_mel_plot = gr.Plot(label='Mel-Spectrogram')

                upload_btn.click(
                    fn=full_predict,
                    inputs=upload_input,
                    outputs=[upload_labels, upload_prob_plot, upload_mel_plot, upload_result],
                )

            with gr.Tab('即時錄音'):
                with gr.Row():
                    with gr.Column(scale=1):
                        mic_input = gr.Audio(
                            label='點擊錄音（建議 2~3 秒）',
                            sources=['microphone'],
                            type='numpy',
                        )
                        mic_btn = gr.Button('辨識情緒', variant='primary')

                    with gr.Column(scale=2):
                        mic_result = gr.Markdown('等待錄音...')
                        mic_labels = gr.Label(label='情緒機率分布', num_top_classes=6)
                        mic_prob_plot = gr.Plot(label='機率分布圖')
                        mic_mel_plot = gr.Plot(label='Mel-Spectrogram')

                mic_btn.click(
                    fn=full_predict,
                    inputs=mic_input,
                    outputs=[mic_labels, mic_prob_plot, mic_mel_plot, mic_result],
                )

        gr.Markdown(
            '---\n'
            '**模型資訊**：wav2vec2-base fine-tuned, 6 類情緒, '
            'StratifiedGroupKFold 5-fold 平均準確率 73.9%\n\n'
            '*東吳大學「深度學習創新與應用」期末專題*'
        )

    return demo


if __name__ == '__main__':
    demo = build_interface()
    demo.launch()
