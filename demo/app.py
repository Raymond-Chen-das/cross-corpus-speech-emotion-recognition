"""
SER Demo — 即時語音情緒辨識（Gradio 介面）

使用 wav2vec2 fine-tuned fold-1 best checkpoint 進行 6 類情緒預測。
支援上傳音訊檔案與即時麥克風錄音。

用法：
    cd demo
    python app.py
"""

from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import plotly.graph_objects as go
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

# === 路徑與常數 ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT_PATH = PROJECT_ROOT / 'models' / 'checkpoints' / 'wav2vec' / 'wav2vec_fold1_best.pt'
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

DEVICE = torch.device('cpu')  # Demo 用 CPU 即可


# === 載入模型（啟動時一次性載入） ===
def load_model():
    """載入 wav2vec2 fine-tuned 模型。"""
    print('載入模型中...')
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base')

    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        'facebook/wav2vec2-base',
        num_labels=6,
        classifier_proj_size=256,
    )
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    print('模型載入完成！')
    return model, feature_extractor


MODEL, FEATURE_EXTRACTOR = load_model()


# === 音訊前處理 ===
def preprocess_audio(audio_path_or_tuple):
    """將輸入音訊轉換為模型所需格式。

    Parameters
    ----------
    audio_path_or_tuple : str 或 tuple
        Gradio 音訊元件的輸出：檔案路徑 (str) 或 (sample_rate, numpy_array)
    """
    if isinstance(audio_path_or_tuple, tuple):
        sr, audio = audio_path_or_tuple
        audio = audio.astype(np.float32)
        # 若為 stereo 取第一聲道
        if audio.ndim > 1:
            audio = audio[:, 0]
        # 正規化到 [-1, 1]
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / max(abs(audio.max()), abs(audio.min()))
        # Resample 到 16kHz
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
    """對輸入音訊進行情緒預測。

    Returns
    -------
    tuple: (label_dict, plotly_fig, result_text)
    """
    if audio_input is None:
        return None, None, "請上傳音訊或使用麥克風錄音"

    # 前處理
    audio = preprocess_audio(audio_input)

    # Feature extraction
    inputs = FEATURE_EXTRACTOR(
        audio, sampling_rate=TARGET_SR,
        return_tensors='pt', padding=False,
    )
    input_values = inputs['input_values'].to(DEVICE)

    # 推論
    outputs = MODEL(input_values)
    logits = outputs.logits[0]
    probs = torch.softmax(logits, dim=0).cpu().numpy()

    # 結果
    pred_idx = int(probs.argmax())
    pred_emotion = EMOTIONS[pred_idx]
    confidence = float(probs[pred_idx])

    # Gradio label 格式：{label: confidence}
    label_dict = {EMOTION_LABELS_ZH[e]: float(probs[i]) for i, e in enumerate(EMOTIONS)}

    # Plotly 機率分布圖
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
    """取得 demo/samples/ 中的預錄樣本路徑。"""
    if not SAMPLES_DIR.exists():
        return []
    samples = sorted(SAMPLES_DIR.glob('*.wav'))
    return [str(p) for p in samples]


# === Gradio 介面 ===
def build_interface():
    """建構 Gradio Demo 介面。"""
    sample_paths = get_sample_paths()

    with gr.Blocks(
        title='SER Demo — 語音情緒辨識',
    ) as demo:
        gr.Markdown(
            '# 語音情緒辨識 Demo\n'
            '使用 **wav2vec2** fine-tuned 模型即時辨識 6 類情緒：'
            '憤怒、厭惡、恐懼、快樂、中性、悲傷\n\n'
            '> 模型在 RAVDESS + CREMA-D + TESS + SAVEE 四個英語語料庫上訓練'
        )

        with gr.Tabs():
            # === Tab 1: 上傳音訊 ===
            with gr.Tab('上傳音訊檔'):
                with gr.Row():
                    with gr.Column(scale=1):
                        upload_input = gr.Audio(
                            label='上傳音訊（.wav / .mp3）',
                            type='filepath',
                        )
                        upload_btn = gr.Button('辨識情緒', variant='primary')

                        # 預錄樣本選單
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

            # === Tab 2: 即時錄音 ===
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
    demo.launch(
        server_name='127.0.0.1',
        server_port=7860,
        max_file_size='10mb',
        share=False,
        theme=gr.themes.Soft(),
    )

    # 若 port 7860 被佔用，改用：
    #   python demo/app.py  (自動找空閒 port)
    # 或設定環境變數：
    #   GRADIO_SERVER_PORT=7861 python demo/app.py
