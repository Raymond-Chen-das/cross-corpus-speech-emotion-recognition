"""
把 fine-tuned wav2vec2 checkpoint（.pt state_dict）轉成標準 HF 模型格式，
並（選擇性）推送到 Hugging Face Model Hub。

此腳本是「本地工具」，在你電腦上跑一次即可，不需要 push 到 Space。

用法：
    1. 先登入：huggingface-cli login   （貼一個有 write 權限的 token）
    2. 把下方 REPO_ID 改成你的 "username/模型名"
    3. python hf_space/convert_to_hf.py
"""

from pathlib import Path

import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

# === 設定（只需改 REPO_ID）===
REPO_ID = "RaymondChendas/wav2vec2-base-ser"   # 你的 username/模型名（模型名可自行更改）
PUSH_TO_HUB = True                            # 設 False 則只在本地產生 model_export/，不上傳

BASE_MODEL = "facebook/wav2vec2-base"
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad"]  # 與訓練時的標籤順序一致（字母序）

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT_PATH = PROJECT_ROOT / "models" / "checkpoints" / "wav2vec" / "wav2vec_fold1_best.pt"
EXPORT_DIR = Path(__file__).resolve().parent / "model_export"


def main():
    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"找不到 checkpoint：{CKPT_PATH}\n"
            "請確認 fold-1 best checkpoint 在這個路徑，或修改 CKPT_PATH。"
        )

    # 1. 用與訓練時完全相同的設定，重建模型架構
    print(f"重建模型架構（{BASE_MODEL}, num_labels=6, classifier_proj_size=256）...")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=6,
        classifier_proj_size=256,
    )

    # 2. 載入 fine-tuned 權重
    print(f"載入 fine-tuned 權重：{CKPT_PATH.name}")
    state = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)  # strict=True：keys 必須完全吻合，吻合才代表轉檔正確

    # 3. 寫入情緒標籤，讓模型「自我描述」（之後 from_pretrained 就自帶 id2label）
    model.config.id2label = {i: e for i, e in enumerate(EMOTIONS)}
    model.config.label2id = {e: i for i, e in enumerate(EMOTIONS)}

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(BASE_MODEL)

    # 4. 存成標準 HF 格式（config.json + model.safetensors + preprocessor_config.json）
    print(f"存成 HF 格式到本地：{EXPORT_DIR}")
    model.save_pretrained(EXPORT_DIR)
    feature_extractor.save_pretrained(EXPORT_DIR)

    # 5. 上傳到 Hub
    if PUSH_TO_HUB:
        print(f"推送到 Hub：{REPO_ID}（需先完成 huggingface-cli login）")
        model.push_to_hub(REPO_ID)
        feature_extractor.push_to_hub(REPO_ID)
        print("完成！模型已上傳到 Hub。")
    else:
        print(f"完成！已存到本地 {EXPORT_DIR}（未上傳，PUSH_TO_HUB=False）。")


if __name__ == "__main__":
    main()
