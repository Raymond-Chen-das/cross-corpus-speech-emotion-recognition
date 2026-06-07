# HF Spaces 部署指南（Model Hub 方案）

> 架構：fine-tuned 模型放 **HF Model Hub**（一個 model repo），Space 用 `from_pretrained` 載入。
> Space 本身很輕（只有程式碼 + 396KB 樣本），模型則成為可重用的公開模型。
>
> 全程約 20–30 分鐘。需要你親自做的是「登入 / 上傳」這幾步（用到你的帳號憑證）。

---

## 檔案清單（我已準備好）

| 檔案 | 用途 | 去哪 |
|------|------|------|
| `convert_to_hf.py` | 把 361MB 的 `.pt` 轉成 HF 格式並上傳模型 | **本地跑**，不進 Space |
| `app.py` | Gradio 介面（從 Hub 載模型） | → Space |
| `requirements.txt` | 套件（已釘實測版本） | → Space |
| `README.md` | Space 卡片（含 SDK 設定） | → Space |
| `samples/` | 6 個保險樣本 | → Space |
| `model_export/` | 轉檔後的本地模型（步驟 2 產生，361MB） | **不要**進 Space 或 GitHub |

---

## 前置：你需要準備的

1. **Hugging Face 帳號**：https://huggingface.co/join （免費）
2. **Write token**：https://huggingface.co/settings/tokens → New token → 權限選 **Write** → 複製備用
3. 把你的 **username** 告訴我（或自己替換下面的 `RaymondChendas`）

---

## 步驟 1：登入 Hugging Face

```powershell
# 確保有 CLI（transformers 已含 huggingface_hub，這行確保是最新）
pip install -U huggingface_hub

# 登入：貼上剛剛的 Write token
huggingface-cli login
```

---

## 步驟 2：轉檔 + 上傳模型到 Model Hub

1. 打開 `hf_space/convert_to_hf.py`，把第一行設定改成你的：
   ```python
   REPO_ID = "RaymondChendas/wav2vec2-base-ser"   # ← 改這裡
   ```
2. 在專案根目錄執行：
   ```powershell
   python hf_space/convert_to_hf.py
   ```
   它會：重建模型架構 → 載入 fold-1 權重 → 寫入情緒標籤 → 存成 HF 格式 → 上傳到你的 Model Hub。
   成功的話，你的模型會出現在 `https://huggingface.co/RaymondChendas/wav2vec2-base-ser`。

> 若只想先在本地驗證、不上傳：把腳本裡 `PUSH_TO_HUB = False`，會只產生 `hf_space/model_export/`。

---

## 步驟 3：設定 app.py 的模型來源

打開 `hf_space/app.py`，把這行的預設值改成你的 model repo：

```python
MODEL_ID = os.environ.get("MODEL_ID", "RaymondChendas/wav2vec2-base-ser")
```

（或者不改程式碼，改用步驟 4 建好 Space 後，到 Space → Settings → Variables 新增 `MODEL_ID`。）

---

## 步驟 4：建立 Space 並上傳程式

**4a. 在網頁建立 Space**
- 到 https://huggingface.co/new-space
- Owner 選你、名稱例如 `speech-emotion-recognition`
- SDK 選 **Gradio**、硬體選 **CPU basic（free）**、可見性選 **Public**
- 建立後會得到一個空的 git repo

**4b. 把程式推上去**（只推這 4 樣，**不要**推 `model_export/`、`convert_to_hf.py`、`DEPLOY.md`）

```powershell
# 把 Space clone 到專案外面的暫存資料夾
cd ..
git clone https://huggingface.co/spaces/RaymondChendas/speech-emotion-recognition
cd speech-emotion-recognition

# 從專案複製 4 樣 payload（請依你的實際路徑調整來源）
copy "..\SER_project\hf_space\app.py" .
copy "..\SER_project\hf_space\requirements.txt" .
copy "..\SER_project\hf_space\README.md" .
xcopy "..\SER_project\hf_space\samples" ".\samples\" /E /I

git add .
git commit -m "Deploy SER wav2vec2 Gradio demo"
git push
```

推上去後，Space 會自動開始 build。第一次 build 約 3–8 分鐘（裝套件）。

---

## 步驟 5：驗證

- 打開 `https://huggingface.co/spaces/RaymondChendas/speech-emotion-recognition`
- 等 build 完成（右上角狀態變 **Running**）
- 用「上傳音訊檔」→ 選一個保險樣本 → 按「辨識情緒」，確認結果正確
- 再測「即時錄音」（瀏覽器會要麥克風權限）

> **報告當天**：免費 Space 閒置會休眠。提前 5 分鐘打開網址，讓它從休眠喚醒（冷啟動約 30 秒~1 分），暖機完再上台。本地 Gradio 仍是離線備案。

---

## 疑難排解

| 症狀 | 處理 |
|------|------|
| Space build 失敗，說 `sdk_version` 不支援 | 把 `README.md` frontmatter 的 `sdk_version` 改成 HF 提示的鄰近版本 |
| build 時某套件裝不起來 | 把 `requirements.txt` 對應套件的 `==x.y.z` 放寬成 `>=x.y`（最後手段） |
| Space 載模型報 401 / 找不到 | 確認 model repo 是 **Public**，且 `MODEL_ID` 拼字正確 |
| 不小心把 361MB `model_export/` 推進 Space | `git rm -r --cached model_export && git commit && git push`，並確認沒被 LFS 追蹤 |

---

*此指南對應 `hf_space/` 內的部署檔案。模型權重來源：`models/checkpoints/wav2vec/wav2vec_fold1_best.pt`。*
