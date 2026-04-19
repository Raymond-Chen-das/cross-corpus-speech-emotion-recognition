"""
統一圖表配色與樣式模組

為所有 notebook (NB09, NB10) 與 demo 提供一致的 plotly 配色與版面設定。
使用 colorblind-friendly palette（基於 Plotly Safe 色盤微調）。
"""

# ============================================================
#  情緒配色（6 類，colorblind-friendly）
# ============================================================

EMOTION_COLORS = {
    'angry':   '#EE6677',   # 紅粉（比純紅更易區分）
    'disgust': '#AA3377',   # 紫紅
    'fear':    '#CCBB44',   # 黃綠
    'happy':   '#228833',   # 綠
    'neutral': '#4477AA',   # 藍
    'sad':     '#66CCEE',   # 青
}

EMOTION_ORDER = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']


# ============================================================
#  資料集配色（4 個語料庫）
# ============================================================

DATASET_COLORS = {
    'RAVDESS': '#EE6677',
    'CREMA-D': '#4477AA',
    'TESS':    '#228833',
    'SAVEE':   '#CCBB44',
}

DATASET_ORDER = ['RAVDESS', 'CREMA-D', 'TESS', 'SAVEE']


# ============================================================
#  模型配色（DL 三模型 + baseline）
# ============================================================

MODEL_COLORS = {
    'CNN':      '#4477AA',
    'LSTM':     '#EE6677',
    'wav2vec2': '#228833',
    'SVM':      '#CCBB44',
    'RF':       '#AA3377',
    'LR':       '#66CCEE',
}


# ============================================================
#  字體與版面設定
# ============================================================

FONT_SIZES = {
    'title': 18,
    'axis_title': 14,
    'tick': 12,
    'legend': 12,
    'annotation': 11,
}

# 常用圖表尺寸
SINGLE_FIG_SIZE = {'width': 800, 'height': 500}
TRIPLE_FIG_SIZE = {'width': 1400, 'height': 500}
QUAD_FIG_SIZE = {'width': 1200, 'height': 1000}


def apply_common_layout(fig, title=None, **kwargs):
    """套用統一的 plotly layout 設定。

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
    title : str, optional
    **kwargs : 額外傳入 fig.update_layout 的參數
    """
    layout_args = dict(
        template='plotly_white',
        font=dict(size=FONT_SIZES['tick']),
        title=dict(
            text=title,
            font=dict(size=FONT_SIZES['title']),
        ) if title else None,
        legend=dict(
            font=dict(size=FONT_SIZES['legend']),
            itemsizing='constant',
        ),
    )
    layout_args.update(kwargs)
    fig.update_layout(**layout_args)

    fig.update_xaxes(title_font=dict(size=FONT_SIZES['axis_title']),
                     tickfont=dict(size=FONT_SIZES['tick']))
    fig.update_yaxes(title_font=dict(size=FONT_SIZES['axis_title']),
                     tickfont=dict(size=FONT_SIZES['tick']))
    return fig


def get_emotion_colors(emotions=None):
    """回傳情緒配色 dict（可選子集）。"""
    if emotions is None:
        return EMOTION_COLORS.copy()
    return {e: EMOTION_COLORS[e] for e in emotions if e in EMOTION_COLORS}


def get_dataset_colors(datasets=None):
    """回傳資料集配色 dict（可選子集）。"""
    if datasets is None:
        return DATASET_COLORS.copy()
    return {d: DATASET_COLORS[d] for d in datasets if d in DATASET_COLORS}


def get_model_colors(models=None):
    """回傳模型配色 dict（可選子集）。"""
    if models is None:
        return MODEL_COLORS.copy()
    return {m: MODEL_COLORS[m] for m in models if m in MODEL_COLORS}