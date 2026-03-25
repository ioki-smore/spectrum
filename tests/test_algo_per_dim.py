"""
各算法独立维度检测可视化（完整端到端）
======================================
使用模拟多维 CPU 数据，完整端到端测试流程：

  评估层级：
    - 点级  (point-level)  ：每个时间点独立判断是否异常
    - 窗口级 (window-level) ：滑动窗口内 ≥70% 点为异常 → 窗口异常

  输出内容：
    1. 每算法独立图（per-algorithm）：
         - 每维度一行：原始信号 + 检测标记（TP/FP）+ 维度贡献分
         - 最后一行：综合异常分数 + 窗口异常带
         - 标题：点级 P/R/F1  |  窗口级 P/R/F1
    2. 集成投票图（Ensemble）：
         - 每维度一行：全模型预测叠加 vs 集成预测
         - 最后一行：投票数时序（显示各阈值线）
    3. 结构化 CSV（test_results/algo_per_dim/results_summary.csv）：
         - 每行：algo, eval_level, precision, recall, f1, tp, fp, tn, fn, threshold, elapsed_s
    4. 文本汇总报告（results_summary.txt）

输出目录：test_results/algo_per_dim/
"""

import sys
import time
import warnings
import math
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import polars as pl
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── CJK 字体 ─────────────────────────────────────────────────────────────────
_CJK = ['PingFang SC', 'Heiti TC', 'STHeiti', 'Arial Unicode MS']
_found = [f for f in matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
          if any(n.lower() in f.lower() for n in _CJK)]
if _found:
    matplotlib.rcParams['font.sans-serif'] = _CJK + ['DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

from models.historical import HistoricalThresholdModel
from models.gsr import GSR
from models.sr import SR
from models.lstm import LSTM
from models.usad import USAD
from models.gsr_ae import GSR_AE

# ─────────────────────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────────────────────

FEATURES    = ['user_pct', 'system_pct', 'load1']
SAVE_DIR    = Path('test_results/algo_per_dim')
N_TRAIN     = 1800
N_TEST      = 900
SEED        = 42

# 压测段（测试集索引）
STRESS_RANGES: List[Tuple[int, int]] = [(150, 270), (380, 500), (680, 830)]

# 窗口级评估参数
WINDOW_SIZE          = 10    # 每个窗口包含的时间点数
WINDOW_ANOMALY_RATIO = 0.70  # 窗口内 ≥70% 点为异常 → 窗口异常

# 集成投票策略
VOTING_STRATEGY = 'majority'   # 'majority' | 'any' | 'all'

# 算法调色板
ALGO_COLORS = {
    'Historical': '#2563eb',
    'GSR':        '#059669',
    'SR':         '#d97706',
    'LSTM':       '#8b5cf6',
    'USAD':       '#db2777',
    'GSR_AE':     '#0891b2',
    'Ensemble':   '#374151',
}

# ─────────────────────────────────────────────────────────────────────────────
# 数据模拟
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(n_train: int = N_TRAIN, n_test: int = N_TEST,
                  seed: int = SEED) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成模拟 CPU 多维时序数据：user_pct / system_pct / load1。"""

    def normal(n: int, s: int = 0) -> np.ndarray:
        r = np.random.RandomState(seed + s)
        user   = np.clip(r.normal(12.0, 4.0,  n), 1.0,  40.0)
        system = np.clip(r.normal(4.0,  1.5,  n), 0.5,  12.0)
        load1  = np.clip(r.normal(0.4,  0.15, n), 0.05,  1.2)
        return np.column_stack([user, system, load1])

    def stress(n: int, s: int = 100) -> np.ndarray:
        r = np.random.RandomState(seed + s)
        user   = np.clip(r.normal(82.0, 6.0, n),  60.0, 99.0)
        system = np.clip(r.normal(12.0, 3.0, n),   5.0, 30.0)
        load1  = np.clip(r.normal(9.5,  1.2, n),   4.0, 16.0)
        return np.column_stack([user, system, load1])

    train  = normal(n_train, s=0)
    test   = normal(n_test,  s=1)
    labels = np.zeros(n_test, dtype=int)

    for s_start, s_end in STRESS_RANGES:
        seg_len = s_end - s_start
        test[s_start:s_end] = stress(seg_len, s=seed + s_start)
        labels[s_start:s_end] = 1

    return train, test, labels


# ─────────────────────────────────────────────────────────────────────────────
# 公共工具
# ─────────────────────────────────────────────────────────────────────────────

def _pad_to(arr: np.ndarray, target: int) -> np.ndarray:
    if len(arr) >= target:
        return arr[:target]
    return np.concatenate([np.zeros(target - len(arr)), arr])


def _pad_2d_to(arr: np.ndarray, target: int) -> np.ndarray:
    if len(arr) >= target:
        return arr[:target]
    pad = np.zeros((target - len(arr), arr.shape[1]))
    return np.vstack([pad, arr])


def to_pl(arr: np.ndarray, features: List[str]) -> pl.DataFrame:
    return pl.DataFrame(dict(zip(features, arr.T)))


def to_pd(arr: np.ndarray, features: List[str]) -> pd.DataFrame:
    return pd.DataFrame(arr, columns=features)


def points_to_windows(arr: np.ndarray, win: int, ratio: float) -> np.ndarray:
    """点级 0/1 → 窗口级 0/1。窗口内异常点占比 ≥ ratio → 1。"""
    n = len(arr)
    n_win = n // win
    if n_win == 0:
        return np.array([], dtype=int)
    trimmed = arr[:n_win * win].reshape(n_win, win)
    return (trimmed.mean(axis=1) >= ratio).astype(int)


def _cm4(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = int((y_true == 0).sum()), 0, 0, 0
    return int(tn), int(fp), int(fn), int(tp)


def eval_pt(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = _cm4(y_true, y_pred)
    return {'precision': p, 'recall': r, 'f1': f, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}


def eval_win(y_true: np.ndarray, y_pred: np.ndarray,
            win: int = WINDOW_SIZE, ratio: float = WINDOW_ANOMALY_RATIO) -> dict:
    wt = points_to_windows(y_true, win, ratio)
    wp = points_to_windows(y_pred, win, ratio)
    if len(wt) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    p = precision_score(wt, wp, zero_division=0)
    r = recall_score(wt, wp, zero_division=0)
    f = f1_score(wt, wp, zero_division=0)
    tn, fp, fn, tp = _cm4(wt, wp)
    return {'precision': p, 'recall': r, 'f1': f, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}


def _status(f1: float) -> str:
    return '✅' if f1 >= 0.80 else ('⚠️' if f1 >= 0.50 else '❌')


# ─────────────────────────────────────────────────────────────────────────────
# 算法运行器（统一返回格式）
# ─────────────────────────────────────────────────────────────────────────────

def _result(name: str, scores: np.ndarray, contribs: np.ndarray,
            threshold: float, model, features: List[str]) -> dict:
    preds = (scores > threshold).astype(int)
    return {'name': name, 'scores': scores, 'contribs': contribs,
            'threshold': threshold, 'predictions': preds,
            'model': model, 'features': features}


def run_historical(train: np.ndarray, test: np.ndarray,
                   features: List[str]) -> dict:
    config = {
        'historical_threshold_percentile': 99.5,
        'historical_smoothing_enabled': True,
        'historical_smoothing_method': 'median',
        'historical_smoothing_window_min': 3,
        'historical_smoothing_window_max': 9,
    }
    model = HistoricalThresholdModel(name='Historical', config=config)
    model.fit(to_pd(train, features))
    scores   = model.predict(to_pd(test, features)).unwrap()
    cr       = model.get_contribution(to_pd(test, features))
    contribs = cr.unwrap() if cr.is_ok() else np.zeros((len(test), len(features)))
    return _result('Historical', scores, contribs, model.auto_detection_threshold, model, features)


def run_gsr(train: np.ndarray, test: np.ndarray,
            features: List[str]) -> dict:
    config = {
        'gsr_window_size': 8, 'batch_size': 256, 'gsr_auto_tune': True,
        'gsr_tune_window_sizes': [4, 8], 'gsr_tune_dev_weights': [1.0, 2.0],
        'gsr_tune_spec_weights': [0.0, 0.1], 'gsr_threshold_margin': 0.9,
    }
    model    = GSR(name='GSR', config=config, input_dim=len(features))
    train_pl = to_pl(train, features)
    test_pl  = to_pl(test, features)
    model.fit(train_pl)
    n        = len(test)
    scores   = _pad_to(model.predict(test_pl).unwrap(), n)
    cr       = model.get_contribution(test_pl)
    contribs = (_pad_2d_to(cr.unwrap(), n) if cr.is_ok() and len(cr.unwrap()) > 0
                else np.zeros((n, len(features))))
    thr = getattr(model, 'auto_threshold', None)
    if thr is None or thr <= 0:
        thr = float(np.percentile(scores[scores > 0], 95)) if (scores > 0).any() else 1.0
    return _result('GSR', scores, contribs, thr, model, features)


def run_sr(train: np.ndarray, test: np.ndarray,
           features: List[str]) -> dict:
    config   = {'window_size': 16, 'batch_size': 256, 'sr_filter_size': 3, 'extend_points': 5}
    model    = SR(name='SR', config=config, input_dim=len(features))
    train_pl = to_pl(train, features)
    test_pl  = to_pl(test, features)
    model.fit(train_pl)
    n        = len(test)
    scores   = _pad_to(model.predict(test_pl).unwrap(), n)
    cr       = model.get_contribution(test_pl)
    if cr.is_ok() and len(cr.unwrap()) > 0:
        c = cr.unwrap()
        contribs = _pad_2d_to(c, n) if c.ndim == 2 else np.zeros((n, len(features)))
    else:
        contribs = np.zeros((n, len(features)))
    return _result('SR', scores, contribs, 3.0, model, features)


def run_lstm(train: np.ndarray, test: np.ndarray,
             features: List[str]) -> dict:
    config = {
        'window_size': 32, 'lstm_hidden_dim': 32, 'lstm_layers': 1,
        'epochs': 20, 'batch_size': 64, 'lstm_error_check_window': 5,
    }
    model    = LSTM(name='LSTM', config=config, input_dim=len(features))
    train_pl = to_pl(train, features)
    test_pl  = to_pl(test, features)
    model.fit(train_pl)
    n        = len(test)
    scores   = _pad_to(model.predict(test_pl).unwrap(), n)
    cr       = model.get_contribution(test_pl)
    if cr.is_ok() and len(cr.unwrap()) > 0:
        c = cr.unwrap()
        contribs = _pad_2d_to(c, n) if c.ndim == 2 else np.zeros((n, len(features)))
    else:
        contribs = np.zeros((n, len(features)))
    ts  = model.predict(train_pl).unwrap()
    thr = max(float(np.percentile(ts, 95)) if len(ts) > 0 else 1e-8, 1e-8)
    return _result('LSTM', scores, contribs, thr, model, features)


def run_usad(train: np.ndarray, test: np.ndarray,
             features: List[str]) -> dict:
    config = {
        'window_size': 32, 'latent_size': 10, 'epochs': 20, 'batch_size': 64,
        'usad_error_check_window': 5,
    }
    model    = USAD(name='USAD', config=config, input_dim=len(features))
    train_pl = to_pl(train, features)
    test_pl  = to_pl(test, features)
    model.fit(train_pl)
    n        = len(test)
    scores   = _pad_to(model.predict(test_pl).unwrap(), n)
    cr       = model.get_contribution(test_pl)
    if cr.is_ok() and len(cr.unwrap()) > 0:
        c = cr.unwrap()
        contribs = _pad_2d_to(c, n) if c.ndim == 2 else np.zeros((n, len(features)))
    else:
        contribs = np.zeros((n, len(features)))
    ts  = model.predict(train_pl).unwrap()
    thr = max(float(np.percentile(ts, 99)) if len(ts) > 0 else 1e-8, 1e-8)
    return _result('USAD', scores, contribs, thr, model, features)


def run_gsr_ae(train: np.ndarray, test: np.ndarray,
               features: List[str]) -> dict:
    config = {
        'gsr_ae_window_size': 32, 'batch_size': 64,
        'gsr_ae_epochs': 30, 'gsr_ae_latent_dim': 8, 'gsr_ae_lr': 1e-3,
        'gsr_ae_sigma': 3.0,
        'gsr_ae_amp_feature_indices': [features.index('load1')],
        'gsr_ae_amp_threshold': 2.0,
        'gsr_ae_amp_check_window': 5,
        'gsr_ae_th_lo_q': 0.05,
        'gsr_ae_suppression_factor': 0.05,
    }
    model    = GSR_AE(name='GSR_AE', config=config, input_dim=len(features))
    train_pl = to_pl(train, features)
    test_pl  = to_pl(test, features)
    model.fit(train_pl)
    n        = len(test)
    scores   = _pad_to(model.predict(test_pl).unwrap(), n)
    cr       = model.get_contribution(test_pl)
    if cr.is_ok() and len(cr.unwrap()) > 0:
        c = cr.unwrap()
        contribs = _pad_2d_to(c, n) if c.ndim == 2 else np.zeros((n, len(features)))
    else:
        contribs = np.zeros((n, len(features)))
    # Use p99 of training scores (suppressed normal data) as threshold
    ts  = model.predict(train_pl).unwrap()
    thr = max(float(np.percentile(ts, 99)) if len(ts) > 0 else 1e-8, 1e-8)
    return _result('GSR_AE', scores, contribs, thr, model, features)


# ─────────────────────────────────────────────────────────────────────────────
# 可视化工具
# ─────────────────────────────────────────────────────────────────────────────

def _ax_clean(ax):
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _draw_stress(ax, t_test, stress_ranges, n_test):
    for s, e in stress_ranges:
        ax.axvspan(t_test[s], t_test[min(e, n_test) - 1], color='#fed7aa', alpha=0.35)


def _draw_train(ax, t_train, n_train):
    ax.axvspan(t_train[0], n_train - 1, color='#d1fae5', alpha=0.20)
    ax.axvline(n_train, color='#9ca3af', lw=1.2, linestyle='--', alpha=0.6)


def _win_bands(win_preds: np.ndarray, win: int, offset: int) -> List[Tuple[int, int]]:
    return [(offset + wi * win, offset + (wi + 1) * win - 1)
            for wi, v in enumerate(win_preds) if v]


# ─────────────────────────────────────────────────────────────────────────────
# 单算法图（点级检测 + 窗口异常带）
# ─────────────────────────────────────────────────────────────────────────────

def plot_algo(train: np.ndarray, test: np.ndarray, labels: np.ndarray,
              result: dict, stress_ranges: List[Tuple[int, int]], save_dir: Path,
              win: int = WINDOW_SIZE, win_ratio: float = WINDOW_ANOMALY_RATIO) -> None:
    name, scores, contribs  = result['name'], result['scores'], result['contribs']
    threshold, preds         = result['threshold'], result['predictions']
    features                 = result['features']
    color                    = ALGO_COLORS.get(name, '#6b7280')

    n_feat  = len(features)
    n_train = len(train)
    n_test  = len(test)
    t_train = np.arange(n_train)
    t_test  = np.arange(n_train, n_train + n_test)

    pt = eval_pt(labels, preds)
    wn = eval_win(labels, preds, win, win_ratio)

    dim_rate, dim_max = [], []
    for i in range(n_feat):
        col = contribs[:, i]
        dim_max.append(float(col.max()))
        if preds.sum() > 0 and contribs.shape[1] > 1:
            ai   = np.where(preds == 1)[0]
            rate = float((np.argmax(contribs[ai], axis=1) == i).sum() / len(ai))
        else:
            rate = 0.0
        dim_rate.append(rate)

    wp_bands = _win_bands(points_to_windows(preds, win, win_ratio), win, n_train)

    fig = plt.figure(figsize=(20, 3.5 * n_feat + 3.5))
    gs  = gridspec.GridSpec(n_feat + 1, 2, figure=fig,
                            width_ratios=[3, 1], height_ratios=[3] * n_feat + [2.5],
                            hspace=0.14, wspace=0.08,
                            left=0.07, right=0.97, top=0.91, bottom=0.06)

    fig.suptitle(
        f'{name}  —  多维 CPU 压测异常检测\n'
        f'点级   P={pt["precision"]:.3f}  R={pt["recall"]:.3f}  F1={pt["f1"]:.3f}'
        f'  TP={pt["tp"]} FP={pt["fp"]} FN={pt["fn"]}  {_status(pt["f1"])}\n'
        f'窗口级  P={wn["precision"]:.3f}  R={wn["recall"]:.3f}  F1={wn["f1"]:.3f}'
        f'  TP={wn["tp"]} FP={wn["fp"]} FN={wn["fn"]}  {_status(wn["f1"])}'
        f'  (w={win}, ratio≥{win_ratio:.0%})',
        fontsize=10, fontweight='bold', color=color, linespacing=1.6
    )

    true_idx = np.where(labels == 1)[0]
    for i, feat in enumerate(features):
        ax_sig  = fig.add_subplot(gs[i, 0])
        ax_cont = fig.add_subplot(gs[i, 1])
        for ax in (ax_sig, ax_cont):
            _ax_clean(ax)

        _draw_train(ax_sig, t_train, n_train)
        _draw_stress(ax_sig, t_test, stress_ranges, n_test)
        ax_sig.plot(t_train, train[:, i], color='#22c55e', lw=0.8, alpha=0.75, label='训练')
        ax_sig.plot(t_test,  test[:, i],  color='#3b82f6', lw=0.8, alpha=0.85, label='测试')
        if len(true_idx):
            ax_sig.scatter(t_test[true_idx], test[true_idx, i],
                           color='#ef4444', s=8, alpha=0.30, marker='x', label='真实异常')
        pred_idx = np.where(preds == 1)[0]
        if len(pred_idx):
            tp_m, fp_m = labels[pred_idx] == 1, labels[pred_idx] == 0
            if tp_m.any():
                ax_sig.scatter(t_test[pred_idx[tp_m]], test[pred_idx[tp_m], i],
                               color=color, s=28, alpha=0.70, marker='o',
                               facecolors='none', linewidths=1.5, label='TP', zorder=5)
            if fp_m.any():
                ax_sig.scatter(t_test[pred_idx[fp_m]], test[pred_idx[fp_m], i],
                               color='#f97316', s=28, alpha=0.70, marker='s',
                               facecolors='none', linewidths=1.5, label='FP', zorder=5)
        ax_sig.set_title(f'{feat}  [贡献率 {dim_rate[i]:.0%}]',
                         fontsize=10, fontweight='bold', pad=3)
        ax_sig.set_ylabel(feat, fontsize=9)
        ax_sig.legend(loc='upper left', fontsize=7, ncol=5, framealpha=0.85)
        ax_sig.grid(True, alpha=0.15)

        _draw_stress(ax_cont, t_test, stress_ranges, n_test)
        ax_cont.fill_between(t_test, contribs[:, i], alpha=0.22, color=color)
        ax_cont.plot(t_test, contribs[:, i], color=color, lw=0.9, alpha=0.85, label=feat)
        ax_cont.set_title(f'贡献分  max={dim_max[i]:.3f}', fontsize=9, pad=3)
        ax_cont.set_ylabel('Contrib', fontsize=8)
        ax_cont.legend(loc='upper left', fontsize=7, framealpha=0.85)
        ax_cont.grid(True, alpha=0.15)

    ax_sc = fig.add_subplot(gs[-1, :])
    _ax_clean(ax_sc)
    _draw_train(ax_sc, t_train, n_train)
    _draw_stress(ax_sc, t_test, stress_ranges, n_test)
    for ws, we in wp_bands:
        ax_sc.axvspan(ws, we, color='#93c5fd', alpha=0.30, zorder=1)

    all_t  = np.concatenate([t_train, t_test])
    padded = np.full(len(all_t), np.nan)
    padded[n_train:] = scores
    ax_sc.fill_between(all_t, padded, alpha=0.15, color=color, zorder=2)
    ax_sc.plot(all_t, padded, color=color, lw=1.0, label='异常分数', zorder=3)
    ax_sc.axhline(threshold, color='#ef4444', lw=1.8, linestyle='--',
                  label=f'阈值={threshold:.4g}', zorder=4)

    tp_i = np.where((preds == 1) & (labels == 1))[0]
    fp_i = np.where((preds == 1) & (labels == 0))[0]
    fn_i = np.where((preds == 0) & (labels == 1))[0]
    if len(tp_i): ax_sc.scatter(t_test[tp_i], scores[tp_i], color='#2563eb', s=25, zorder=5, label=f'TP({len(tp_i)})')
    if len(fp_i): ax_sc.scatter(t_test[fp_i], scores[fp_i], color='#ef4444', s=25, zorder=5, label=f'FP({len(fp_i)})')
    if len(fn_i): ax_sc.scatter(t_test[fn_i], scores[fn_i], color='#f59e0b', s=30, zorder=5, marker='v', label=f'FN({len(fn_i)})')

    win_patch = mpatches.Patch(color='#93c5fd', alpha=0.5, label=f'窗口异常带(w={win})')
    h, lb = ax_sc.get_legend_handles_labels()
    ax_sc.legend(h + [win_patch], lb + [win_patch.get_label()],
                 loc='upper left', fontsize=7.5, ncol=8, framealpha=0.85)
    ax_sc.set_title('综合异常分数  +  窗口异常带（蓝色）', fontsize=10, fontweight='bold', pad=3)
    ax_sc.set_xlabel('时间步', fontsize=9)
    ax_sc.set_ylabel('Score', fontsize=9)
    ax_sc.grid(True, alpha=0.15)

    save_path = save_dir / f'{name}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  [图] {save_path.name}')


# ─────────────────────────────────────────────────────────────────────────────
# 集成投票
# ─────────────────────────────────────────────────────────────────────────────

def ensemble_vote(results: List[dict], strategy: str = VOTING_STRATEGY) -> dict:
    n_models = len(results)
    n_test   = len(results[0]['predictions'])
    n_feat   = results[0]['contribs'].shape[1]
    all_preds = np.stack([r['predictions'] for r in results], axis=0)
    votes     = all_preds.sum(axis=0).astype(float)
    scores    = votes / n_models

    if strategy == 'any':
        thr = 1.0 / n_models
    elif strategy == 'all':
        thr = 1.0 - 1e-9
    else:
        thr = math.ceil(n_models / 2) / n_models - 1e-9

    preds = (scores > thr).astype(int)

    avg_c = np.zeros((n_test, n_feat))
    for r in results:
        c = r['contribs']
        for fi in range(n_feat):
            mx = c[:, fi].max()
            avg_c[:, fi] += c[:, fi] / mx if mx > 0 else c[:, fi]
    avg_c /= n_models

    return {
        'name': f'Ensemble({strategy})',
        'scores': scores, 'contribs': avg_c, 'threshold': thr,
        'predictions': preds, 'model': None,
        'features': results[0]['features'],
        'vote_counts': votes, 'sub_results': results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 集成图（投票数柱状 + 各模型预测叠加）
# ─────────────────────────────────────────────────────────────────────────────

def plot_ensemble(train: np.ndarray, test: np.ndarray, labels: np.ndarray,
                  ens: dict, stress_ranges: List[Tuple[int, int]], save_dir: Path,
                  win: int = WINDOW_SIZE, win_ratio: float = WINDOW_ANOMALY_RATIO) -> None:
    sub     = ens['sub_results']
    name    = ens['name']
    preds   = ens['predictions']
    votes   = ens['vote_counts']
    contribs = ens['contribs']
    features = ens['features']
    color    = ALGO_COLORS.get('Ensemble', '#374151')
    n_subs   = len(sub)

    n_feat  = len(features)
    n_train = len(train)
    n_test  = len(test)
    t_train = np.arange(n_train)
    t_test  = np.arange(n_train, n_train + n_test)

    pt = eval_pt(labels, preds)
    wn = eval_win(labels, preds, win, win_ratio)
    wp_bands = _win_bands(points_to_windows(preds, win, win_ratio), win, n_train)
    sub_colors = [ALGO_COLORS.get(r['name'], '#9ca3af') for r in sub]
    true_idx   = np.where(labels == 1)[0]

    fig = plt.figure(figsize=(20, 3.5 * n_feat + 3.5))
    gs  = gridspec.GridSpec(n_feat + 1, 2, figure=fig,
                            width_ratios=[3, 1], height_ratios=[3] * n_feat + [3],
                            hspace=0.14, wspace=0.08,
                            left=0.07, right=0.97, top=0.91, bottom=0.06)

    fig.suptitle(
        f'{name}  —  多算法投票集成（{n_subs} 模型）\n'
        f'点级   P={pt["precision"]:.3f}  R={pt["recall"]:.3f}  F1={pt["f1"]:.3f}'
        f'  TP={pt["tp"]} FP={pt["fp"]} FN={pt["fn"]}  {_status(pt["f1"])}\n'
        f'窗口级  P={wn["precision"]:.3f}  R={wn["recall"]:.3f}  F1={wn["f1"]:.3f}'
        f'  TP={wn["tp"]} FP={wn["fp"]} FN={wn["fn"]}  {_status(wn["f1"])}'
        f'  (w={win}, ratio≥{win_ratio:.0%})',
        fontsize=10, fontweight='bold', color=color, linespacing=1.6
    )

    for i, feat in enumerate(features):
        ax_sig  = fig.add_subplot(gs[i, 0])
        ax_cont = fig.add_subplot(gs[i, 1])
        for ax in (ax_sig, ax_cont):
            _ax_clean(ax)

        _draw_train(ax_sig, t_train, n_train)
        _draw_stress(ax_sig, t_test, stress_ranges, n_test)
        ax_sig.plot(t_train, train[:, i], color='#22c55e', lw=0.8, alpha=0.65, label='训练')
        ax_sig.plot(t_test,  test[:, i],  color='#9ca3af', lw=0.7, alpha=0.75, label='测试信号')
        if len(true_idx):
            ax_sig.scatter(t_test[true_idx], test[true_idx, i],
                           color='#ef4444', s=6, alpha=0.25, marker='x', label='真实异常')
        for ri, r in enumerate(sub):
            sp = np.where(r['predictions'] == 1)[0]
            if len(sp):
                ax_sig.scatter(t_test[sp], test[sp, i],
                               color=sub_colors[ri], s=12, alpha=0.35,
                               marker='.', label=r['name'], zorder=3)
        ens_idx = np.where(preds == 1)[0]
        if len(ens_idx):
            tp_m = labels[ens_idx] == 1
            fp_m = ~tp_m
            if tp_m.any():
                ax_sig.scatter(t_test[ens_idx[tp_m]], test[ens_idx[tp_m], i],
                               color=color, s=40, alpha=0.85, marker='o',
                               facecolors='none', linewidths=2.0, label='集成TP', zorder=6)
            if fp_m.any():
                ax_sig.scatter(t_test[ens_idx[fp_m]], test[ens_idx[fp_m], i],
                               color='#f97316', s=40, alpha=0.85, marker='s',
                               facecolors='none', linewidths=2.0, label='集成FP', zorder=6)
        ax_sig.set_title(feat, fontsize=10, fontweight='bold', pad=3)
        ax_sig.set_ylabel(feat, fontsize=9)
        ax_sig.legend(loc='upper left', fontsize=6.5, ncol=5, framealpha=0.85)
        ax_sig.grid(True, alpha=0.15)

        _draw_stress(ax_cont, t_test, stress_ranges, n_test)
        ax_cont.fill_between(t_test, contribs[:, i], alpha=0.22, color=color)
        ax_cont.plot(t_test, contribs[:, i], color=color, lw=0.9, alpha=0.85, label=feat)
        ax_cont.set_title('平均归一化贡献分', fontsize=9, pad=3)
        ax_cont.set_ylabel('Contrib', fontsize=8)
        ax_cont.legend(loc='upper left', fontsize=7, framealpha=0.85)
        ax_cont.grid(True, alpha=0.15)

    ax_v = fig.add_subplot(gs[-1, :])
    _ax_clean(ax_v)
    _draw_train(ax_v, t_train, n_train)
    _draw_stress(ax_v, t_test, stress_ranges, n_test)
    for ws, we in wp_bands:
        ax_v.axvspan(ws, we, color='#93c5fd', alpha=0.28, zorder=1)
    bar_c = np.where(preds == 1, color, '#d1d5db')
    ax_v.bar(t_test, votes, color=bar_c, alpha=0.75, width=1.0, zorder=2)

    thr_defs = [
        ('any（≥1票）', 1.0, '#059669'),
        (f'majority（≥{math.ceil(n_subs/2)}票）', float(math.ceil(n_subs / 2)), '#ef4444'),
        (f'all（{n_subs}票）', float(n_subs), '#8b5cf6'),
    ]
    for lbl, tv, lc in thr_defs:
        ax_v.axhline(tv - 0.5, color=lc, lw=1.5, linestyle='--', alpha=0.80, label=lbl)

    ax_v.set_yticks(range(n_subs + 1))
    ax_v.set_ylabel('投票数', fontsize=9)
    ax_v.set_xlabel('时间步', fontsize=9)
    ax_v.set_title(f'各时间步投票数  |  当前策略: {VOTING_STRATEGY}  +  窗口异常带（蓝）',
                   fontsize=10, fontweight='bold', pad=3)
    win_p = mpatches.Patch(color='#93c5fd', alpha=0.5, label=f'窗口异常带(w={win})')
    h, lb = ax_v.get_legend_handles_labels()
    ax_v.legend(h + [win_p], lb + [win_p.get_label()],
                loc='upper left', fontsize=7.5, ncol=5, framealpha=0.85)
    ax_v.grid(True, alpha=0.15)

    save_path = save_dir / 'Ensemble.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  [图] {save_path.name}')


# ─────────────────────────────────────────────────────────────────────────────
# 结构化输出（CSV + 文本报告）
# ─────────────────────────────────────────────────────────────────────────────

def save_results(all_results: List[dict], ens_result: dict,
                 labels: np.ndarray, save_dir: Path,
                 win: int = WINDOW_SIZE, win_ratio: float = WINDOW_ANOMALY_RATIO) -> None:
    """
    输出格式与 results/ 目录保持一致：

      summary.csv          : interval, start_time, end_time, top_k_metrics,
                             is_false_alarm, processed, max_score, mean_score
      {algo}_details.csv   : timestamp, metric, score, is_anomaly
                             [+ {sub_model}_score / {sub_model}_is_anomaly  (仅 Ensemble)]
      algo_metrics.csv     : 点级 + 窗口级 P/R/F1 汇总（测试评估专用）
      results_summary.txt  : 可读报告
    """
    TOP_K = 3
    summary_rows = []

    for r in all_results + [ens_result]:
        name     = r['name']
        scores   = r['scores']
        preds    = r['predictions']
        contribs = r['contribs']
        features = r['features']
        thr      = r['threshold']
        n        = len(scores)
        timestamps = np.arange(n)

        # ── {algo}_details.csv ─────────────────────────────────────────────
        # 每行 = 一个时间步 × 一个特征维度
        # metric = 特征名（user_pct / system_pct / load1）
        # score  = 该维度贡献分（contribs[:, i]）
        # algo_score = 该时间步综合异常分
        # is_anomaly = 该时间步整体是否异常
        sub_sorted = (sorted(r['sub_results'], key=lambda x: x['name'])
                      if 'sub_results' in r else [])
        feat_frames = []
        for fi, feat in enumerate(features):
            df_feat = pd.DataFrame({
                'timestamp':  np.arange(n),
                'metric':     feat,
                'score':      contribs[:, fi],
                'algo_score': scores,
                'is_anomaly': preds.astype(bool),
            })
            for sub in sub_sorted:
                sn = sub['name']
                df_feat[f'{sn}_score']      = sub['scores']
                df_feat[f'{sn}_is_anomaly'] = sub['predictions'].astype(bool)
            feat_frames.append(df_feat)
        det_df = (pd.concat(feat_frames, ignore_index=True)
                    .sort_values(['timestamp', 'metric'])
                    .reset_index(drop=True))

        safe = name.replace('(', '_').replace(')', '').replace(' ', '_')
        det_path = save_dir / f'{safe}_details.csv'
        det_df.to_csv(det_path, index=False, encoding='utf-8-sig')
        print(f'  [CSV] {det_path.name}')

        # ── summary.csv event row ─────────────────────────────────────────
        anom_idx = np.where(preds == 1)[0]
        if len(anom_idx) == 0:
            summary_rows.append({
                'interval': name, 'start_time': None, 'end_time': None,
                'top_k_metrics': '', 'is_false_alarm': False,
                'processed': False, 'max_score': 0.0, 'mean_score': 0.0,
            })
        else:
            s, e = int(anom_idx[0]), int(anom_idx[-1])
            seg  = scores[s:e + 1]
            if contribs is not None and len(contribs) > 0:
                avg_c   = contribs[s:e + 1].mean(axis=0)
                abs_c   = np.abs(avg_c)
                total_c = abs_c.sum()
                if total_c > 0:
                    pct_c = abs_c / total_c * 100.0
                else:
                    pct_c = np.zeros_like(abs_c)
                top_idx = np.argsort(pct_c)[-TOP_K:][::-1]
                top_k   = ';'.join(
                    f'{features[int(i)]}({pct_c[int(i)]:.1f}%)' for i in top_idx
                )
            else:
                top_k = ''
            summary_rows.append({
                'interval': name, 'start_time': s, 'end_time': e,
                'top_k_metrics': top_k, 'is_false_alarm': False,
                'processed': False,
                'max_score': round(float(seg.max()), 6),
                'mean_score': round(float(seg.mean()), 6),
            })

    # ── summary.csv ──────────────────────────────────────────────────────
    sum_path = save_dir / 'summary.csv'
    pd.DataFrame(summary_rows).to_csv(sum_path, index=False, encoding='utf-8-sig')
    print(f'  [CSV] {sum_path.name}')

    # ── algo_metrics.csv (评估指标，测试专用) ─────────────────────────────
    metrics_rows = []
    for r in all_results + [ens_result]:
        name, preds = r['name'], r['predictions']
        elapsed, thr = r.get('elapsed', 0.0), r['threshold']
        for level, m in [('point',  eval_pt(labels, preds)),
                         ('window', eval_win(labels, preds, win, win_ratio))]:
            metrics_rows.append({
                'algo': name, 'eval_level': level,
                'precision': round(m['precision'], 4),
                'recall':    round(m['recall'],    4),
                'f1':        round(m['f1'],        4),
                'tp': m['tp'], 'fp': m['fp'], 'tn': m['tn'], 'fn': m['fn'],
                'threshold': round(thr, 6), 'elapsed_s': round(elapsed, 2),
                'status': _status(m['f1']).replace('✅','PASS').replace('⚠️','WARN').replace('❌','FAIL'),
            })
    mdf = pd.DataFrame(metrics_rows)
    mpath = save_dir / 'algo_metrics.csv'
    mdf.to_csv(mpath, index=False, encoding='utf-8-sig')
    print(f'  [CSV] {mpath.name}')

    # ── results_summary.txt ───────────────────────────────────────────────
    lines = ['=' * 76, '端到端算法评估报告',
             f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
             f'数据集  : 训练={N_TRAIN}  测试={N_TEST}  压测段={STRESS_RANGES}',
             f'特征列  : {FEATURES}',
             f'窗口参数: win={win}  anomaly_ratio≥{win_ratio:.0%}',
             f'投票策略: {VOTING_STRATEGY}', '=' * 76]

    for level in ('point', 'window'):
        sub = mdf[mdf['eval_level'] == level]
        tag = '点级  (point-level)' if level == 'point' else f'窗口级 (w={win}, ratio≥{win_ratio:.0%})'
        lines += [f'\n【{tag}】',
                  f"  {'算法':<22s} {'P':>8s} {'R':>8s} {'F1':>8s}"
                  f"  {'TP':>4s} {'FP':>4s} {'TN':>4s} {'FN':>4s}  状态",
                  '  ' + '─' * 72]
        for _, row in sub.iterrows():
            lines.append(
                f"  {row['algo']:<22s} {row['precision']:>8.4f} {row['recall']:>8.4f}"
                f" {row['f1']:>8.4f}  {row['tp']:>4d} {row['fp']:>4d}"
                f" {row['tn']:>4d} {row['fn']:>4d}  {row['status']}")

    pt_best = mdf[mdf['eval_level'] == 'point'].sort_values('f1', ascending=False).iloc[0]
    wn_best = mdf[mdf['eval_level'] == 'window'].sort_values('f1', ascending=False).iloc[0]
    lines += ['\n' + '=' * 76,
              f"点级最佳  : {pt_best['algo']:22s} F1={pt_best['f1']:.4f}",
              f"窗口级最佳: {wn_best['algo']:22s} F1={wn_best['f1']:.4f}",
              '=' * 76]

    txt_path = save_dir / 'results_summary.txt'
    txt_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'  [TXT] {txt_path.name}')
    print()
    for l in lines:
        print(l)


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

RUNNERS = [
    ('Historical', run_historical),
    ('GSR',        run_gsr),
    ('SR',         run_sr),
    ('LSTM',       run_lstm),
    ('USAD',       run_usad),
    ('GSR_AE',     run_gsr_ae),
]


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print('=' * 76)
    print('端到端算法测试  ——  多维 CPU 压测  ——  点级 + 窗口级 + 投票集成')
    print('=' * 76)
    print(f'窗口参数: w={WINDOW_SIZE}  ratio≥{WINDOW_ANOMALY_RATIO:.0%}  |  投票策略: {VOTING_STRATEGY}')

    train, test, labels = build_dataset()
    n_stress = int(labels.sum())
    print(f'数据集  训练={len(train)}  测试={len(test)}'
          f'  正常={len(labels)-n_stress}  压测={n_stress}')
    print(f'压测段  {STRESS_RANGES}  特征={FEATURES}')

    algo_results: List[dict] = []
    for idx, (key, runner) in enumerate(RUNNERS, 1):
        print(f'\n[{idx}/{len(RUNNERS)}] {key} ...')
        t0 = time.time()
        try:
            result            = runner(train, test, FEATURES)
            result['elapsed'] = time.time() - t0
            pt = eval_pt(labels, result['predictions'])
            wn = eval_win(labels, result['predictions'])
            print(f'  {result["elapsed"]:.1f}s  '
                  f'点级 F1={pt["f1"]:.3f}(P={pt["precision"]:.3f} R={pt["recall"]:.3f}) {_status(pt["f1"])}'
                  f'  |  窗口级 F1={wn["f1"]:.3f}(P={wn["precision"]:.3f} R={wn["recall"]:.3f}) {_status(wn["f1"])}')
            print(f'  阈值={result["threshold"]:.6g}  '
                  f'TP={pt["tp"]} FP={pt["fp"]} FN={pt["fn"]}')
            c = result['contribs']
            for i, feat in enumerate(FEATURES):
                col = c[:, i]
                print(f'    {feat:12s}  max={col.max():.4f}'
                      f'  anom_mean={col[labels==1].mean():.4f}'
                      f'  norm_mean={col[labels==0].mean():.4f}')
            plot_algo(train, test, labels, result, STRESS_RANGES, SAVE_DIR)
            algo_results.append(result)
        except Exception as e:
            import traceback
            print(f'  ❌ 失败: {e}')
            traceback.print_exc()

    print(f'\n[{len(RUNNERS)+1}/{len(RUNNERS)+1}] Ensemble ({VOTING_STRATEGY}) ...')
    ens_result            = ensemble_vote(algo_results, VOTING_STRATEGY)
    ens_result['elapsed'] = 0.0
    pt = eval_pt(labels, ens_result['predictions'])
    wn = eval_win(labels, ens_result['predictions'])
    print(f'  点级 F1={pt["f1"]:.3f}(P={pt["precision"]:.3f} R={pt["recall"]:.3f}) {_status(pt["f1"])}'
          f'  |  窗口级 F1={wn["f1"]:.3f}(P={wn["precision"]:.3f} R={wn["recall"]:.3f}) {_status(wn["f1"])}')
    plot_ensemble(train, test, labels, ens_result, STRESS_RANGES, SAVE_DIR)

    print(f'\n输出结果...')
    save_results(algo_results, ens_result, labels, SAVE_DIR)
    print(f'\n图表目录: {SAVE_DIR.resolve()}')
    print('=' * 76)


if __name__ == '__main__':
    main()
