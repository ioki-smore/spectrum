"""
CPU 异常检测端到端测试
=====================
1. 模拟 CPU 正常 + 压测数据
2. 分别测试 HistoricalThreshold / GSR / SR 三个算法
3. 测试多算法投票集成
4. 窗口级评估（窗口内 >70% 异常 → 窗口异常）
5. 输出点级 + 窗口级 P/R/F1 对比表
"""

import sys
import numpy as np
import pandas as pd
import polars as pl
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

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

@dataclass
class TestConfig:
    """端到端测试配置，所有阈值 / 窗口参数集中管理。"""
    # ── 数据 ──────────────────────────────────────────────────────────────
    features: List[str] = field(default_factory=lambda: ['user_pct', 'system_pct', 'load1'])
    n_train: int = 1800           # 训练集大小
    n_test: int = 900             # 测试集大小
    seed: int = 42

    # ── 阈值算法 ──────────────────────────────────────────────────────────
    # 千分之一 = p99.9：训练数据中仅 0.1% 超过此阈值
    threshold_percentile: float = 99.9

    # ── 窗口评估 ──────────────────────────────────────────────────────────
    window_size: int = 10          # 窗口大小（点数）
    window_anomaly_ratio: float = 0.7  # 窗口内 >70% 点异常 → 窗口异常

    # ── 集成投票 ──────────────────────────────────────────────────────────
    # 'majority': 多数投票（>= ceil(n/2) 个模型判异常 → 异常）
    # 'any':      任一模型判异常 → 异常
    # 'all':      所有模型都判异常 → 异常
    voting_strategy: str = 'majority'

    # ── 输出 ──────────────────────────────────────────────────────────────
    save_dir: str = 'test_results'


# ─────────────────────────────────────────────────────────────────────────────
# 1. 数据模拟器
# ─────────────────────────────────────────────────────────────────────────────

class CPUDataSimulator:
    """模拟 CPU 指标时序数据。"""

    @staticmethod
    def normal_segment(n: int, seed: int = 0) -> np.ndarray:
        rng = np.random.RandomState(seed)
        user   = np.clip(rng.normal(12.0, 4.0, n), 1.0, 40.0)
        system = np.clip(rng.normal(4.0,  1.5, n), 0.5, 12.0)
        load1  = np.clip(rng.normal(0.4,  0.15, n), 0.05, 1.2)
        return np.column_stack([user, system, load1])

    @staticmethod
    def stress_segment(n: int, seed: int = 100) -> np.ndarray:
        rng = np.random.RandomState(seed)
        user   = np.clip(rng.normal(82.0, 6.0,  n), 60.0, 99.0)
        system = np.clip(rng.normal(12.0, 3.0,  n), 5.0,  30.0)
        load1  = np.clip(rng.normal(9.5,  1.2,  n), 4.0,  16.0)
        return np.column_stack([user, system, load1])

    @classmethod
    def build_dataset(cls, cfg: TestConfig):
        rng = np.random.RandomState(cfg.seed)
        train = cls.normal_segment(cfg.n_train, seed=cfg.seed)

        stress_ranges = [(150, 270), (380, 500), (680, 830)]
        test   = cls.normal_segment(cfg.n_test, seed=cfg.seed + 1)
        labels = np.zeros(cfg.n_test, dtype=int)

        for s, e in stress_ranges:
            seg_len = e - s
            test[s:e] = cls.stress_segment(seg_len, seed=cfg.seed + s)
            labels[s:e] = 1

        return train, test, labels, stress_ranges


# ─────────────────────────────────────────────────────────────────────────────
# 2. 窗口级评估
# ─────────────────────────────────────────────────────────────────────────────

def points_to_windows(point_preds: np.ndarray, window_size: int,
                      anomaly_ratio: float = 0.7) -> np.ndarray:
    """
    将点级预测聚合为窗口级预测。

    Args:
        point_preds: (N,) 的 0/1 数组
        window_size: 窗口大小
        anomaly_ratio: 窗口内异常点占比超过此值 → 窗口判异常

    Returns:
        (n_windows,) 的 0/1 数组
    """
    n = len(point_preds)
    n_windows = n // window_size
    if n_windows == 0:
        return np.array([], dtype=int)

    trimmed = point_preds[:n_windows * window_size]
    reshaped = trimmed.reshape(n_windows, window_size)
    ratios = reshaped.mean(axis=1)
    return (ratios >= anomaly_ratio).astype(int)


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算 P/R/F1/CM。"""
    if len(y_true) == 0 or len(y_pred) == 0:
        return {'precision': 0, 'recall': 0, 'f1': 0, 'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return {'precision': p, 'recall': r, 'f1': f,
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)}


# ─────────────────────────────────────────────────────────────────────────────
# 3. 算法封装（统一接口：返回点级 scores + predictions）
# ─────────────────────────────────────────────────────────────────────────────

def run_historical(train: np.ndarray, test: np.ndarray,
                   features: List[str], cfg: TestConfig) -> dict:
    """运行 HistoricalThresholdModel (simple p99.9)。"""
    config = {
        'historical_threshold_percentile': cfg.threshold_percentile,
    }
    model = HistoricalThresholdModel(name='Historical', config=config)
    model.fit(pd.DataFrame(train, columns=features))
    scores = model.predict(pd.DataFrame(test, columns=features)).unwrap()
    threshold = model.auto_detection_threshold
    preds = (scores > threshold).astype(int)
    return {'name': 'Historical', 'scores': scores, 'predictions': preds,
            'threshold': threshold, 'model': model}


def run_gsr(train: np.ndarray, test: np.ndarray,
            features: List[str], cfg: TestConfig) -> dict:
    """运行 GSR 模型。"""
    n_feat = len(features)
    config = {
        'gsr_window_size': 8,
        'batch_size': 256,
        'gsr_auto_tune': True,
        'gsr_tune_window_sizes': [4, 8],
        'gsr_tune_dev_weights': [1.0, 2.0],
        'gsr_tune_spec_weights': [0.0, 0.1],
        'gsr_threshold_margin': 0.9,
    }
    model = GSR(name='GSR', config=config, input_dim=n_feat)
    train_pl = pl.DataFrame(dict(zip(features, train.T)))
    test_pl  = pl.DataFrame(dict(zip(features, test.T)))

    model.fit(train_pl)
    scores_raw = model.predict(test_pl).unwrap()

    # GSR 输出长度 = n - window_size + 1，前面补 0 对齐到 n
    n_test = len(test)
    if len(scores_raw) < n_test:
        pad = np.zeros(n_test - len(scores_raw))
        scores = np.concatenate([pad, scores_raw])
    else:
        scores = scores_raw[:n_test]

    # 使用 auto_threshold
    threshold = getattr(model, 'auto_threshold', None)
    if threshold is None or threshold <= 0:
        threshold = float(np.percentile(scores[scores > 0], 95)) if (scores > 0).any() else 1.0
    preds = (scores > threshold).astype(int)
    return {'name': 'GSR', 'scores': scores, 'predictions': preds,
            'threshold': threshold, 'model': model}


def run_sr(train: np.ndarray, test: np.ndarray,
           features: List[str], cfg: TestConfig) -> dict:
    """运行 SR (Spectral Residual) 模型。"""
    n_feat = len(features)
    config = {
        'window_size': 16,
        'batch_size': 256,
        'sr_filter_size': 3,
        'extend_points': 5,
    }
    model = SR(name='SR', config=config, input_dim=n_feat)
    train_pl = pl.DataFrame(dict(zip(features, train.T)))
    test_pl  = pl.DataFrame(dict(zip(features, test.T)))

    model.fit(train_pl)
    scores_raw = model.predict(test_pl).unwrap()

    n_test = len(test)
    if len(scores_raw) < n_test:
        pad = np.zeros(n_test - len(scores_raw))
        scores = np.concatenate([pad, scores_raw])
    else:
        scores = scores_raw[:n_test]

    # SR z-score 阈值：3σ
    threshold = 3.0
    preds = (scores > threshold).astype(int)
    return {'name': 'SR', 'scores': scores, 'predictions': preds,
            'threshold': threshold, 'model': model}


def run_lstm(train: np.ndarray, test: np.ndarray,
             features: List[str], cfg: TestConfig) -> dict:
    """运行 LSTM Autoencoder 模型。"""
    n_feat = len(features)
    window_size = 32
    config = {
        'window_size': window_size,
        'lstm_hidden_dim': 32,
        'lstm_layers': 1,
        'epochs': 20,
        'batch_size': 64,
        'lstm_error_check_window': 5,
    }
    model = LSTM(name='LSTM', config=config, input_dim=n_feat)
    train_pl = pl.DataFrame(dict(zip(features, train.T)))
    test_pl  = pl.DataFrame(dict(zip(features, test.T)))

    model.fit(train_pl)
    scores_raw = model.predict(test_pl).unwrap()

    n_test = len(test)
    if len(scores_raw) < n_test:
        pad = np.zeros(n_test - len(scores_raw))
        scores = np.concatenate([pad, scores_raw])
    else:
        scores = scores_raw[:n_test]

    # 自动阈值：训练集分数的 p95
    train_scores = model.predict(train_pl).unwrap()
    if len(train_scores) > 0:
        threshold = float(np.percentile(train_scores, 95))
    else:
        threshold = float(np.percentile(scores[scores > 0], 95)) if (scores > 0).any() else 1.0
    threshold = max(threshold, 1e-8)
    preds = (scores > threshold).astype(int)
    return {'name': 'LSTM', 'scores': scores, 'predictions': preds,
            'threshold': threshold, 'model': model}


def run_usad(train: np.ndarray, test: np.ndarray,
             features: List[str], cfg: TestConfig) -> dict:
    """运行 USAD 模型。"""
    n_feat = len(features)
    window_size = 32
    config = {
        'window_size': window_size,
        'latent_size': 10,
        'epochs': 20,
        'batch_size': 64,
    }
    model = USAD(name='USAD', config=config, input_dim=n_feat)
    train_pl = pl.DataFrame(dict(zip(features, train.T)))
    test_pl  = pl.DataFrame(dict(zip(features, test.T)))

    model.fit(train_pl)
    scores_raw = model.predict(test_pl).unwrap()

    n_test = len(test)
    if len(scores_raw) < n_test:
        pad = np.zeros(n_test - len(scores_raw))
        scores = np.concatenate([pad, scores_raw])
    else:
        scores = scores_raw[:n_test]

    # 自动阈值：训练集分数的 p95
    train_scores = model.predict(train_pl).unwrap()
    if len(train_scores) > 0:
        threshold = float(np.percentile(train_scores, 95))
    else:
        threshold = float(np.percentile(scores[scores > 0], 95)) if (scores > 0).any() else 1.0
    threshold = max(threshold, 1e-8)
    preds = (scores > threshold).astype(int)
    return {'name': 'USAD', 'scores': scores, 'predictions': preds,
            'threshold': threshold, 'model': model}


def run_gsr_ae(train: np.ndarray, test: np.ndarray,
               features: List[str], cfg: TestConfig) -> dict:
    """运行 GSR_AE (GSR + CNN Autoencoder) 模型。"""
    n_feat = len(features)
    window_size = 32
    config = {
        'gsr_ae_window_size': window_size,
        'batch_size': 64,
        'gsr_ae_epochs': 20,
        'gsr_ae_latent_dim': 16,
        'gsr_ae_lr': 1e-3,
    }
    model = GSR_AE(name='GSR_AE', config=config, input_dim=n_feat)
    train_pl = pl.DataFrame(dict(zip(features, train.T)))
    test_pl  = pl.DataFrame(dict(zip(features, test.T)))

    model.fit(train_pl)
    scores_raw = model.predict(test_pl).unwrap()

    n_test = len(test)
    if len(scores_raw) < n_test:
        pad = np.zeros(n_test - len(scores_raw))
        scores = np.concatenate([pad, scores_raw])
    else:
        scores = scores_raw[:n_test]

    # 使用模型自带阈值
    threshold = getattr(model, 'th_hi', None)
    if threshold is None or threshold <= 0:
        train_scores = model.predict(train_pl).unwrap()
        if len(train_scores) > 0:
            threshold = float(np.percentile(train_scores, 95))
        else:
            threshold = float(np.percentile(scores[scores > 0], 95)) if (scores > 0).any() else 1.0
    threshold = max(threshold, 1e-8)
    preds = (scores > threshold).astype(int)
    return {'name': 'GSR_AE', 'scores': scores, 'predictions': preds,
            'threshold': threshold, 'model': model}


# ─────────────────────────────────────────────────────────────────────────────
# 4. 集成投票
# ─────────────────────────────────────────────────────────────────────────────

def ensemble_vote(results: List[dict], strategy: str = 'majority') -> np.ndarray:
    """
    多模型投票集成。

    Args:
        results: 每个算法的 run_* 返回值
        strategy: 'majority' / 'any' / 'all'

    Returns:
        (N,) 集成预测
    """
    all_preds = np.stack([r['predictions'] for r in results], axis=0)  # (n_models, N)
    n_models = all_preds.shape[0]

    if strategy == 'any':
        return (all_preds.sum(axis=0) >= 1).astype(int)
    elif strategy == 'all':
        return (all_preds.sum(axis=0) >= n_models).astype(int)
    else:  # majority
        return (all_preds.sum(axis=0) >= np.ceil(n_models / 2)).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# 5. 可视化
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(test: np.ndarray, labels: np.ndarray,
                    results: List[dict], ensemble_preds: np.ndarray,
                    stress_ranges: list, cfg: TestConfig,
                    point_table: List[dict], window_table: List[dict]) -> None:
    """生成多算法对比可视化。"""
    save_path = Path(cfg.save_dir) / 'cpu_e2e_comparison.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)

    all_names = [r['name'] for r in results] + ['Ensemble']
    all_preds = [r['predictions'] for r in results] + [ensemble_preds]
    all_scores = [r['scores'] for r in results]
    n_algo = len(all_names)
    n_test = len(test)
    t = np.arange(n_test)

    colors = ['#2563eb', '#059669', '#d97706', '#8b5cf6']
    fig, axes = plt.subplots(n_algo + 1, 1, figsize=(18, 3.0 * (n_algo + 1)),
                             sharex=True, gridspec_kw={'hspace': 0.18})

    # ── 第一行：原始 CPU 信号 + 压测区 ────────────────────────────────────
    ax0 = axes[0]
    ax0.set_facecolor('white')
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    for s, e in stress_ranges:
        ax0.axvspan(s, e - 1, color='#fed7aa', alpha=0.4)
    ax0.plot(t, test[:, 0], color='#2563eb', lw=0.7, alpha=0.85, label='user_pct')
    ax0.plot(t, test[:, 1], color='#059669', lw=0.7, alpha=0.85, label='system_pct')
    ax0.set_title('CPU 指标信号（测试集）', fontsize=10, fontweight='bold')
    ax0.set_ylabel('使用率 %', fontsize=9)
    ax0.legend(loc='upper left', fontsize=8, ncol=3)
    ax0.grid(True, alpha=0.2)

    # ── 每个算法一行：异常分数 + 预测标记 ─────────────────────────────────
    for i, (name, preds) in enumerate(zip(all_names, all_preds)):
        ax = axes[i + 1]
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for s, e in stress_ranges:
            ax.axvspan(s, e - 1, color='#fed7aa', alpha=0.35)

        # 对非集成算法画异常分数
        if i < len(all_scores):
            scores = all_scores[i]
            ax.fill_between(t, scores, alpha=0.15, color=colors[i % len(colors)])
            ax.plot(t, scores, color=colors[i % len(colors)], lw=0.8, alpha=0.8, label='异常分数')
            thr = results[i]['threshold']
            ax.axhline(thr, color='#ef4444', lw=1.5, linestyle='--', label=f'阈值={thr:.2f}')

        # 预测 vs 真实标记
        tp_idx = np.where((preds == 1) & (labels == 1))[0]
        fp_idx = np.where((preds == 1) & (labels == 0))[0]
        fn_idx = np.where((preds == 0) & (labels == 1))[0]

        if i < len(all_scores):
            scores_plot = all_scores[i]
            if len(tp_idx): ax.scatter(t[tp_idx], scores_plot[tp_idx], color='#2563eb', s=20, zorder=5, label=f'TP({len(tp_idx)})')
            if len(fp_idx): ax.scatter(t[fp_idx], scores_plot[fp_idx], color='#ef4444', s=20, zorder=5, label=f'FP({len(fp_idx)})')
            if len(fn_idx): ax.scatter(t[fn_idx], scores_plot[fn_idx], color='#f59e0b', s=25, zorder=5, marker='v', label=f'FN({len(fn_idx)})')
        else:
            # 集成：画投票条形
            ax.fill_between(t, preds, alpha=0.3, color='#8b5cf6', label='集成预测')
            ax.fill_between(t, labels * 0.5, alpha=0.2, color='#ef4444', label='真实标签')

        # 从 table 获取指标
        pt_row = next((r for r in point_table if r['算法'] == name), None)
        wn_row = next((r for r in window_table if r['算法'] == name), None)
        title_extra = ''
        if pt_row:
            title_extra += f'  点级 F1={pt_row["f1"]:.3f}'
        if wn_row:
            title_extra += f'  窗口级 F1={wn_row["f1"]:.3f}'

        ax.set_title(f'{name}{title_extra}', fontsize=10, fontweight='bold')
        ax.set_ylabel('Score', fontsize=9)
        ax.legend(loc='upper left', fontsize=7.5, ncol=6, framealpha=0.85)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel('时间步', fontsize=9)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"对比图已保存: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg = TestConfig()

    print("=" * 75)
    print("CPU 异常检测端到端测试  —  多算法 + 窗口评估 + 投票集成")
    print("=" * 75)

    # ── 生成数据 ─────────────────────────────────────────────────────────────
    train, test, labels, stress_ranges = CPUDataSimulator.build_dataset(cfg)
    n_stress = int(labels.sum())
    n_normal = len(labels) - n_stress

    print(f"\n配置")
    print(f"  阈值模式         : percentile (p{cfg.threshold_percentile})")
    print(f"  窗口大小         : {cfg.window_size} 点/窗口")
    print(f"  窗口异常比例阈值 : {cfg.window_anomaly_ratio * 100:.0f}%")
    print(f"  投票策略         : {cfg.voting_strategy}")

    print(f"\n数据集")
    print(f"  训练集 : {len(train)} 条")
    print(f"  测试集 : {len(test)} 条（正常 {n_normal} + 压测 {n_stress}）")
    print(f"  压测段 : {stress_ranges}")

    # ── 运行各算法 ───────────────────────────────────────────────────────────
    print(f"\n{'─' * 75}")
    print(f"运行各算法")
    print(f"{'─' * 75}")

    algo_results = []
    runners = [
        ('Historical',  f'HistoricalThresholdModel (p{cfg.threshold_percentile})', run_historical),
        ('GSR',         'GSR (Global Spectral Residual)',                           run_gsr),
        ('SR',          'SR (Spectral Residual)',                                   run_sr),
        ('LSTM',        'LSTM Autoencoder',                                         run_lstm),
        ('USAD',        'USAD (UnSupervised Anomaly Detection)',                    run_usad),
        ('GSR_AE',      'GSR_AE (GSR + CNN Autoencoder)',                           run_gsr_ae),
    ]
    n_algo = len(runners)

    for idx, (key, desc, runner) in enumerate(runners, 1):
        print(f"\n[{idx}/{n_algo}] {desc} ...")
        try:
            result = runner(train, test, cfg.features, cfg)
            algo_results.append(result)
            if key == 'Historical':
                print(f"      阈值: { {k: f'{v:.4f}' for k, v in result['model'].thresholds.items()} }")
            else:
                print(f"      阈值: {result['threshold']:.4f}")
        except Exception as e:
            print(f"      ❌ 失败: {e}")
            import traceback
            traceback.print_exc()

    # ── 集成投票 ─────────────────────────────────────────────────────────────
    print(f"\n[Ensemble] 投票集成 (strategy={cfg.voting_strategy}) ...")
    ensemble_preds = ensemble_vote(algo_results, cfg.voting_strategy)

    # ── 点级评估 ─────────────────────────────────────────────────────────────
    print(f"\n{'═' * 75}")
    print(f"【点级评估】")
    print(f"{'═' * 75}")

    point_table = []
    for r in algo_results:
        m = eval_metrics(labels, r['predictions'])
        m['算法'] = r['name']
        point_table.append(m)
    m_ens = eval_metrics(labels, ensemble_preds)
    m_ens['算法'] = 'Ensemble'
    point_table.append(m_ens)

    _print_table(point_table)

    # ── 窗口级评估 ───────────────────────────────────────────────────────────
    print(f"\n{'═' * 75}")
    print(f"【窗口级评估】窗口={cfg.window_size}  异常比例阈值={cfg.window_anomaly_ratio:.0%}")
    print(f"{'═' * 75}")

    win_labels = points_to_windows(labels, cfg.window_size, cfg.window_anomaly_ratio)

    window_table = []
    for r in algo_results:
        win_preds = points_to_windows(r['predictions'], cfg.window_size, cfg.window_anomaly_ratio)
        m = eval_metrics(win_labels, win_preds)
        m['算法'] = r['name']
        window_table.append(m)
    win_ens = points_to_windows(ensemble_preds, cfg.window_size, cfg.window_anomaly_ratio)
    m_ens_w = eval_metrics(win_labels, win_ens)
    m_ens_w['算法'] = 'Ensemble'
    window_table.append(m_ens_w)

    _print_table(window_table)

    n_win = len(win_labels)
    n_win_anom = int(win_labels.sum())
    print(f"\n  窗口总数: {n_win}（正常 {n_win - n_win_anom} + 压测 {n_win_anom}）")

    # ── 投票详情 ─────────────────────────────────────────────────────────────
    print(f"\n{'═' * 75}")
    print(f"【投票详情】")
    print(f"{'═' * 75}")

    all_preds_stack = np.stack([r['predictions'] for r in algo_results], axis=0)
    vote_counts = all_preds_stack.sum(axis=0)
    for v in range(len(algo_results) + 1):
        n_pts = (vote_counts == v).sum()
        pct = n_pts / len(vote_counts) * 100
        print(f"  {v}/{len(algo_results)} 个模型判异常: {n_pts:>5d} 点 ({pct:5.1f}%)")

    # 各模型间一致性
    print(f"\n  算法间一致性（点级）:")
    names = [r['name'] for r in algo_results]
    for i in range(len(algo_results)):
        for j in range(i + 1, len(algo_results)):
            agree = (algo_results[i]['predictions'] == algo_results[j]['predictions']).mean()
            print(f"    {names[i]} vs {names[j]}: {agree:.1%} 一致")

    # ── 可视化 ───────────────────────────────────────────────────────────────
    plot_comparison(test, labels, algo_results, ensemble_preds,
                    stress_ranges, cfg, point_table, window_table)

    # ── 最终汇总 ─────────────────────────────────────────────────────────────
    best_pt = max(point_table, key=lambda x: x['f1'])
    best_wn = max(window_table, key=lambda x: x['f1'])

    print(f"\n{'═' * 75}")
    print(f"测试完成")
    print(f"  点级最佳:   {best_pt['算法']}  F1={best_pt['f1']:.4f}")
    print(f"  窗口级最佳: {best_wn['算法']}  F1={best_wn['f1']:.4f}")
    print(f"{'═' * 75}")


def _print_table(rows: List[dict]) -> None:
    """打印结果表格。"""
    header = f"  {'算法':<18s} {'Precision':>9s} {'Recall':>8s} {'F1':>8s}  {'TP':>4s} {'FP':>4s} {'TN':>4s} {'FN':>4s}  状态"
    print(header)
    print(f"  {'─' * 76}")
    for r in rows:
        f1 = r['f1']
        status = "✅" if f1 >= 0.80 else ("⚠️" if f1 >= 0.50 else "❌")
        print(f"  {r['算法']:<18s} {r['precision']:>9.4f} {r['recall']:>8.4f} {f1:>8.4f}"
              f"  {r['tp']:>4d} {r['fp']:>4d} {r['tn']:>4d} {r['fn']:>4d}  {status}")


if __name__ == '__main__':
    main()
