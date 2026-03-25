"""
CPU 指标阈值算法测试
====================
模拟正常 CPU 采集数据 + 压测场景，验证 HistoricalThresholdModel 能否检测出压测异常。

场景设计：
  - 训练集：正常运行（低 CPU 占用）
  - 测试集：正常段 + 多段压测（高 CPU 占用）

指标：user_pct, system_pct, load1（三特征）
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── CJK 字体（macOS） ────────────────────────────────────────────────────────
_CJK = ['PingFang SC', 'Heiti TC', 'STHeiti', 'Arial Unicode MS']
_found = [f for f in matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
          if any(n.lower() in f.lower() for n in _CJK)]
if _found:
    matplotlib.rcParams['font.sans-serif'] = _CJK + ['DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

from models.historical import HistoricalThresholdModel


# ─────────────────────────────────────────────────────────────────────────────
# 1. 数据模拟器
# ─────────────────────────────────────────────────────────────────────────────

class CPUDataSimulator:
    """模拟 CPU 指标时序数据（10s 采样间隔）。"""

    FEATURES = ['user_pct', 'system_pct', 'load1']

    @staticmethod
    def normal_segment(n: int, seed: int = 0) -> np.ndarray:
        """正常负载段：低 CPU、低负载、有轻微随机波动。"""
        rng = np.random.RandomState(seed)
        user   = np.clip(rng.normal(12.0, 4.0, n), 1.0, 40.0)
        system = np.clip(rng.normal(4.0,  1.5, n), 0.5, 12.0)
        load1  = np.clip(rng.normal(0.4,  0.15, n), 0.05, 1.2)
        return np.column_stack([user, system, load1])

    @staticmethod
    def stress_segment(n: int, seed: int = 100) -> np.ndarray:
        """压测负载段：CPU 使用率骤升、load1 大幅上涨。"""
        rng = np.random.RandomState(seed)
        user   = np.clip(rng.normal(82.0, 6.0,  n), 60.0, 99.0)
        system = np.clip(rng.normal(12.0, 3.0,  n), 5.0,  30.0)
        load1  = np.clip(rng.normal(9.5,  1.2,  n), 4.0,  16.0)
        return np.column_stack([user, system, load1])

    @classmethod
    def build_dataset(cls, seed: int = 42):
        """
        构造训练集 + 测试集。

        训练集：1800 条正常数据（~5 小时，10s 间隔）
        测试集：正常段 + 3 段压测 + 正常段，共 900 条

        压测注入位置（测试集内的索引范围）：
          [150, 270)   →  2 分钟压测
          [380, 500)   →  2 分钟压测
          [680, 830)   →  2.5 分钟压测
        """
        rng = np.random.RandomState(seed)

        # ── 训练集 ──────────────────────────────────────────────────────────
        train = cls.normal_segment(1800, seed=seed)

        # ── 测试集 ──────────────────────────────────────────────────────────
        stress_ranges = [(150, 270), (380, 500), (680, 830)]
        n_test = 900
        test   = cls.normal_segment(n_test, seed=seed + 1)
        labels = np.zeros(n_test, dtype=int)

        for s, e in stress_ranges:
            seg_len = e - s
            test[s:e] = cls.stress_segment(seg_len, seed=seed + s)
            labels[s:e] = 1

        return train, test, labels, stress_ranges


# ─────────────────────────────────────────────────────────────────────────────
# 2. 评估函数
# ─────────────────────────────────────────────────────────────────────────────

def to_df(arr: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(arr, columns=CPUDataSimulator.FEATURES)


def run_threshold_test(train: np.ndarray, test: np.ndarray, labels: np.ndarray,
                       config: dict | None = None) -> dict:
    """训练 HistoricalThresholdModel 并在测试集上评估。"""
    if config is None:
        config = {
            'historical_smoothing_method': 'median',
            'historical_smoothing_window_min': 3,
            'historical_smoothing_window_max': 9,
            'historical_robust_max_percentile': 95.0,
            'historical_stationary_margin': 1.10,
            'historical_mad_multiplier': 3.5,
            'historical_trending_k_base': 1.2,
            'historical_trending_k_max': 4.0,
            'historical_ac1_low_threshold': 0.2,
            'historical_ac1_high_threshold': 0.6,
            'historical_score_tier1': 0.1,
            'historical_score_tier2': 0.3,
            'historical_score_tier1_weight': 0.5,
            'historical_score_tier3_weight': 1.5,
        }

    model = HistoricalThresholdModel(name='CPU_Threshold', config=config)

    fit_res = model.fit(to_df(train))
    if fit_res.is_err():
        raise RuntimeError(f"fit() 失败: {fit_res.err_value}")

    pred_res = model.predict(to_df(test))
    if pred_res.is_err():
        raise RuntimeError(f"predict() 失败: {pred_res.err_value}")

    scores      = pred_res.unwrap()
    threshold   = model.auto_detection_threshold          # 固定 1.0（归一化）
    predictions = (scores > threshold).astype(int)

    precision = precision_score(labels, predictions, zero_division=0)
    recall    = recall_score(labels, predictions, zero_division=0)
    f1        = f1_score(labels, predictions, zero_division=0)
    cm        = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        'model': model,
        'scores': scores,
        'predictions': predictions,
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'thresholds': model.thresholds,          # 每个特征的原始阈值
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. 可视化
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(train: np.ndarray, test: np.ndarray, labels: np.ndarray,
                 result: dict, stress_ranges: list,
                 save_path: str = 'test_results/cpu_threshold_test.png') -> None:
    """生成 4 行图（3 特征 + 异常分数面板）。"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    features      = CPUDataSimulator.FEATURES
    scores        = result['scores']
    predictions   = result['predictions']
    model         = result['model']
    det_threshold = result['threshold']

    n_train = len(train)
    n_test  = len(test)
    train_t = np.arange(n_train)
    test_t  = np.arange(n_train, n_train + n_test)

    true_idx = np.where(labels == 1)[0]
    pred_idx = np.where(predictions == 1)[0]
    tp_idx   = np.where((predictions == 1) & (labels == 1))[0]
    fp_idx   = np.where((predictions == 1) & (labels == 0))[0]
    fn_idx   = np.where((predictions == 0) & (labels == 1))[0]

    n_rows = len(features) + 1
    height_ratios = [3] * len(features) + [2]
    fig, axes = plt.subplots(n_rows, 1, figsize=(18, 3.5 * len(features) + 3),
                             sharex=True,
                             gridspec_kw={'height_ratios': height_ratios, 'hspace': 0.10})

    p = result['precision']
    r = result['recall']
    f = result['f1']
    fig.suptitle(
        f'CPU 阈值算法测试 — HistoricalThresholdModel\n'
        f'精确率 P={p:.3f}  召回率 R={r:.3f}  F1={f:.3f}  '
        f'TP={result["tp"]}  FP={result["fp"]}  FN={result["fn"]}',
        fontsize=13, fontweight='bold'
    )

    feature_units = {'user_pct': '%', 'system_pct': '%', 'load1': ''}

    for i, feat in enumerate(features):
        ax = axes[i]
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 训练区背景
        ax.axvspan(train_t[0], n_train, color='#d1fae5', alpha=0.30, label='训练区')
        ax.axvline(n_train, color='#6b7280', lw=1.2, linestyle='--', alpha=0.7)

        # 信号曲线
        ax.plot(train_t, train[:, i], color='#22c55e', lw=0.7, alpha=0.75, label='正常（训练）')
        ax.plot(test_t,  test[:, i],  color='#2563eb', lw=0.7, alpha=0.85, label='测试')

        # 特征阈值线
        if feat in model.thresholds:
            thr = model.thresholds[feat]
            ax.axhline(thr, color='#ef4444', lw=1.8, linestyle='--',
                       label=f'阈值 {thr:.2f}{feature_units[feat]}')
        if feat in model.feature_stats:
            med = model.feature_stats[feat]['median']
            ax.axhline(med, color='#f59e0b', lw=1.2, linestyle=':', alpha=0.8,
                       label=f'中位数 {med:.2f}')

        # 压测区背景（橙色）
        for s, e in stress_ranges:
            ax.axvspan(test_t[s], test_t[e - 1], color='#fed7aa', alpha=0.35)

        # 真实异常标注
        if len(true_idx):
            ax.scatter(test_t[true_idx], test[true_idx, i],
                       color='#ef4444', s=12, alpha=0.5, marker='x',
                       label='真实压测点', zorder=4)

        # 预测异常标注
        if len(pred_idx):
            ax.scatter(test_t[pred_idx], test[pred_idx, i],
                       color='#8b5cf6', s=35, alpha=0.55, marker='o',
                       facecolors='none', linewidths=1.5, label='预测异常', zorder=5)

        unit = f' ({feature_units[feat]})' if feature_units[feat] else ''
        ax.set_title(f'{feat}{unit}', fontsize=10, fontweight='bold', pad=3)
        ax.set_ylabel(feat, fontsize=9)
        ax.legend(loc='upper left', fontsize=7.5, ncol=5, framealpha=0.85)
        ax.grid(True, alpha=0.20)

    # ── 异常分数面板 ─────────────────────────────────────────────────────────
    ax_s = axes[-1]
    ax_s.set_facecolor('white')
    ax_s.spines['top'].set_visible(False)
    ax_s.spines['right'].set_visible(False)

    # 对齐坐标：训练区填 NaN
    all_t = np.concatenate([train_t, test_t])
    padded = np.full(len(all_t), np.nan)
    padded[n_train:] = scores

    ax_s.axvspan(train_t[0], n_train, color='#d1fae5', alpha=0.25)
    ax_s.axvline(n_train, color='#6b7280', lw=1.2, linestyle='--', alpha=0.7)

    for s, e in stress_ranges:
        ax_s.axvspan(test_t[s], test_t[e - 1], color='#fed7aa', alpha=0.35)

    ax_s.fill_between(all_t, padded, alpha=0.20, color='#8b5cf6')
    ax_s.plot(all_t, padded, color='#8b5cf6', lw=1.0, label='异常分数')
    ax_s.axhline(det_threshold, color='#ef4444', lw=2.0, linestyle='--',
                 label=f'检测阈值 = {det_threshold:.2f}')

    if len(tp_idx):
        ax_s.scatter(test_t[tp_idx], scores[tp_idx], color='#2563eb', s=35, zorder=5,
                     label=f'TP ({len(tp_idx)})')
    if len(fp_idx):
        ax_s.scatter(test_t[fp_idx], scores[fp_idx], color='#ef4444', s=35, zorder=5,
                     label=f'FP ({len(fp_idx)})')
    if len(fn_idx):
        ax_s.scatter(test_t[fn_idx], scores[fn_idx], color='#f59e0b', s=40, zorder=5,
                     marker='v', label=f'FN ({len(fn_idx)})')

    ax_s.set_title('异常分数（归一化）', fontsize=10, fontweight='bold', pad=3)
    ax_s.set_xlabel('时间步（10s/步）', fontsize=9)
    ax_s.set_ylabel('Score', fontsize=9)
    ax_s.legend(loc='upper left', fontsize=7.5, ncol=6, framealpha=0.85)
    ax_s.grid(True, alpha=0.20)

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"图表已保存: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("CPU 指标压测异常检测测试  —  HistoricalThresholdModel")
    print("=" * 70)

    # ── 生成数据 ─────────────────────────────────────────────────────────────
    sim = CPUDataSimulator()
    train, test, labels, stress_ranges = sim.build_dataset(seed=42)

    n_train    = len(train)
    n_test     = len(test)
    n_stress   = int(labels.sum())
    n_normal   = n_test - n_stress
    duration_train = n_train * 10 / 60  # 分钟

    print(f"\n数据集概览")
    print(f"  训练集: {n_train} 条（约 {duration_train:.0f} 分钟，正常负载）")
    print(f"  测试集: {n_test} 条（正常 {n_normal} 条 + 压测 {n_stress} 条）")
    print(f"  压测段: {stress_ranges}  (测试集索引范围，10s/步)")

    print(f"\n训练数据统计（正常负载）")
    for j, feat in enumerate(CPUDataSimulator.FEATURES):
        col = train[:, j]
        print(f"  {feat:12s}  mean={col.mean():.2f}  std={col.std():.2f}"
              f"  min={col.min():.2f}  max={col.max():.2f}")

    print(f"\n压测段数据统计")
    stress_idx = np.where(labels == 1)[0]
    for j, feat in enumerate(CPUDataSimulator.FEATURES):
        col = test[stress_idx, j]
        print(f"  {feat:12s}  mean={col.mean():.2f}  std={col.std():.2f}"
              f"  min={col.min():.2f}  max={col.max():.2f}")

    # ── 训练与检测 ───────────────────────────────────────────────────────────
    print(f"\n正在训练 HistoricalThresholdModel …")
    result = run_threshold_test(train, test, labels)

    print(f"\n学习到的特征阈值")
    for feat, thr in result['thresholds'].items():
        print(f"  {feat:12s}  threshold={thr:.4f}")

    # ── 结果汇报 ─────────────────────────────────────────────────────────────
    p  = result['precision']
    r  = result['recall']
    f1 = result['f1']
    tp = result['tp']
    fp = result['fp']
    fn = result['fn']
    tn = result['tn']

    status = "✅ PASS" if f1 >= 0.80 else ("⚠️  WARN" if f1 >= 0.50 else "❌ FAIL")

    print(f"\n{'─' * 70}")
    print(f"检测结果")
    print(f"{'─' * 70}")
    print(f"  精确率 Precision : {p:.4f}")
    print(f"  召回率 Recall    : {r:.4f}")
    print(f"  F1 Score         : {f1:.4f}  {status}")
    print(f"  混淆矩阵  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"{'─' * 70}")

    # ── 可视化 ───────────────────────────────────────────────────────────────
    save_path = 'test_results/cpu_threshold_test.png'
    plot_results(train, test, labels, result, stress_ranges, save_path=save_path)

    print(f"\n{'=' * 70}")
    print(f"测试完成  F1={f1:.4f}  {status}")
    print(f"{'=' * 70}")
    return result


if __name__ == '__main__':
    main()
