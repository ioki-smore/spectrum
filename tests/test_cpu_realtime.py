"""
CPU 指标真实采集 + 阈值算法测试
================================
1. 采集阶段一（训练集）：正常负载，持续 TRAIN_SECS 秒
2. 采集阶段二（测试集）：交替正常 / CPU 压测段，压测由脚本自动注入
3. 用 HistoricalThresholdModel 训练并检测，输出 P/R/F1 及可视化图

压测方式：multiprocessing 启动 N 个忙循环进程（不依赖外部工具）
采集频率：INTERVAL_SECS 秒/次
总耗时约：TRAIN_SECS + TEST_SECS 秒
"""

import os
import sys
import time
import threading
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import matplotlib
import matplotlib.pyplot as plt
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


# ─────────────────────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────────────────────

INTERVAL_SECS  = 1.0     # 采样间隔（秒）
TRAIN_SECS     = 60      # 训练集采集时长（秒）—— 正常负载基线
# 测试集段配置：(时长秒, 是否压测)
TEST_SEGMENTS  = [
    (20, False),   # 正常
    (30, True),    # 压测
    (20, False),   # 正常
    (30, True),    # 压测
    (20, False),   # 正常
    (30, True),    # 压测
    (20, False),   # 正常
]
# 压测时启动的忙循环进程数（None = CPU 核心数）
STRESS_WORKERS = None

FEATURES = ['user_pct', 'system_pct', 'load1']
SAVE_PATH = 'test_results/cpu_realtime_test.png'


# ─────────────────────────────────────────────────────────────────────────────
# 压测工具
# ─────────────────────────────────────────────────────────────────────────────

def _busy_loop(_):
    """子进程：持续死循环消耗 CPU。"""
    while True:
        _ = sum(i * i for i in range(10000))


class StressController:
    """通过 multiprocessing 动态开启/关闭 CPU 压测进程。"""

    def __init__(self, n_workers: int = None):
        cpu_count = multiprocessing.cpu_count()
        self.n_workers = n_workers or cpu_count
        self._pool: list[multiprocessing.Process] = []

    def start(self):
        if self._pool:
            return
        for _ in range(self.n_workers):
            p = multiprocessing.Process(target=_busy_loop, args=(None,), daemon=True)
            p.start()
            self._pool.append(p)

    def stop(self):
        for p in self._pool:
            p.terminate()
            p.join(timeout=2)
        self._pool.clear()

    def __del__(self):
        self.stop()


# ─────────────────────────────────────────────────────────────────────────────
# CPU 指标采集
# ─────────────────────────────────────────────────────────────────────────────

def sample_cpu() -> tuple[float, float, float]:
    """采集一次 CPU 指标。返回 (user_pct, system_pct, load1)。"""
    ct = psutil.cpu_times_percent(interval=None)
    user   = ct.user
    system = ct.system
    load1  = os.getloadavg()[0]
    return user, system, load1


def collect_phase(duration_secs: float, interval: float,
                  label: int, stress_ctrl: StressController | None = None,
                  is_stress: bool = False,
                  phase_name: str = '') -> tuple[np.ndarray, np.ndarray]:
    """
    采集一段数据。

    Returns:
        data   : shape (N, 3)
        labels : shape (N,)  全为 label
    """
    if is_stress and stress_ctrl:
        stress_ctrl.start()
        phase_tag = f'[压测中]'
    else:
        if stress_ctrl:
            stress_ctrl.stop()
        phase_tag = '[正常]  '

    n_samples = max(1, int(duration_secs / interval))
    rows   = []
    # 首次调用 cpu_times_percent 需要一个参考点，先预热一次（interval=0 不阻塞）
    psutil.cpu_times_percent(interval=0)

    print(f"  {phase_tag} {phase_name}  采集 {n_samples} 条 (~{duration_secs:.0f}s)  ", end='', flush=True)
    start = time.time()
    for i in range(n_samples):
        next_tick = start + (i + 1) * interval
        row = sample_cpu()
        rows.append(row)
        sleep_for = next_tick - time.time()
        if sleep_for > 0:
            time.sleep(sleep_for)
        if (i + 1) % 10 == 0:
            print('.', end='', flush=True)

    print(f'  完成')
    data   = np.array(rows, dtype=float)
    labels = np.full(len(data), label, dtype=int)
    return data, labels


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def run_collection() -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    执行完整采集流程，返回 (train, test, labels, stress_ranges)。
    stress_ranges: 压测段在测试集中的 [start, end) 索引列表
    """
    stress_ctrl = StressController(n_workers=STRESS_WORKERS)
    total_test_secs = sum(d for d, _ in TEST_SEGMENTS)
    total_secs = TRAIN_SECS + total_test_secs
    n_cpu = multiprocessing.cpu_count()

    print(f"\n{'─' * 70}")
    print(f"CPU 真实采集计划")
    print(f"{'─' * 70}")
    print(f"  训练集: {TRAIN_SECS}s 正常负载")
    print(f"  测试集: {total_test_secs}s（段配置: {TEST_SEGMENTS}）")
    print(f"  总耗时: ~{total_secs}s (~{total_secs/60:.1f} 分钟)")
    print(f"  压测进程数: {stress_ctrl.n_workers} / {n_cpu} 核")
    print(f"  采样间隔: {INTERVAL_SECS}s")
    print(f"{'─' * 70}")
    print()

    # ── 阶段一：训练集（正常基线） ─────────────────────────────────────────
    print(f"=== 阶段一：训练集采集（正常负载，{TRAIN_SECS}s）===")
    train, _ = collect_phase(TRAIN_SECS, INTERVAL_SECS, label=0,
                             stress_ctrl=stress_ctrl, is_stress=False,
                             phase_name='训练基线')
    print()

    # ── 阶段二：测试集（交替正常/压测） ────────────────────────────────────
    print(f"=== 阶段二：测试集采集（正常+压测交替，{total_test_secs}s）===")
    test_chunks  = []
    label_chunks = []
    stress_ranges: list[tuple[int, int]] = []
    cursor = 0

    for seg_idx, (dur, is_stress) in enumerate(TEST_SEGMENTS):
        label = 1 if is_stress else 0
        phase_name = f'段[{seg_idx+1}/{len(TEST_SEGMENTS)}]'
        data, lbls = collect_phase(dur, INTERVAL_SECS, label=label,
                                   stress_ctrl=stress_ctrl, is_stress=is_stress,
                                   phase_name=phase_name)
        if is_stress:
            stress_ranges.append((cursor, cursor + len(data)))
        test_chunks.append(data)
        label_chunks.append(lbls)
        cursor += len(data)

    stress_ctrl.stop()

    test   = np.vstack(test_chunks)
    labels = np.concatenate(label_chunks)
    return train, test, labels, stress_ranges


# ─────────────────────────────────────────────────────────────────────────────
# 评估
# ─────────────────────────────────────────────────────────────────────────────

def to_df(arr: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(arr, columns=FEATURES)


def run_threshold_test(train: np.ndarray, test: np.ndarray,
                       labels: np.ndarray) -> dict:
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
    model = HistoricalThresholdModel(name='CPU_Realtime', config=config)

    fit_res = model.fit(to_df(train))
    if fit_res.is_err():
        raise RuntimeError(f"fit() 失败: {fit_res.err_value}")

    pred_res = model.predict(to_df(test))
    if pred_res.is_err():
        raise RuntimeError(f"predict() 失败: {pred_res.err_value}")

    scores      = pred_res.unwrap()
    threshold   = model.auto_detection_threshold
    predictions = (scores > threshold).astype(int)

    precision = precision_score(labels, predictions, zero_division=0)
    recall    = recall_score(labels, predictions, zero_division=0)
    f1        = f1_score(labels, predictions, zero_division=0)
    cm        = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        'model': model, 'scores': scores, 'predictions': predictions,
        'threshold': threshold,
        'precision': precision, 'recall': recall, 'f1': f1,
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'thresholds': model.thresholds,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(train: np.ndarray, test: np.ndarray, labels: np.ndarray,
                 result: dict, stress_ranges: list) -> None:
    Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)

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

    n_rows = len(FEATURES) + 1
    height_ratios = [3] * len(FEATURES) + [2]
    fig, axes = plt.subplots(n_rows, 1, figsize=(18, 3.5 * len(FEATURES) + 3),
                             sharex=True,
                             gridspec_kw={'height_ratios': height_ratios, 'hspace': 0.10})

    p = result['precision'];  r = result['recall'];  f = result['f1']
    fig.suptitle(
        f'CPU 真实采集  —  阈值算法检测压测异常\n'
        f'精确率 P={p:.3f}  召回率 R={r:.3f}  F1={f:.3f}  '
        f'TP={result["tp"]}  FP={result["fp"]}  FN={result["fn"]}',
        fontsize=13, fontweight='bold'
    )

    feature_units = {'user_pct': '%', 'system_pct': '%', 'load1': ''}

    for i, feat in enumerate(FEATURES):
        ax = axes[i]
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.axvspan(train_t[0], n_train, color='#d1fae5', alpha=0.28, label='训练区（正常基线）')
        ax.axvline(n_train, color='#6b7280', lw=1.2, linestyle='--', alpha=0.7)

        ax.plot(train_t, train[:, i], color='#22c55e', lw=0.8, alpha=0.8, label='训练（正常）')
        ax.plot(test_t,  test[:, i],  color='#2563eb', lw=0.8, alpha=0.85, label='测试')

        if feat in model.thresholds:
            thr = model.thresholds[feat]
            ax.axhline(thr, color='#ef4444', lw=1.8, linestyle='--',
                       label=f'阈值 {thr:.2f}{feature_units[feat]}')
        if feat in model.feature_stats:
            med = model.feature_stats[feat]['median']
            ax.axhline(med, color='#f59e0b', lw=1.2, linestyle=':', alpha=0.8,
                       label=f'中位数 {med:.2f}')

        for s, e in stress_ranges:
            ax.axvspan(test_t[s], test_t[min(e, n_test) - 1], color='#fed7aa', alpha=0.35)

        if len(true_idx):
            ax.scatter(test_t[true_idx], test[true_idx, i],
                       color='#ef4444', s=10, alpha=0.45, marker='x',
                       label='真实压测点', zorder=4)
        if len(pred_idx):
            ax.scatter(test_t[pred_idx], test[pred_idx, i],
                       color='#8b5cf6', s=35, alpha=0.55, marker='o',
                       facecolors='none', linewidths=1.5, label='预测异常', zorder=5)

        unit = f' ({feature_units[feat]})' if feature_units[feat] else ''
        ax.set_title(f'{feat}{unit}', fontsize=10, fontweight='bold', pad=3)
        ax.set_ylabel(feat, fontsize=9)
        ax.legend(loc='upper left', fontsize=7.5, ncol=5, framealpha=0.85)
        ax.grid(True, alpha=0.20)

    # ── 异常分数 ──────────────────────────────────────────────────────────────
    ax_s = axes[-1]
    ax_s.set_facecolor('white')
    ax_s.spines['top'].set_visible(False)
    ax_s.spines['right'].set_visible(False)

    all_t  = np.concatenate([train_t, test_t])
    padded = np.full(len(all_t), np.nan)
    padded[n_train:] = scores

    ax_s.axvspan(train_t[0], n_train, color='#d1fae5', alpha=0.25)
    ax_s.axvline(n_train, color='#6b7280', lw=1.2, linestyle='--', alpha=0.7)

    for s, e in stress_ranges:
        ax_s.axvspan(test_t[s], test_t[min(e, n_test) - 1], color='#fed7aa', alpha=0.35)

    ax_s.fill_between(all_t, padded, alpha=0.20, color='#8b5cf6')
    ax_s.plot(all_t, padded, color='#8b5cf6', lw=1.0, label='异常分数')
    ax_s.axhline(det_threshold, color='#ef4444', lw=2.0, linestyle='--',
                 label=f'检测阈值 = {det_threshold:.2f}')

    if len(tp_idx): ax_s.scatter(test_t[tp_idx], scores[tp_idx],
                                  color='#2563eb', s=35, zorder=5, label=f'TP ({len(tp_idx)})')
    if len(fp_idx): ax_s.scatter(test_t[fp_idx], scores[fp_idx],
                                  color='#ef4444', s=35, zorder=5, label=f'FP ({len(fp_idx)})')
    if len(fn_idx): ax_s.scatter(test_t[fn_idx], scores[fn_idx],
                                  color='#f59e0b', s=40, zorder=5, marker='v',
                                  label=f'FN ({len(fn_idx)})')

    ax_s.set_title('异常分数（归一化）', fontsize=10, fontweight='bold', pad=3)
    ax_s.set_xlabel(f'时间步（{INTERVAL_SECS:.0f}s/步）', fontsize=9)
    ax_s.set_ylabel('Score', fontsize=9)
    ax_s.legend(loc='upper left', fontsize=7.5, ncol=6, framealpha=0.85)
    ax_s.grid(True, alpha=0.20)

    plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n图表已保存: {SAVE_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("CPU 真实采集 + 阈值算法压测异常检测")
    print("=" * 70)

    # ── 采集 ─────────────────────────────────────────────────────────────────
    train, test, labels, stress_ranges = run_collection()

    # ── 数据统计 ─────────────────────────────────────────────────────────────
    stress_idx  = np.where(labels == 1)[0]
    normal_idx  = np.where(labels == 0)[0]
    print(f"\n采集完成  训练={len(train)}条  测试={len(test)}条  "
          f"（正常={len(normal_idx)}  压测={len(stress_idx)}）")

    print(f"\n训练集（正常基线）统计")
    for j, feat in enumerate(FEATURES):
        col = train[:, j]
        print(f"  {feat:12s}  mean={col.mean():.2f}  std={col.std():.2f}"
              f"  min={col.min():.2f}  max={col.max():.2f}")

    if len(stress_idx):
        print(f"\n测试集——压测段统计")
        for j, feat in enumerate(FEATURES):
            col = test[stress_idx, j]
            print(f"  {feat:12s}  mean={col.mean():.2f}  std={col.std():.2f}"
                  f"  min={col.min():.2f}  max={col.max():.2f}")

    # ── 训练与检测 ───────────────────────────────────────────────────────────
    print(f"\n训练 HistoricalThresholdModel …")
    result = run_threshold_test(train, test, labels)

    print(f"\n学习到的特征阈值")
    for feat, thr in result['thresholds'].items():
        print(f"  {feat:12s}  threshold={thr:.4f}")

    # ── 结果汇报 ─────────────────────────────────────────────────────────────
    p  = result['precision'];  r  = result['recall'];  f1 = result['f1']
    tp = result['tp'];         fp = result['fp']
    fn = result['fn'];         tn = result['tn']
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
    plot_results(train, test, labels, result, stress_ranges)

    print(f"\n{'=' * 70}")
    print(f"完成  F1={f1:.4f}  {status}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
