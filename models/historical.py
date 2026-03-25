import numpy as np
import polars as pl
from pathlib import Path
from typing import Any, Union, Dict, Tuple
import json
from scipy import signal
from scipy.ndimage import median_filter

from models.base import BaseModel
from utils.errors import Result, Ok, Err, ErrorCode

class HistoricalThresholdModel(BaseModel):
    """
    自适应多策略历史阈值异常检测模型。

    思想：
        训练阶段学习每个特征的「正常历史范围」，
        推断阶段检测超出该范围的值，判定为异常。

    1. 自适应平滑：根据数据自相关动态选择滤波窗口，去除训练噪声
    2. 多策略阈值融合：稳态数据用 percentile+MAD，趋势数据用外推法，按自相关混合
    3. 鲁棒统计：用 MAD 代替 Std、用分位数代替 max，抵抗训练期异常值影响
    4. 分级评分：轻微超出降权，严重超出放大，减少边界误报
    5. 自动检测阈值：对训练数据打分自标定，避免人工猜测检测阈值
    """
    
    def __init__(self, name: str, config: Any):
        super().__init__(name, config)

        # ══════════════════════════════════════════════════════════════
        # 统一流水线，3 个独立开关控制各步骤：
        #   原始数据 ─[平滑]─→ 统计量 ─[阈值策略]─→ 阈值 ─[评分]─→ 分数
        #
        #  旧 simple  = smoothing=False, strategy=percentile, tiered=False
        #  旧 adaptive = smoothing=True,  strategy=fusion,     tiered=True
        #  新增组合例：smoothing=True + percentile（高噪声稳态数据）
        # ══════════════════════════════════════════════════════════════

        # ── 开关 1: 平滑 ─────────────────────────────────────────────
        # True：根据自相关自适应平滑训练数据，去除毛刺后再计算统计量
        # False：直接用原始数据计算统计量（p99.9 等分位数本身已足够鲁棒）
        self.smoothing_enabled = bool(self.get_param('historical_smoothing_enabled', False))

        # ── 开关 2: 阈值策略 ─────────────────────────────────────────
        # 'percentile'：直接取训练数据的 p(threshold_percentile) 作为阈值
        # 'fusion'：    稳态+趋势多策略融合，按自相关系数动态混合
        self.threshold_strategy = self.get_param('historical_threshold_strategy', 'percentile')

        # ── 开关 3: 分级评分 ─────────────────────────────────────────
        # True：轻微超出降权(×0.5)、严重超出放大(×1.5)，减少边界误报
        # False：线性评分 (value-threshold)/threshold，简单直接
        self.tiered_scoring = bool(self.get_param('historical_tiered_scoring', False))

        # ── 阈值参数 ─────────────────────────────────────────────────
        # 百分位阈值（strategy=percentile 时使用）
        self.threshold_percentile = float(self.get_param('historical_threshold_percentile', 99.9))

        # ── 平滑参数（smoothing_enabled=True 时生效）────────────────
        self.smoothing_method = self.get_param('historical_smoothing_method', 'median')
        self.smoothing_window_min = int(self.get_param('historical_smoothing_window_min', 3))
        self.smoothing_window_max = int(self.get_param('historical_smoothing_window_max', 11))

        # ── 融合策略参数（strategy=fusion 时生效）────────────────────
        self.robust_percentile = float(self.get_param('historical_robust_max_percentile', 99.5))
        self.stationary_margin = float(self.get_param('historical_stationary_margin', 1.2))
        self.mad_multiplier = float(self.get_param('historical_mad_multiplier', 5.0))
        self.trending_k_base = float(self.get_param('historical_trending_k_base', 1.5))
        self.trending_k_max = float(self.get_param('historical_trending_k_max', 8.0))
        self.ac1_low = float(self.get_param('historical_ac1_low_threshold', 0.2))
        self.ac1_high = float(self.get_param('historical_ac1_high_threshold', 0.6))

        # ── 分级评分参数（tiered_scoring=True 时生效）────────────────
        self.score_tier1 = float(self.get_param('historical_score_tier1', 0.1))
        self.score_tier2 = float(self.get_param('historical_score_tier2', 0.3))
        self.score_tier1_weight = float(self.get_param('historical_score_tier1_weight', 0.5))
        self.score_tier3_weight = float(self.get_param('historical_score_tier3_weight', 1.5))

        # ── 学习到的状态 ─────────────────────────────────────────────
        self.thresholds: Dict[str, float] = {}
        self.feature_stats: Dict[str, Dict[str, float]] = {}
        self.feature_detection_thresholds: Dict[str, float] = {}
        self.auto_detection_threshold: float = 1.0
        self.is_trained = False
        
    def _compute_autocorrelation(self, series: np.ndarray) -> float:
        """
        计算滞后-1 自相关系数（Lag-1 Autocorrelation），衡量序列的「记忆性」。

        对于序列 x[0], x[1], x[2], ..., x[n-1]，
        滞后-1 自相关衡量的是：「当前值」与「前一个值」有多大的线性相关性。

        不同的自相关程度对应不同的阈值策略：
        - 白噪声（ac1≈0）：历史最大值附近的小裕量就足够了（稳态策略）
        - 趋势数据（ac1≈1）：当前水平可能远超训练最大值，需要外推（趋势策略）
        混合使用两种策略，可以覆盖更广泛的数据模式。

        实现时通过向量点积高效计算（避免显式循环）。
        Args:
            series: 1D numpy 数组

        Returns:
            Lag-1 自相关系数，已截断到 [0, 1]
            （负相关对阈值意义不大，统一按 0 处理）
        """
        if len(series) < 10:
            return 0.0  # 数据太少时无意义，返回 0（按稳态处理）

        # 中心化：减去均值，只分析「波动」部分，消除直流偏置的影响
        series_centered = series - np.mean(series)

        # C(0) = E[(x-μ)²]：样本方差（分母用 N，与标准自相关一致）
        c0 = np.dot(series_centered, series_centered) / len(series)

        # C(1) = E[(x_t-μ)(x_{t-1}-μ)]：滞后-1 协方差
        # series_centered[:-1] = [x_0, x_1, ..., x_{n-2}]（前 n-1 个）
        # series_centered[1:]  = [x_1, x_2, ..., x_{n-1}]（后 n-1 个）
        # 点积 = Σ (x_t - μ)(x_{t-1} - μ)，除以 (n-1) 取均值
        c1 = np.dot(series_centered[:-1], series_centered[1:]) / (len(series) - 1)

        if c0 == 0:
            return 0.0  # 序列为常数（方差为0），按稳态处理

        ac1 = c1 / c0
        # clip 到 [0, 1]：负自相关（值交替震荡）在阈值设计中等同于无相关
        return float(np.clip(ac1, 0.0, 1.0))
    
    def _adaptive_smooth(self, series: np.ndarray, ac1: float) -> np.ndarray:
        """
        根据自相关系数自适应选择平滑窗口，对训练数据进行去噪。

        训练数据中可能存在少量尖峰/异常值（例如训练期偶发的流量毛刺）。
        如果不平滑就直接计算 p95/p99，这些尖峰会使分位数偏高，导致阈值过于宽松。
        平滑后，统计量更能代表真实正常水平。

        【为什么窗口大小要随自相关变化？】
        - ac1 低（白噪声）：数据随机波动大，用大窗口多取样本取中位数 → 过滤随机噪声
          例如：CPU 利用率在 0~100% 随机跳变，需要大窗口才能看清趋势
        - ac1 高（趋势）：数据平滑变化，用小窗口保留局部趋势 → 避免过度平滑
          例如：内存缓慢泄漏，用大窗口会抹掉早期的正常上升趋势

        【两种平滑方法】
        - 'median'（默认）：滚动中位数，对单点尖峰完全免疫
          原理：窗口内的中位数对极值不敏感（改变一个极值不影响中位数）
        - 'savgol'：Savitzky-Golay 滤波器，拟合窗口内的多项式
          原理：保留多项式趋势（如线性、二次），适合平滑趋势数据
          缺点：对孤立尖峰不够鲁棒

        Args:
            series: 1D numpy 数组（训练特征值）
            ac1:    自相关系数（由 _compute_autocorrelation 计算）

        Returns:
            平滑后的 numpy 数组，形状与输入相同
        """
        if len(series) < 3:
            return series  # 数据太少，无法平滑

        # 根据 ac1 线性插值选择窗口大小
        # ac1 < 0.3（噪声）→ window_max（充分平滑）
        # ac1 > 0.6（趋势）→ window_min（轻微平滑）
        # [0.3, 0.6] 之间线性过渡
        if ac1 < 0.3:
            window = self.smoothing_window_max
        elif ac1 > 0.6:
            window = self.smoothing_window_min
        else:
            ratio = (ac1 - 0.3) / 0.3  # 0.0（噪声端）→ 1.0（趋势端）
            window = int(self.smoothing_window_max - ratio * (self.smoothing_window_max - self.smoothing_window_min))

        window = max(3, min(window, len(series)))

        if self.smoothing_method == 'savgol' and window >= 5:
            # Savitzky-Golay 要求窗口为奇数（对称窗口）
            if window % 2 == 0:
                window += 1
            try:
                # polyorder=2：拟合二次多项式，能捕捉非线性趋势
                return signal.savgol_filter(series, window_length=window, polyorder=2)
            except:
                pass  # 如果失败，回退到中位数滤波

        # 滚动中位数：center=True 使窗口以当前点为中心（双边滤波，无延迟偏差）
        # 使用 scipy.ndimage.median_filter 替代 pandas rolling，避免 pandas 依赖
        return median_filter(series, size=window, mode='nearest')
        
    def _parse_input(self, data):
        """统一输入解析，返回 (features, data_matrix)。"""
        if isinstance(data, pl.DataFrame):
            cols = data.columns
            return cols, data.to_numpy()
        elif hasattr(data, 'columns') and hasattr(data, 'values'):
            # pandas DataFrame compatibility
            return list(data.columns), data.values
        else:
            return [f"feature_{i}" for i in range(data.shape[1])], data

    def _vectorized_tiered_scoring(self, base_excess: np.ndarray) -> np.ndarray:
        """
        向量化分级评分（tiered_scoring=True 时使用）。
        用 numpy 条件赋值替代 Python 逐点循环，性能提升 ~50x。
        """
        scores = base_excess.copy()
        mask1 = (scores > 0) & (scores < self.score_tier1)
        mask3 = scores >= self.score_tier2
        scores[mask1] *= self.score_tier1_weight
        scores[mask3] *= self.score_tier3_weight
        return scores

    def _score_features(self, data_matrix: np.ndarray, features: list) -> np.ndarray:
        """
        统一的特征评分逻辑，全向量化。
        tiered_scoring=False：线性评分 (value - threshold) / threshold
        tiered_scoring=True： 分级评分 + 归一化
        """
        n_samples = data_matrix.shape[0]
        scores = np.zeros(n_samples)

        for i, col in enumerate(features):
            if col not in self.thresholds:
                continue
            threshold = self.thresholds[col]
            feat_det_thresh = self.feature_detection_thresholds.get(col, 0.02)

            if threshold > 0:
                base_excess = np.maximum(0.0, data_matrix[:, i] - threshold) / threshold
            else:
                base_excess = np.maximum(0.0, data_matrix[:, i] - threshold)

            if self.tiered_scoring:
                base_excess = self._vectorized_tiered_scoring(base_excess)

            scores = np.maximum(scores, base_excess / feat_det_thresh)
        return scores

    def fit(self, train_data: Union[pl.DataFrame, Any]) -> Result[None]:
        """
        训练阶段：统一流水线，对每个特征独立学习阈值。

        流水线：原始数据 ─[可选平滑]─→ 统计量 ─[阈值策略]─→ 阈值 ─[标定]─→ 检测基线
        """
        try:
            features, data_matrix = self._parse_input(train_data)
            self.thresholds = {}
            self.feature_stats = {}

            for i, col in enumerate(features):
                self._fit_feature(col, data_matrix[:, i])

            # 逐特征独立标定检测阈值（向量化）
            # Use trimmed data (p0.5 to p98) to compute calibration scores,
            # so that training-period contamination doesn't inflate the threshold.
            # This raises contamination tolerance from ~1% to ~2-3%.
            self.feature_detection_thresholds = {}
            for i, col in enumerate(features):
                col_thresh = self.thresholds.get(col)
                if col_thresh is None:
                    continue
                valid_col = data_matrix[:, i]
                valid_col = valid_col[~np.isnan(valid_col)]

                # Trim extremes to resist contamination
                lo, hi = np.percentile(valid_col, [0.5, 98.0])
                trimmed_col = valid_col[(valid_col >= lo) & (valid_col <= hi)]
                if len(trimmed_col) < 10:
                    trimmed_col = valid_col  # fallback if too few points

                if col_thresh > 0:
                    base_excess = np.maximum(0.0, trimmed_col - col_thresh) / col_thresh
                else:
                    base_excess = np.maximum(0.0, trimmed_col - col_thresh)

                if self.tiered_scoring:
                    base_excess = self._vectorized_tiered_scoring(base_excess)

                tuned = float(np.percentile(base_excess, 99.0) * 1.2 + 1e-6)
                self.feature_detection_thresholds[col] = max(tuned, 0.02)

            self.auto_detection_threshold = 1.0
            self.is_trained = True
            return Ok(None)

        except Exception as e:
            print(f"Error in HistoricalThresholdModel.fit: {e}")
            import traceback
            traceback.print_exc()
            return Err(ErrorCode.TRAINING_FAILED)

    def _fit_feature(self, col: str, series: np.ndarray) -> None:
        """
        单特征训练流水线（统一路径，无 mode 分支）。

        Step 1 [可选]: 自相关 + 自适应平滑  ← smoothing_enabled
        Step 2: 计算统计量
        Step 3: 阈值计算                     ← threshold_strategy
        """
        valid_data = series[~np.isnan(series)]
        if len(valid_data) < 10:
            return

        # ── Step 1: 可选平滑 ──────────────────────────────────────
        ac1 = self._compute_autocorrelation(valid_data)
        if self.smoothing_enabled:
            working_data = self._adaptive_smooth(valid_data, ac1)
        else:
            working_data = valid_data

        # ── Step 2: 统计量 ────────────────────────────────────────
        median = float(np.median(working_data))
        stats = {'median': median, 'ac1': ac1}

        # ── Step 3: 阈值计算 ──────────────────────────────────────
        if self.threshold_strategy == 'fusion':
            # 多策略融合：稳态 + 趋势，按自相关混合
            p95 = float(np.percentile(working_data, 95))
            p99 = float(np.percentile(working_data, 99))
            p99_5 = float(np.percentile(working_data, self.robust_percentile))
            max_val = float(np.percentile(working_data, self.robust_percentile))
            q25 = float(np.percentile(working_data, 25))
            q75 = float(np.percentile(working_data, 75))
            iqr = q75 - q25
            mad = float(np.median(np.abs(working_data - median)))
            volatility = iqr / (median + 1e-8) if median > 0 else 0.0

            threshold_stat = min(
                p99 * self.stationary_margin,
                median + self.mad_multiplier * mad
            )
            spread = max(max_val - median, iqr, 1e-8)
            k = min(self.trending_k_base + 6.0 * ac1, self.trending_k_max)
            threshold_trend = max_val + k * spread

            alpha_ac1 = float(np.clip(
                (ac1 - self.ac1_low) / (self.ac1_high - self.ac1_low), 0.0, 1.0
            ))
            n_smooth = len(working_data)
            slope = float(np.polyfit(range(n_smooth), working_data, 1)[0])
            slope_normalized = abs(slope) * n_smooth / (spread + 1e-8)
            alpha_trend = float(np.clip(slope_normalized, 0.0, 1.0))
            alpha = min(alpha_ac1, alpha_trend)
            threshold_base = (1.0 - alpha) * threshold_stat + alpha * threshold_trend

            volatility_factor = 1.0 + 0.3 * min(volatility, 0.5)
            threshold_final = threshold_base * volatility_factor

            stats.update({
                'mad': mad, 'iqr': iqr, 'p95': p95, 'p99': p99,
                'p99_5': p99_5, 'max': max_val, 'volatility': volatility,
                'threshold_stat': threshold_stat, 'threshold_trend': threshold_trend,
                'alpha': alpha, 'alpha_ac1': alpha_ac1, 'alpha_trend': alpha_trend,
                'slope': slope, 'slope_normalized': slope_normalized,
                'volatility_factor': volatility_factor,
            })
        else:
            # 百分位阈值
            threshold_final = float(np.percentile(working_data, self.threshold_percentile))
            stats['threshold_percentile'] = self.threshold_percentile

        self.thresholds[col] = float(threshold_final)
        self.feature_stats[col] = stats
            
    def predict(self, data: Union[pl.DataFrame, Any]) -> Result[np.ndarray]:
        """
        推断阶段：为每个时间步打分，返回 [0, +∞) 的异常分数数组。
        全向量化，无 Python 逐点循环。
        最终判断：score > 1.0 → 异常
        """
        if not self.is_trained:
            return Err(ErrorCode.MODEL_NOT_TRAINED)
        try:
            features, data_matrix = self._parse_input(data)
            scores = self._score_features(data_matrix, features)
            return Ok(scores)
        except Exception as e:
            print(f"Error in HistoricalThresholdModel.predict: {e}")
            import traceback
            traceback.print_exc()
            return Err(ErrorCode.INFERENCE_FAILED)

    def get_contribution(self, data: Union[pl.DataFrame, Any]) -> Result[np.ndarray]:
        """
        计算每个特征对异常分数的贡献（向量化）。
        返回 (n_samples, n_features) 的矩阵。
        """
        if not self.is_trained:
            return Err(ErrorCode.MODEL_NOT_TRAINED)
        try:
            features, data_matrix = self._parse_input(data)
            n_samples, n_features = data_matrix.shape
            contributions = np.zeros((n_samples, n_features))

            for i, col in enumerate(features):
                if col not in self.thresholds:
                    continue
                threshold = self.thresholds[col]
                if threshold > 0:
                    base_excess = np.maximum(0.0, data_matrix[:, i] - threshold) / threshold
                else:
                    base_excess = np.maximum(0.0, data_matrix[:, i] - threshold)

                if self.tiered_scoring:
                    base_excess = self._vectorized_tiered_scoring(base_excess)

                contributions[:, i] = base_excess

            return Ok(contributions)
        except Exception as e:
            print(f"Error in HistoricalThresholdModel.get_contribution: {e}")
            import traceback
            traceback.print_exc()
            return Err(ErrorCode.INFERENCE_FAILED)
            
    def save(self, path: Union[str, Path]) -> Result[None]:
        """Save model state to JSON."""
        if not self.is_trained:
            return Err(ErrorCode.MODEL_NOT_TRAINED)
        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            if save_path.suffix in ('.pt', '.pth'):
                save_path = save_path.with_suffix('.json')

            state = {
                'thresholds': self.thresholds,
                'feature_stats': self.feature_stats,
                'feature_detection_thresholds': self.feature_detection_thresholds,
                'auto_detection_threshold': self.auto_detection_threshold,
                'hyperparameters': {
                    'smoothing_enabled': self.smoothing_enabled,
                    'threshold_strategy': self.threshold_strategy,
                    'tiered_scoring': self.tiered_scoring,
                    'threshold_percentile': self.threshold_percentile,
                    'smoothing_method': self.smoothing_method,
                    'smoothing_window_min': self.smoothing_window_min,
                    'smoothing_window_max': self.smoothing_window_max,
                    'robust_percentile': self.robust_percentile,
                    'stationary_margin': self.stationary_margin,
                    'mad_multiplier': self.mad_multiplier,
                    'trending_k_base': self.trending_k_base,
                    'trending_k_max': self.trending_k_max,
                    'ac1_low': self.ac1_low,
                    'ac1_high': self.ac1_high,
                    'score_tier1': self.score_tier1,
                    'score_tier2': self.score_tier2,
                    'score_tier1_weight': self.score_tier1_weight,
                    'score_tier3_weight': self.score_tier3_weight,
                },
            }
            with open(save_path, 'w') as f:
                json.dump(state, f, indent=2)
            return Ok(None)
        except Exception as e:
            print(f"Error saving HistoricalThresholdModel: {e}")
            import traceback
            traceback.print_exc()
            return Err(ErrorCode.IO_ERROR)

    def load(self, path: Union[str, Path]) -> Result[None]:
        """Load model state from JSON."""
        try:
            load_path = Path(path)
            if load_path.suffix in ('.pt', '.pth'):
                load_path = load_path.with_suffix('.json')
            if not load_path.exists():
                return Err(ErrorCode.MODEL_NOT_FOUND)

            with open(load_path, 'r') as f:
                state = json.load(f)

            self.thresholds = state.get('thresholds', {})
            self.feature_stats = state.get('feature_stats', {})
            self.feature_detection_thresholds = state.get('feature_detection_thresholds', {})
            self.auto_detection_threshold = state.get('auto_detection_threshold', 1.0)

            if 'hyperparameters' in state:
                hp = state['hyperparameters']
                self.smoothing_enabled = hp.get('smoothing_enabled', self.smoothing_enabled)
                self.threshold_strategy = hp.get('threshold_strategy', self.threshold_strategy)
                self.tiered_scoring = hp.get('tiered_scoring', self.tiered_scoring)
                self.threshold_percentile = hp.get('threshold_percentile', self.threshold_percentile)
                self.smoothing_method = hp.get('smoothing_method', self.smoothing_method)
                self.smoothing_window_min = hp.get('smoothing_window_min', self.smoothing_window_min)
                self.smoothing_window_max = hp.get('smoothing_window_max', self.smoothing_window_max)
                self.robust_percentile = hp.get('robust_percentile', self.robust_percentile)
                self.stationary_margin = hp.get('stationary_margin', self.stationary_margin)
                self.mad_multiplier = hp.get('mad_multiplier', self.mad_multiplier)
                self.trending_k_base = hp.get('trending_k_base', self.trending_k_base)
                self.trending_k_max = hp.get('trending_k_max', self.trending_k_max)
                self.ac1_low = hp.get('ac1_low', self.ac1_low)
                self.ac1_high = hp.get('ac1_high', self.ac1_high)
                self.score_tier1 = hp.get('score_tier1', self.score_tier1)
                self.score_tier2 = hp.get('score_tier2', self.score_tier2)
                self.score_tier1_weight = hp.get('score_tier1_weight', self.score_tier1_weight)
                self.score_tier3_weight = hp.get('score_tier3_weight', self.score_tier3_weight)

            self.is_trained = True
            return Ok(None)
        except Exception as e:
            print(f"Error loading HistoricalThresholdModel: {e}")
            import traceback
            traceback.print_exc()
            return Err(ErrorCode.IO_ERROR)
