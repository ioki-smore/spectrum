import numpy as np
from scipy.stats import genpareto
from utils.logger import get_logger

logger = get_logger("utils.thresholding")

def fit_pot(scores: np.ndarray, risk: float = 1e-4, level: float = 0.98) -> float:
    """
    Fits Generalized Pareto Distribution to the tail of scores to determine a robust threshold.
    
    Args:
        scores: array of anomaly scores (reconstruction errors).
        risk: probability of observing an anomaly (q). Default 1e-4.
        level: percentile to choose initial threshold t (0-1). Default 0.98.
    
    Returns:
        threshold: The calculated threshold.
    """
    try:
        # Ensure numpy array
        scores = np.array(scores)
        
        if len(scores) == 0:
            logger.warning("Empty scores provided for POT. Returning infinity.")
            return float('inf')

        # 1. Choose initial threshold t
        t = np.percentile(scores, level * 100)
        
        # 2. Get peaks (excesses over t)
        peaks = scores[scores > t] - t
        num_peaks = len(peaks)
        n = len(scores)
        
        if num_peaks < 10:
            logger.warning(f"Too few peaks ({num_peaks}) for POT fitting. Fallback to max + margin.")
            return np.max(scores) * 1.1
            
        # 3. Fit GPD
        # genpareto.fit returns (shape, loc, scale)
        # We fix loc=0 because we are fitting the excesses
        c, loc, scale = genpareto.fit(peaks, floc=0)
        
        # 4. Calculate final threshold
        # Formula: z_q = t + (scale / c) * (( (risk * n) / num_peaks ) ** (-c) - 1)
        # Handle c close to 0 (Exponential case - use limit formula)
        
        if abs(c) < 1e-8:
            # Exponential limit: threshold = t - scale * log(risk * n / num_peaks)
            threshold = t - scale * np.log(risk * n / num_peaks)
        else:
            term = ((risk * n) / num_peaks) ** (-c) - 1
            threshold = t + (scale / c) * term
        
        logger.info(f"POT Threshold calculated: {threshold:.6f} (Initial t: {t:.6f}, Peaks: {num_peaks}, Shape: {c:.4f}, Scale: {scale:.4f})")
        return threshold
        
    except Exception as e:
        logger.error(f"Error in POT thresholding: {e}. Fallback to 3-sigma.")
        mean = np.mean(scores)
        std = np.std(scores)
        return mean + 3 * std
