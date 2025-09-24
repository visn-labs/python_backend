import random
from src.main.keyframe_extractor.adaptive_threshold import AdaptiveThresholdManager

def test_adaptive_threshold_activation():
    mgr = AdaptiveThresholdManager(window_size=50, min_history=5, k_low=1.0, k_high=2.0, default_low=100, default_high=200)
    lows = []
    highs = []
    # First few scores use defaults
    for s in [10, 12, 11, 9, 10]:
        l, h = mgr.update(s)
        lows.append(l)
        highs.append(h)
    assert all(l == 100 for l in lows[:-1])  # before reaching min_history
    # After 5th insert adaptation kicks in
    assert lows[-1] != 100 or highs[-1] != 200


def test_adaptive_threshold_trends():
    mgr = AdaptiveThresholdManager(window_size=20, min_history=5, k_low=1.0, k_high=2.0, default_low=50, default_high=100)
    # Low motion phase
    for s in [5,6,5,7,6,5,6,7,5,6]:
        mgr.update(s)
    low1, high1 = mgr.current_thresholds()
    # Higher motion phase
    for s in [30,32,31,29,33,30,31,32,34,30]:
        mgr.update(s)
    low2, high2 = mgr.current_thresholds()
    assert low2 > low1
    assert high2 > high1


def test_adaptive_threshold_ema_smoothing():
    mgr = AdaptiveThresholdManager(window_size=30, min_history=5, k_low=1.0, k_high=2.0, smooth_factor=0.5)
    # Ramp scores upward
    base_low = None
    prev_low = None
    for s in range(5, 25):
        low, high = mgr.update(s)
        if mgr.history_length == 5:
            base_low = low
        if prev_low is not None:
            # Smoothing ensures change is not too abrupt (heuristic: delta < raw score delta * 2)
            assert abs(low - prev_low) < 50
        prev_low = low
    assert base_low is not None
