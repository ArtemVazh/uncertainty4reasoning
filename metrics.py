import numpy as np
from lm_polygraph.ue_metrics.ue_metric import UEMetric, skip_target_nans
from lm_polygraph.ue_metrics import ROCAUC, PRAUC
from sklearn.isotonic import IsotonicRegression


def _preprocess_metrics(estimator: list[float], target: list[int], isotonic_regression: bool):
    t, e = skip_target_nans(target, estimator)
    t, e = np.array(t), np.array(e)
    if isotonic_regression:
        # need strict monotonicity, while isotonic_regression is not strict
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(e.reshape(-1, 1), t)
        e = iso_reg.predict(e)
    return t, e


class BrierScore(UEMetric):
    def __init__(self, isotonic_regression: bool = False):
        self.isotonic_regression = isotonic_regression

    def __str__(self):
        if self.isotonic_regression:
            return "bs-isotonic"
        return "bs"

    def __call__(self, estimator: list[float], target: list[int]) -> float:
        if not all(0 <= x <= 1 for x in estimator):
            return np.nan
        t, e = _preprocess_metrics(estimator, target, self.isotonic_regression)
        return np.mean((t - e) ** 2).item()


class ECE(UEMetric):
    def __init__(self, num_bins: int = 10, isotonic_regression: bool = False):
        self.num_bins = num_bins
        self.isotonic_regression = isotonic_regression

    def __str__(self):
        if self.isotonic_regression:
            return "ece-isotonic"
        return "ece"

    def __call__(self, estimator: list[float], target: list[int]) -> float:
        if all(-1 <= x <= 0 for x in estimator):
            estimator = [x + 1 for x in estimator]
        if all(x >= 0 for x in estimator) and not all(x <= 1 for x in estimator):
            estimator = [1 - np.exp(-x) for x in estimator]
        if not all(0 <= x <= 1 for x in estimator):
            return np.nan
        t, e = _preprocess_metrics(estimator, target, self.isotonic_regression)
        bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
        ece = 0.0
        for i in range(self.num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (e >= bin_lower) & (e < bin_upper)
            if np.sum(in_bin) > 0:
                bin_confidence = np.mean(e[in_bin])
                bin_accuracy = np.mean(t[in_bin])
                bin_error = np.abs(bin_confidence - bin_accuracy)
                ece += bin_error * np.sum(in_bin) / len(e)
        return ece
