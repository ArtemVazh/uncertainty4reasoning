from lm_polygraph.estimators.estimator import Estimator

import numpy as np
from typing import Dict
from scipy.special import expit

import logging

log = logging.getLogger(__name__)


class UHeadEstimator(Estimator):
    def __init__(
            self,
            reduction: str = 'mean',
    ):
        super().__init__(
            ["uncertainty_claim_logits", "claims"],
            "sequence",
        )
        self.reduction = reduction

    def __str__(self):
        return f"UHeadClaim {self.reduction}"

    def _reduce(self, x):
        if self.reduction == 'mean':
            return np.mean(x)
        elif self.reduction == 'min':
            return np.min(x)
        elif self.reduction == 'max':
            return np.max(x)
        raise Exception(f"Unknown reduction type: {self.reduction}")

    def __call__(self, stats: Dict[str, np.ndarray]) -> list[float]:
        seq_ue = []
        for sample_ls, sample_claims in zip(
                stats["uncertainty_claim_logits"],
                stats["claims"],
        ):
            # import pdb; pdb.set_trace()
            claim_ue = expit(sample_ls)
            seq_ue.append(self._reduce(claim_ue))
        return seq_ue
