from lm_polygraph.estimators.estimator import Estimator

import numpy as np
from typing import Dict

import logging

log = logging.getLogger(__name__)


class PRMEstimator(Estimator):
    def __init__(
            self,
            reduction: str = 'mean',
    ):
        super().__init__(
            ["prm_scores", "claims"],
            "sequence",
        )
        self.reduction = reduction

    def __str__(self):
        return f"PRM"

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
        for sample_prms, sample_claims in zip(
                stats["prm_scores"],
                stats["claims"],
        ):
            claim_ue = [-x for x in sample_prms]
            seq_ue.append(self._reduce(claim_ue))
        return seq_ue
