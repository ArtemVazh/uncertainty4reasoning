from lm_polygraph.estimators.estimator import Estimator

import numpy as np
from typing import Dict

import logging

log = logging.getLogger(__name__)


class PRMEstimator(Estimator):
    def __init__(
            self,
            reduction: str = 'mean',
            scores_key: str = 'prm_scores',
    ):
        super().__init__(
            [scores_key, "claims"],
            "sequence",
        )
        self.scores_key = scores_key
        self.reduction = reduction

    def __str__(self):
        if self.scores_key == "prm_scores":
            return f"PRM"
        return self.scores_key

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
                stats[self.scores_key],
                stats["claims"],
        ):
            claim_ue = [-x for x in sample_prms]
            seq_ue.append(self._reduce(claim_ue))
        return seq_ue
