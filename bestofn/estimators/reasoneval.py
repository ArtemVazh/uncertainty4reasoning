from lm_polygraph.estimators.estimator import Estimator

import numpy as np
from typing import Dict

import logging

log = logging.getLogger(__name__)


class ReasonEvalEstimator(Estimator):
    def __init__(
            self,
            reduction: str = 'mean',
            agg: str = '',
    ):
        super().__init__(
            ["reasoneval_scores", "claims"],
            "sequence",
        )
        self.reduction = reduction
        self.agg = agg

    def __str__(self):
        return f"ReasonEval" + (f" {self.agg}" if self.agg else "")

    def _reduce(self, x):
        if self.reduction == 'mean':
            return np.mean(x)
        elif self.reduction == 'min':
            return np.min(x)
        elif self.reduction == 'max':
            return np.max(x)
        raise Exception(f"Unknown reduction type: {self.reduction}")

    def _aggregate(self, x: dict[str, float]) -> float:
        if self.agg == 'redundancy':
            return x['redundancy']
        elif self.agg == 'validity':
            return -x['validity']
        elif self.agg == '':
            return x['redundancy'] - x['validity']
        raise Exception(f"Unknown reduction type: {self.reduction}")

    def __call__(self, stats: Dict[str, np.ndarray]) -> list[float]:
        seq_ue = []
        for sample_prms, sample_claims in zip(
                stats["reasoneval_scores"],
                stats["claims"],
        ):
            claim_ue = [self._aggregate(x) for x in sample_prms]
            seq_ue.append(self._reduce(claim_ue))
        return seq_ue
