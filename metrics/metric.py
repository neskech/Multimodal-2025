"""
This is a simple generic metric class that all metrics should inherit from.
Each metric should implement the `compute` method.
Args:
    pred: Predictions from the model
    target: Ground truth labels
Returns:
    Computed metric value
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple

from torch import Tensor
from traitlets import List


class Metric(ABC):
    @abstractmethod
    def compute(
        self, embeddings: List[Tuple[List[Tensor], Tensor]]
    ) -> Tuple[float, Any] | float:
        """Compute the metric, returning a float score and optional additional info."""
        pass
