from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._anndata import PerturbationAnndataPair
from ._de import DEComparison


@dataclass(frozen=True)
class CombinedMetricData:
    """Expression and DE results required by DEG-weighted metrics."""

    anndata_pair: PerturbationAnndataPair
    de_comparison: DEComparison

    def get_perts(self, include_control: bool = False) -> NDArray[np.str_]:
        return self.anndata_pair.get_perts(include_control=include_control)
