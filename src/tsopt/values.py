# Maintainer:     Ryan Young
# Last Modified:  Dec 06, 2022
import pandas as pd, numpy as np
from tsopt.types import *

@dataclass
class FlowVal:
    idx: int
    val: float

    def __repr__(self):
        return self.val.__repr__()


@dataclass
class FlowSeries:
    idx: int
    sr: NodeSR

    def __repr__(self):
        return self.sr.__repr__()

    def _repr_html_(self):
        return self.sr._repr_html_()


class LayerValues(ListData):

    @property
    def default_template(self):
        return [np.nan for _ in self.mod.layers]

    def __getitem__(self, loc):
        idx = self.mod.layer_index(loc)
        return super().__getitem__(idx)

    def __setitem__(self, loc, val):
        idx = self.mod.layer_index(loc)
        val = float(val)
        super().__setitem__(idx, val)

    def _repr_html_(self):
        sr = pd.Series(self, index=self.mod.abbrevs)
        return sr._repr_html_()



class LayerCapacityValues(LayerValues):

    @property
    def flow(self):
        val = min(self)
        return FlowVal(self.index(val), val)


class LayerDemandValues(LayerValues):

    @property
    def flow(self):
        val = max(self)
        idx = len(self)-1 - list(reversed(self)).index(val)
        return FlowVal(idx, val)

