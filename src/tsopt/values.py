# Maintainer:     Ryan Young
# Last Modified:  Sep 26, 2022
import pandas as pd, numpy as np
from tsopt.types import *


class LayerValues(LayerList):

    @property
    def default_template(self):
        return [np.nan for _ in self.mod.dv.layers]

    def __getitem__(self, loc):
        idx = self.mod.dv.layer_index(loc)
        return super().__getitem__(idx)

    def __setitem__(self, loc, val):
        idx = self.mod.dv.layer_index(loc)
        val = float(val)
        super().__setitem__(idx, val)

    def _repr_html_(self):
        sr = NodeSR(self, index=self.mod.dv.abbrevs)
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

