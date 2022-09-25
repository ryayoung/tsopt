# Maintainer:     Ryan Young
# Last Modified:  Sep 24, 2022
import pandas as pd, numpy as np
from tsopt.basic_types import *
from tsopt.list_data import *
from tsopt.container import *
from dataclasses import dataclass
from tsopt.layer_based import *

class LayerSR(ModSR):
    ''' Like ModSR but stores ref to model to keep consistent nodes '''
    def __init__(self, mod, sr):
        self._mod = mod
        super().__init__(sr, index=self.correct_index)


    @property
    def mod(self): return self._mod

    @property
    def correct_index(self):
        return self.mod.dv.abbrevs

    @property
    def default_template(self):
        return ModSR(np.nan, index=self.correct_index)


    def __getitem__(self, idx):
        try:
            return super().__getitem__(idx)
        except Exception:
            idx = self.mod.dv.layer_index(idx)
            return super().__getitem__(idx)

    def __setitem__(self, idx, val):
        val = float(val)
        try:
            super().__setitem__(idx, val)
        except Exception:
            idx = self.mod.dv.layer_index(idx)
            super().__setitem__(idx, val)



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


class LayerCapacity(LayerValues):

    @property
    def flow(self):
        lst = list(self)
        val = min(lst)
        return FlowVal(lst.index(val), val)


class LayerDemand(LayerValues):

    @property
    def flow(self):
        lst = list(self)
        val = max(lst)
        idx = len(lst)-1 - list(reversed(lst)).index(val)
        return FlowVal(idx, val)



class LayerValuesContainer(ConstraintsContainer):
    capacity_type = LayerCapacity
    demand_type = LayerDemand

    @property
    def df(self):
        layers = self.mod.dv.layers
        return pd.DataFrame(zip(self.demand, self.capacity), index=layers, columns=['demand','capacity'])

    @property
    def diff(self):
        diffs = [self.capacity[i] - self.demand[i] for i in range(0, len(self))]
        return LayerValues(self.mod, diffs)

    def __len__(self):
        return len(self.capacity)

    def _repr_html_(self):
        return self.df._repr_html_()

