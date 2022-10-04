# Maintainer:     Ryan Young
# Last Modified:  Oct 04, 2022

import pandas as pd, numpy as np

from tsopt.types import *
from tsopt.util import *
from tsopt.values import *


class LayerNodes(LayerList):
    dtype = NodeSR

    @property
    def layer(self):
        sums = [sr.sum() if not sr.empty else np.nan for sr in self]
        return LayerValues(self.mod, sums)

    @property
    def default_template(self):
        return self.mod.dv.template_layers()

    def set_element_format(self, idx, sr:any):
        curr = super().__getitem__(idx)
        sr = sr[sr.columns[0]] if isinstance(sr, pd.DataFrame) else pd.Series(sr)
        sr.index, sr.name = curr.index, None
        return sr.replace(-1, np.nan).astype(float)


    def loc_to_layer_and_node_indexes(self, loc) -> (int, int or None):
        # Layer and node indexes (0, 3) or ('Warehouse', 3) or ('W', 3)
        if isinstance(loc, tuple) or isinstance(loc, list):
            layer = self.mod.dv.layer_index(loc[0])
            return (layer, loc[1])
        else:
            try:
                # Layer via int index, string name, or abbrev
                layer = self.mod.dv.layer_index(loc)
                return (layer, None)
            except Exception:
                # Layer and node via string id - 'B2'
                return self.mod.dv.node_str_to_layer_and_node_indexes(loc)


    def __getitem__(self, loc):
        idx, node = self.loc_to_layer_and_node_indexes(loc)
        sr = super().__getitem__(idx)
        if node:
            return sr[node]
        return sr


    def __setitem__(self, loc, val):
        idx, node = self.loc_to_layer_and_node_indexes(loc)
        if not node:
            super().__setitem__(idx, val)
        else:
            sr = super().__getitem__(idx)
            sr[node] = val


    def load(self, loc, filename, excel_file=None) -> None:
        excel = excel_file if excel_file else self.mod.excel_file
        self[loc] = raw_sr_from_file(filename, excel)



class LayerNodeBounds(LayerList):
    dtype = NodeBoundsDF
    def __init__(self, mod, demand, capacity):
        dfs = [ pd.concat([dem, cap], axis=1).rename(columns={0: 'dem', 1: 'cap'})
            for dem, cap in zip(demand, capacity) ]
        super().__init__(mod, dfs)

    def _repr_html_(self):
        dfs = [df for df in self]
        sr = pd.DataFrame([["",""]], columns=self[0].columns, index=[""])
        new = [item for sublist in [[df, sr, sr] for df in dfs[:-1]] for item in sublist] + [dfs[-1]]
        df = pd.concat(new)
        return df._repr_html_()



class NodeConstraints(LayerNodes):

    @property
    def notnull(self):
        return LayerList(self.mod, [NodeSR(sr[sr.notnull()]) for sr in self])

    @property
    def flow(self):
        idx = self.layer.flow.idx
        return FlowSeries(idx, self[idx])

    @property
    def full(self):
        full = [ sr if sr.isfull() else NodeSR(dtype=float) for sr in self ]
        return LayerNodes(self.mod, full)

    def stage(self, stg):
        return [self[stg], self[stg+1]]


class NodeCapacity(NodeConstraints):

    @property
    def layer(self):
        sums = [sr.sum() if sr.isfull() and not sr.empty else np.nan for sr in self]
        return LayerCapacityValues(self.mod, sums)

    @property
    def true(self):
        fill_val = self.layer.flow.val
        return NodeCapacity(self.mod, [sr.fillna(fill_val) for sr in self])


class NodeDemand(NodeConstraints):

    @property
    def layer(self):
        sums = [sr.sum() if not sr.empty else np.nan for sr in self]
        return LayerDemandValues(self.mod, sums)

    @property
    def true(self):
        fill_val = 0
        return NodeDemand(self.mod, [sr.fillna(fill_val) for sr in self])
