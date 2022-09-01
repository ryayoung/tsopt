# Maintainer:     Ryan Young
# Last Modified:  Aug 31, 2022

from copy import copy, deepcopy
import pandas as pd, numpy as np

from tsopt.vector_util import *
from tsopt.layer_based import *


class StageDFs(list):
    def __init__(self, *args):
        self.__nodes = nodes_from_stage_dfs(args[0])
        list.__init__(self, *args)

    @property
    def nodes(self):
        return self.__nodes


    def _get_df_and_slice(self, loc) -> (pd.DataFrame, any):
        '''
        Used by getitem and setitem if they are passed a multi-value
        index.
        - 1st val will select a dataframe, as usual.
        - 2nd and/or 3rd val selects a series from EITHER axis of the df
        '''

        def node_label_to_idx(df, layer_idx, node) -> int:
            if isinstance(node, str):
                if layer_idx == 1:
                    return list(df.columns).index(node)
                return list(df.index).index(node)
            return node

        def layer_from_node_label(df, node) -> int:
            if isinstance(node, str):
                if node.upper() in df.columns: return 1
                if node.upper() in df.index: return 0
            raise ValueError(f"Invalid node, {node}")

        def series_iloc_slice(layer_idx, node_idx):
            if layer_idx == 0:
                return node_idx
            return (slice(None,None,None), node_idx)

        '''
        To select a node series from df, we need two things:
            1. int (0 or 1) representing the layer (input or output layer)
            2. str OR int representing a node in that layer
        If node name is given, instead of index, then we can find the
        layer index automatically.
        ---
        Say we have df for Distributor -> Retailer:
            [0, 1] returns D1's edges with Retailers
            [1, 1] returns R1's edges with Distributors
            [1, 'R1'] (same as above)
            ['R1'] (same as above)
        '''

        key, loc = loc[0], loc[1:]
        df = list.__getitem__(self, key)

        if len(loc) == 1:
            loc = layer_from_node_label(df, loc[0]), loc[0]

        node_idx = node_label_to_idx(df, loc[0], loc[1])

        return df, series_iloc_slice(loc[0], node_idx)


    def __getitem__(self, loc):
        if isinstance(loc, int):
            return list.__getitem__(self, loc)

        df, df_slice = self._get_df_and_slice(loc)

        return df.iloc[df_slice]


    def __setitem__(self, loc, val):

        def validate_dataframe(layer, new):

            if not isinstance(new, pd.DataFrame):
                new = pd.DataFrame(new)

            old = self[layer]

            if not new.shape == old.shape:
                raise ValueError(f"New df must have same shape as original")
            if not tuple(new.columns) == tuple(old.columns):
                new.columns = old.columns
            if not tuple(new.index) == tuple(old.index):
                new.index = old.index

            return new.astype(float)


        def validate_partial_value(df, df_slice, val):

            if isinstance(val, int):
                return float(val)

            if is_list_tuple_or_series(val):
                curr = df.iloc[df_slice]
                val = pd.Series(val).astype(float)
                val.index, val.name = curr.index, curr.name
                return val

            raise ValueError(f"Invalid value type")


        if isinstance(loc, int):
            if not 0 - len(self) <= loc <= len(self):
                raise ValueError(f"Invalid stage, {loc}")
            new_df = validate_dataframe(loc, val)
            list.__setitem__(self, loc, new_df)
        else:
            df, df_slice = self._get_df_and_slice(loc)
            val = validate_partial_value(df, df_slice, val)
            df.iloc[df_slice] = val


class Costs(StageDFs):
    pass



class EdgeConstraints(StageDFs):
    def __init__(self, items, model=None):
        self.__mod = model
        StageDFs.__init__(self, items)

    @property
    def mod(self): return self.__mod

    def true(self, fill_val):
        return EdgeConstraints([df.fillna(fill_val) for df in copy(self)], self.mod)

    @staticmethod
    def combine_if(sr1, sr2, func) -> pd.Series:
        def pick(a,b):
            if not np.isnan(a) and not np.isnan(b):
                return func(a,b)
            if not np.isnan(a):
                return a
            return b
        return sr1.combine(sr2, pick)


    def nodes_by_layer(self, stages, func) -> LayerDict:
        inputs = stages[0][0]
        outputs = stages[-1][1]
        flows = [ self.combine_if(s1[1], s2[0], func) for s1,s2 in staged(stages) ]

        all_layers = [inputs] + flows + [outputs]
        ldict = {i: sr for i,sr in enumerate(all_layers)}

        return LayerDict(self.mod, ldict)




class EdgeCapacity(EdgeConstraints):

    @property
    def true(self):
        fill_val = self.mod.capacity.flow.val
        return EdgeConstraints.true(self, fill_val)


    def nodes_by_stage(self, new=True) -> list:
        ''' Show nodes whose edges are fully constrained by edge upper bounds
        to a value lower than their original capacity '''

        def lowered(stage):
            sums1, sums2 = self[stage].sums(full=True, concat=False)
            cap1, cap2 = self.mod.capacity.stage(stage)
            if new:
                sums1[sums1 >= cap1] = np.nan
                sums2[sums2 >= cap2] = np.nan
            else:
                sums1 = self.combine_if(sums1, cap1, min)
                sums2 = self.combine_if(sums2, cap2, min)
            return [sums1, sums2]

        return [lowered(i) for i in self.mod.dv.range_stage()]


    def nodes_by_layer(self, new=True) -> LayerDict:
        stages = self.nodes_by_stage(new)
        return EdgeConstraints.nodes_by_layer(self, stages, min)



class EdgeDemand(EdgeConstraints):

    @property
    def true(self):
        fill_val = 0
        return EdgeConstraints.true(self, fill_val)


    def nodes_by_stage(self, new=True) -> list:
        ''' Show nodes which have edges constrained to a lower bound totaling
        to greater than the node's demand '''

        def raised(stage):
            sums1, sums2 = self[stage].sums(concat=False)
            dem1, dem2 = self.mod.demand.stage(stage)
            if new:
                sums1[sums1 <= dem1] = np.nan
                sums2[sums2 <= dem2] = np.nan
            else:
                sums1 = self.combine_if(sums1, dem1, max)
                sums2 = self.combine_if(sums2, dem2, max)
            return [sums1, sums2]

        return [raised(i) for i in self.mod.dv.range_stage()]


    def nodes_by_layer(self, new=True) -> LayerDict:
        stages = self.nodes_by_stage(new)
        return EdgeConstraints.nodes_by_layer(self, stages, max)



class EdgeConstraintsContainer:

    def __init__(self, model):
        self.__mod = model
        self.__capacity = EdgeCapacity(model.dv.templates.stages(), model)
        self.__demand = EdgeDemand(model.dv.templates.stages(), model)

    @property
    def mod(self): return self.__mod
    @property
    def capacity(self): return self.__capacity
    @property
    def demand(self): return self.__demand

    @property
    def diff(self):
        diffs = [self.capacity[i] - self.demand[i] for i in range(0, len(self))]
        return EdgeConstraints(diffs)

    @property
    def true_diff(self):
        def stage_diff(i):
            return self.capacity.true[i] - self.demand.true[i]
            cap = self.capacity.true
            dem = self.demand.true
        return EdgeConstraints([stage_diff(i) for i in range(0, len(self))])

    @property
    def node_diff(self):
        cap_updates = self.capacity.raised_by_layer()
        dem_updates = self.demand.lowered_by_layer()













    def __len__(self):
        if len(self.capacity) != len(self.demand):
            raise ValueError(f"There's an issue. Capacity and demand are different lengths")
        return len(self.capacity)








