# Maintainer:     Ryan Young
# Last Modified:  Sep 24, 2022

from copy import copy, deepcopy
import pandas as pd, numpy as np

from tsopt.vector_util import *
from tsopt.basic_types import *
from tsopt.layer_based import *
from tsopt.container import *

class EdgeDFs(StageList):
    dtype = ModDF

    @property
    def default_template(self):
        return self.mod.dv.template_stages()


    def set_element_format(self, idx, df):
        curr = super().__getitem__(idx)
        df = pd.DataFrame(df)
        df.index, df.columns, = curr.index, curr.columns
        return df.replace(-1, np.nan).astype(float)


    def node_iloc_slice(self, idx, node) -> (int, int):
        # Node can either be string or tuple
        # Returns node location with RELATIVE layer (0 for input, 1 for output)
        df = super().__getitem__(idx)
        if isinstance(node, str):
            if node.upper() in df.columns:
                node = (1, list(df.columns).index(node))
            elif node.upper() in df.index:
                node = (0, list(df.index).index(node))
        else:
            node = tuple(node)

        layer, idx = node
        if layer == 1:
            node_slice = (slice(None,None,None), idx)
            return node_slice
        elif layer == 0:
            return idx

    def loc_to_df_stage_and_slice(self, loc) -> (pd.DataFrame, tuple):
        '''
        STAGE: [stage:int] or NODE: [stage:int, node:str|tuple]
        Returns (stage, node_slice) or (stage, None)
        '''
        # -------------------------------------------------
        try:
            loc = tuple(loc)
        except Exception:
            loc = (loc, None)

        idx, node = loc[0], loc[1]
        if not 0 - len(self) <= idx <= len(self):
            raise ValueError(f"Invalid stage, {idx}")

        df = super().__getitem__(idx)

        node_slice = self.node_iloc_slice(idx, node) if node else None
        return df, idx, node_slice


    def __getitem__(self, loc):
        df, idx, node_slice = self.loc_to_df_stage_and_slice(loc)
        if node_slice == None:
            return df
        return df.iloc[node_slice]


    def __setitem__(self, loc, val):
        df, idx, node_slice = self.loc_to_df_stage_and_slice(loc)
        if node_slice == None:
            super().__setitem__(idx, val)
        else:
            if isinstance(val, int):
                val = float(val)
            else:
                val = np.array(val).astype(float)
            df.iloc[node_slice] = val


    def load(self, loc, filename, excel_file=None) -> None:
        excel = excel_file if excel_file else self.mod.excel_file
        self[loc] = raw_df_from_file(filename, excel)


    def _repr_html_(self):
        return "".join([df._repr_html_() for df in self])



class EdgeQuantities(EdgeDFs):
    ''' Stores solved-model quantities. '''

    @property
    def node(self):
        ''' Easy to calculate since output and input will always be equal for any node. '''
        srs = [df.sum(axis=1) for df in self] + [self[-1].sum(axis=0)]


class EdgeConstraints(EdgeDFs):

    def node(self, func, skipna=True, cast_type=NodeSRs):
        '''
        Convert to nodes by layer.
        -   For each layer, decide how to handle nulls. When skipna=True, nulls
            are ignored. When skipna=False, the entire row or column must be
            non-null, otherwise null is returned.
        -   Each flow layer will have two sets of node sums (inputs and outputs).
            So we must compare the two series' and pick which values to use.
            Param func is passed to combine_if() to pick values
        '''
        inputs = self[0].sum(axis=1, skipna=skipna)
        outputs = self[-1].sum(axis=0, skipna=skipna)
        flows = [
                combine_if(stg1.sum(axis=0, skipna=skipna), stg2.sum(axis=1, skipna=skipna), func)
            for stg1, stg2 in staged(self)
        ]
        layers = [inputs] + flows + [outputs]
        return cast_type(self.mod, layers)


    # def nodes_by_layer(self, stages, func) -> NodeSRs:
        # inputs = stages[0][0]
        # outputs = stages[-1][1]
        # flows = [ combine_if(s1[1], s2[0], func) for s1,s2 in staged(stages) ]
# 
        # all_layers = [inputs] + flows + [outputs]
        # ldict = {i: sr for i,sr in enumerate(all_layers)}
# 
        # return NodeSRs(self.mod, ldict)




class EdgeCapacity(EdgeConstraints):

    @property
    def node(self):
        return super().node(min, skipna=False, cast_type=NodeCapacity)

    @property
    def true(self):
        # WUT
        fill_val = self.mod.node.capacity.flow.val
        new_items = [df.fillna(fill_val) for df in self]
        return EdgeCapacity(self.mod, new_items)


    def nodes_by_stage(self, new=True) -> list:
        ''' Show nodes whose edges are fully constrained by edge upper bounds
        to a value lower than their original capacity '''

        def lowered(stage):
            sums1, sums2 = self[stage].sums(full=True, concat=False)
            cap1, cap2 = self.mod.node.capacity.stage(stage)
            if new:
                sums1[sums1 >= cap1] = np.nan
                sums2[sums2 >= cap2] = np.nan
            else:
                sums1 = combine_if(sums1, cap1, min)
                sums2 = combine_if(sums2, cap2, min)
            return [sums1, sums2]

        return [lowered(i) for i in self.mod.dv.range_stage()]


    def nodes_by_layer(self, new=True) -> NodeSRs:
        stages = self.nodes_by_stage(new)
        return EdgeConstraints.nodes_by_layer(self, stages, min)



class EdgeDemand(EdgeConstraints):

    @property
    def node(self):
        return super().node(max, skipna=True, cast_type=NodeDemand)

    @property
    def true(self):
        fill_val = 0
        new_items = [df.fillna(fill_val) for df in copy(self)]
        return EdgeCapacity(self.mod, new_items)


    def nodes_by_stage(self, new=True) -> list:
        ''' Show nodes which have edges constrained to a lower bound totaling
        to greater than the node's demand '''

        def raised(stage):
            sums1, sums2 = self[stage].sums(concat=False)
            dem1, dem2 = self.mod.node.demand.stage(stage)
            if new:
                sums1[sums1 <= dem1] = np.nan
                sums2[sums2 <= dem2] = np.nan
            else:
                sums1 = combine_if(sums1, dem1, max)
                sums2 = combine_if(sums2, dem2, max)
            return [sums1, sums2]

        return [raised(i) for i in self.mod.dv.range_stage()]


    def nodes_by_layer(self, new=True) -> NodeSRs:
        stages = self.nodes_by_stage(new)
        return EdgeConstraints.nodes_by_layer(self, stages, max)



class EdgeConstraintsContainer(ConstraintsContainer):
    capacity_type = EdgeCapacity
    demand_type = EdgeDemand

    @property
    def diff(self):
        diffs = [self.capacity[i] - self.demand[i] for i in range(0, len(self))]
        return EdgeConstraints(self.mod, diffs)


    def get_node_diff(self, new):
        cap_updates = self.capacity.nodes_by_layer(new).values()
        dem_updates = self.demand.nodes_by_layer(new).values()
        diffs = [cap - dem for cap,dem in zip(cap_updates, dem_updates)]
        return NodeConstraints(self.mod, diffs)

    @property
    def true_diff(self):
        def stage_diff(i):
            return self.capacity.true[i] - self.demand.true[i]
        return EdgeConstraints(self.mod, [stage_diff(i) for i in range(0, len(self.capacity))])


    # How can we get the diffs by stage conveniently?
    @property
    def node_diff(self):
        return self.get_node_diff(True)

    @property
    def true_node_diff(self):
        return self.get_node_diff(False)





    def __len__(self):
        return len(self.capacity)









