# Maintainer:     Ryan Young
# Last Modified:  Dec 06, 2022

import pandas as pd, numpy as np

from tsopt.types import *
from tsopt.nodes import *

class StageEdges(ListData):

    @property
    def default_template(self):
        return self.mod.template_stages()

    def set_element_format(self, idx, df):
        curr = super().__getitem__(idx)
        if isinstance(df, int) or isinstance(df, float):
            curr.iloc[:] = df
            return curr
        df = pd.DataFrame(df)
        df.index, df.columns, = self.mod.stage_nodes[idx]
        return df.replace(-1, np.nan).astype(float)

    @property
    def melted(self):
        return StageEdgesMelted(self.mod, self)

    @staticmethod
    def find_node(df, loc) -> (int, int):
        if isinstance(loc, tuple):
            return loc
        # Loc must be string
        node, cols, rows = loc.upper(), list(df.columns), list(df.index)
        if node in cols:
            return 1, cols.index(node)
        elif node in rows:
            return 0, rows.index(node)


    def node_iloc_slice(self, idx, loc) -> (int, int):
        # Node can either be string or tuple
        # Returns node location with RELATIVE layer (0 for input, 1 for output)
        df = super().__getitem__(idx)
        loc = tuple(self.find_node(df, i) for i in loc)

        vals = [slice(None,None,None),slice(None,None,None)]
        for axis, idx in loc:
            vals[axis] = idx

        return tuple(vals)


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

        idx, loc = loc[0], loc[1:]

        if not 0 - len(self) <= idx <= len(self):
            raise ValueError(f"Invalid stage, {idx}")

        df = super().__getitem__(idx)

        node_slice = self.node_iloc_slice(idx, loc) if loc[0] else None
        return df, idx, node_slice


    def __getitem__(self, loc):
        df, idx, node_slice = self.loc_to_df_stage_and_slice(loc)
        if node_slice == None:
            return df
        return df.iloc[node_slice]


    def __setitem__(self, loc, val):
        df, idx, node_slice = self.loc_to_df_stage_and_slice(loc)
        if node_slice == None:
            val = self.set_element_format(idx, val)
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


    def sync_length(self, fill_val=np.nan):
        diff = len(self.mod.nodes) - (1 + len(self))
        if diff > 0:
            for _ in range(diff):
                idxs, cols = self.mod.stage_nodes[len(self)]
                self.append(self.cls_dtype(fill_val, index=idxs, columns=cols))
        elif diff < 0:
            for _ in range(diff):
                self.pop()


    def push_nodes(self, layer, n, fill_val=np.nan):
        ''' If flow layer, we must push to columns of [layer-1] and to index of [layer] '''
        abbrev = self.mod.abbrevs[layer]
        if layer == 0:
            self[0] = self[0].push(abbrev, n, axis=0, fill_val=fill_val)
        elif layer == len(self):
            self[-1] = self[-1].push(abbrev, n, axis=1, fill_val=fill_val)
        else:
            self[layer-1] = self[layer-1].push(abbrev, n, axis=1, fill_val=fill_val)
            self[layer] = self[layer].push(abbrev, n, axis=0, fill_val=fill_val)


    def pop_nodes(self, layer, n):
        ''' If flow layer, we must pop from columns of [layer-1] and from index of [layer] '''
        if layer == 0:
            self[0] = self[0].pop(n, axis=0)
        elif layer == len(self):
            self[-1] = self[-1].pop(n, axis=1)
        else:
            self[layer-1] = self[layer-1].pop(n, axis=1) # Columns
            self[layer] = self[layer].pop(n, axis=0) # Indexes


    def refactor_nodes(self, layer_idx=None):
        ''' Use this only when names change, not when shape changes '''
        nodes = self.mod.nodes
        if not layer_idx:
            for i, df in enumerate(self):
                self[i].index = nodes[i]
                self[i].columns = nodes[i+1]
        else:
            layer_nodes = self.mod.nodes[layer_idx]
            if layer_idx == 0:
                self[0].index = layer_nodes
            elif layer_idx == len(self):
                self[-1].columns = layer_nodes
            else:
                self[layer_idx-1].columns = layer_nodes
                self[layer_idx].index = layer_nodes



class StageEdgeCosts(StageEdges):

    @property
    def default_template(self):
        return self.mod.template_stages(fill=0.0)

    def sync_length(self, fill_val=0.0):
        super().sync_length(fill_val)

    def push_nodes(self, layer, n, fill_val=0.0):
        super().push_nodes(layer, n, fill_val)


    def set_element_format(self, idx, df):
        curr = super().__getitem__(idx)
        if isinstance(df, int) or isinstance(df, float):
            curr.iloc[:] = float(df)
            return curr
        df = pd.DataFrame(df)
        df.index, df.columns, = self.mod.stage_nodes[idx]
        return df.replace(-1, 0).astype(float)



class StageEdgesMelted(ListData):
    dtype = EdgeMeltedDF
    def __init__(self, mod, stage_edges):
        dfs = [df.reset_index(
                ).melt(id_vars='index'
                ).rename(columns={'index':'inp','variable':'out','value':'val'})
            for df in stage_edges]
        super().__init__(mod, dfs)


    @property
    def notnull(self):
        return ListData(self.mod, [self.cls_dtype(df[~df.val.isna()]) for df in self])

    @property
    def series(self):
        dfs = [df for df in self]
        for i, df in enumerate(dfs):
            dfs[i].index = dfs[i].input + dfs[i].output
            dfs[i] = dfs[i]['val']
        return dfs


class StageEdgeBoundsMelted(ListData):
    dtype = EdgeMeltedBoundsDF
    def __init__(self, mod, demand_edges_melted, capacity_edges_melted):
        dfs = [pd.merge(dem, cap, how='inner', on=['inp', 'out']).rename(columns={'val_x':'dem', 'val_y':'cap'})
               for dem, cap in zip(demand_edges_melted, capacity_edges_melted)]
        super().__init__(mod, dfs)



class EdgeQuantities(StageEdges):
    ''' Stores SOLVED-MODEL quantities. '''

    @property
    def node(self):
        ''' Easy to calculate since output and input will always be equal for any node. '''
        return LayerNodes(self.mod, [df.sum(axis=1) for df in self] + [self[-1].sum(axis=0)])


class EdgeConstraints(StageEdges):

    @property
    def notnull(self):
        return ListData(self.mod, [EdgeDF(df[~df.val.isna()]) for df in self.melted])



    def node(self, func, skipna=True, cast_type=None):
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
        fill_val = self.mod.node.cap.flow.val
        new_items = [df.fillna(fill_val) for df in self]
        return EdgeCapacity(self.mod, new_items)


    def nodes_by_stage(self, new=True) -> list:
        ''' Show nodes whose edges are fully constrained by edge upper bounds
        to a value lower than their original capacity '''

        def lowered(stage):
            sums1, sums2 = self[stage].sums(full=True, concat=False)
            cap1, cap2 = self.mod.node.cap.stage(stage)
            if new:
                sums1[sums1 >= cap1] = np.nan
                sums2[sums2 >= cap2] = np.nan
            else:
                sums1 = combine_if(sums1, cap1, min)
                sums2 = combine_if(sums2, cap2, min)
            return [sums1, sums2]

        return [lowered(i) for i in self.mod.range_stage()]


    def nodes_by_layer(self, new=True) -> LayerNodes:
        stages = self.nodes_by_stage(new)
        return super().nodes_by_layer(stages, min)



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
            dem1, dem2 = self.mod.node.dem.stage(stage)
            if new:
                sums1[sums1 <= dem1] = np.nan
                sums2[sums2 <= dem2] = np.nan
            else:
                sums1 = combine_if(sums1, dem1, max)
                sums2 = combine_if(sums2, dem2, max)
            return [sums1, sums2]

        return [raised(i) for i in self.mod.range_stage()]


    def nodes_by_layer(self, new=True) -> LayerNodes:
        stages = self.nodes_by_stage(new)
        return super().nodes_by_layer(stages, max)



