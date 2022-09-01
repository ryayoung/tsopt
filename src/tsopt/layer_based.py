# Maintainer:     Ryan Young
# Last Modified:  Aug 31, 2022

import pandas as pd, numpy as np

from tsopt.vector_util import *
from tsopt.constants import *

class LayerDict(dict):
    '''
    A dictionary containing a series for each layer in a model.
    - Keys stored as integers, representing layer index.
    - Values must be pd.Series with index matching that layer's nodes.
    - If dv.layers == ['foo', 'bar', 'blah'], only keys 0, 1, and 2 are valid.
        - If layer 'bar' has 3 nodes, the index of the series at key 3
            must be ['B1', 'B2', 'B3']
    '''
    def __init__(self, mod, *args):
        self.__mod = mod
        dict.__init__(self, *args)

    @property
    def mod(self): return self.__mod

    def locate_key(self, k):
        if k in self.mod.dv.layers:
            return self.mod.dv.layers.index(k)
        elif k in self:
            return k
        elif isinstance(k, int):
            if k < 0:
                return len(self)-k
        raise ValueError(f"Key {k} invalid")


    def __getitem__(self, key):
        valid_key = self.locate_key(key)
        return dict.__getitem__(self, valid_key)


    def __setitem__(self, k, v):
        k = self.locate_key(k)
        if not 0 <= k < len(self):
            raise ValueError(f'Key {k} does not represent a valid layer')
        dict.__setitem__(self, k, v)



class Constraints(LayerDict):
    def __init__(self, mod, *args):
        if len(args) == 0:
            args = tuple( [ { i: vec for i,vec in enumerate(mod.dv.templates.layers()) } ] )
        LayerDict.__init__(self, mod, *args)

    @property
    def min(self):
        return min(self.sums.values())

    @property
    def max(self):
        return max(self.sums.values())

    @property
    def sums(self):
        sums = {k : v.sum() if is_frame(v) else np.nan for k,v in self.items()}
        return LayerDict(self.mod, sums)


    def stage(self, stg, concat=False):
        if concat:
            return pd.concat([self[stg], self[stg+1]])

        return [self[stg], self[stg+1]]


    def val_to_series(self, v):
        if isinstance(v, str):
            sr = raw_sr_from_file(v, self.mod.excel_file)
            sr = sr.replace(-1, np.nan)
            return sr
        if isinstance(v, pd.DataFrame):
            return v[v.columns[0]]
        try:
            test_if_iterable = iter(v)
            return pd.Series(v)
        except Exception:
            raise ValueError(f'Invalid data type for constraint value {k}')


    def locate_node(self, key, node):
        ''' Assume key is integer, and already validated '''
        available_nodes = dict.__getitem__(self, key).index
        if isinstance(node, int):
            try:
                return available_nodes[node]
            except Exception:
                raise ValueError(f"Invalid node, {node}, for layer {key}")

        node = node.upper()
        if node not in available_nodes:
            raise ValueError(f"Invalid node, {node}, for layer {key}")

        return node


    def locate_series_and_node(self, loc):
        key, node = loc
        sr = LayerDict.__getitem__(self, key)

        valid_key = self.locate_key(key)
        node = self.locate_node(valid_key, node)

        return (sr, node)


    def __getitem__(self, loc):
        if not isinstance(loc, tuple):
            return LayerDict.__getitem__(self, loc)

        sr, node = self.locate_series_and_node(loc)
        return sr[node]


    def set_series(self, key, val):
        key = self.locate_key(key)
        sr = self.val_to_series(val)

        nodes = self.mod.dv.nodes[key]
        if len(nodes) != sr.nrows:
            raise ValueError(f"{key} must have a row for each node")
        sr.index = nodes
        LayerDict.__setitem__(self, key, sr)


    def __setitem__(self, loc, val):
        if not isinstance(loc, tuple):
            return self.set_series(loc, val)

        if isinstance(val, str):
            raise ValueError(f"Invalid value for item {loc}")

        sr, node = self.locate_series_and_node(loc)

        sr[node] = val



    def set_from_dict(self, data):
        for k,v in data.items():
            self[k] = v



class Flow:
    def __init__(self, val, series, index, layer):
        self.val = val
        self.series = series
        self.index = index
        self.layer = layer


class Capacity(Constraints):

    @property
    def full(self):
        full = {k:v for k,v in self.items() if v.isfull()}
        return Capacity(self.mod, full)

    @property
    def flow(self):
        # Return lowest capacity where the entire layer is full
        val = self.full.min
        key = [k for k,v in self.full.sums.items() if v == val][0]
        sr = self[key]
        layer = self.mod.dv.layers[key]
        return Flow(val, sr, key, layer)



class Demand(Constraints):

    @property
    def full(self):
        full = {k:v for k,v in self.items() if v.isfull()}
        return Demand(self.mod, full)

    @property
    def flow(self):
        # Return highest demand even if layer isn't full
        val = self.max
        key = [k for k,v in self.sums.items() if v == val][-1]
        sr = self[key]
        layer = self.mod.dv.layers[key]
        return Flow(val, sr, key, layer)
