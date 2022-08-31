# Maintainer:     Ryan Young
# Last Modified:  Aug 30, 2022
import pandas as pd
import numpy as np
from typing import List, Generator

from tsopt.constants import ModelConstants
from tsopt.exceptions import InfeasibleLayerConstraint
from tsopt.text_util import *
from tsopt.vector_util import *


class SourceData:
    """
    Responsible for processing user data into everthing needed to build a pyomo model
    for transportation problems.
    """
    def __init__(self,
            layers: list,
            cost: list = None,
            capacity: str or pd.DataFrame or dict = None,
            demand: str or pd.DataFrame or dict = None,
            excel_file=None,
            units: str = None,
            sizes: list = None,
        ):
        self.units = units if units != None else 'units'
        self.sizes = sizes
        self.excel_file = excel_file

        coefs = cost if cost else sizes
        self.dv = ModelConstants(layers, coefs, self.excel_file)

        self.__capacity = Capacity(self.dv, self.excel_file)
        self.__demand = Demand(self.dv, self.excel_file)
        if capacity:
            self.capacity = capacity
        if demand:
            self.demand = demand


    @property
    def excel_file(self): return self.__excel_file

    @excel_file.setter
    def excel_file(self, new):
        if new == None:
            self.__excel_file = None
        elif type(new) == str:
            self.__excel_file = pd.ExcelFile(new)
        elif type(new) == pd.ExcelFile:
            self.__excel_file = new
        else:
            raise ValueError("Invalid data type for 'excel_file' argument")


    @property
    def cost(self): return self.dv.cost


    @property
    def capacity(self): return self.__capacity

    @capacity.setter
    def capacity(self, new):
        if isinstance(new, dict):
            self.__capacity.set_from_dict(new)
            return

        if isinstance(new, list) or isinstance(new, tuple):
            assert len(new) == len(self), f"Invalid number of capacity constraints"
            for i, val in enumerate(new):
                self.__capacity[i] = val
            return

        self.__capacity[0] = new


    @property
    def demand(self): return self.__demand

    @demand.setter
    def demand(self, new):
        if isinstance(new, dict):
            self.__demand.set_from_dict(new)
            return

        if isinstance(new, list) or isinstance(new, tuple):
            assert len(new) == len(self), f"Invalid number of demand constraints"
            for i, val in enumerate(new):
                self.__demand[i] = val
            return

        self.__demand[-1] = new



    @property
    def constraint_df_bounds(self):
        return {k: pd.concat([ self.demand.get(k), self.capacity.get(k) ], axis=1)
            for k in self.dv.range() if k in self.demand.full or k in self.capacity.full
        }


    def __len__(self):
        return len(self.dv)



class LayerDict(dict):
    '''
    A dictionary containing a series for each layer in a model.
    - Keys stored as integers, representing layer index.
    - Values must be pd.Series with index matching that layer's nodes.
    - If dv.layers == ['foo', 'bar', 'blah'], only keys 0, 1, and 2 are valid.
        - If layer 'bar' has 3 nodes, the index of the series at key 3
            must be ['B1', 'B2', 'B3']
    '''
    def __init__(self, dv, *args):
        self.__dv = dv
        dict.__init__(self, *args)

    @property
    def dv(self):
        return self.__dv

    def locate(self, k):
        if k < 0:
            return len(self)-k
        if k in self: return k
        elif k in self.dv.layers:
            return self.dv.layers.index(k)
        raise ValueError(f"Key {k} invalid")


    def __getitem__(self, k):
        valid_key = self.locate(k)
        return dict.__getitem__(self, valid_key)

    def __setitem__(self, k, v):
        k = self.locate(k)
        if not 0 <= k < len(self):
            raise ValueError(f'Key {k} does not represent a valid layer')
        dict.__setitem__(self, k, v)



class Constraints(LayerDict):
    def __init__(self, dv, excel_file, *args):
        self.excel_file = excel_file

        if len(args) == 0:
            args = tuple( [ { i: vec for i,vec in enumerate(dv.templates.layers()) } ] )
        LayerDict.__init__(self, dv, *args)


    @property
    def min(self):
        return min(self.sums.values())

    @property
    def max(self):
        return max(self.sums.values())

    @property
    def sums(self):
        is_frame = lambda val: isinstance(val, pd.core.generic.NDFrame)
        sums = {k : v.sum() if is_frame(v) else np.nan for k,v in self.items()}
        return LayerDict(self.dv, sums)


    def val_to_series(self, v):
        if isinstance(v, str):
            return raw_sr_from_file(v, self.excel_file)
        if isinstance(v, pd.DataFrame):
            return v[v.columns[0]]
        try:
            test_if_iterable = iter(v)
            return pd.Series(v)
        except Exception:
            raise ValueError(f'Invalid data type for constraint value {k}')


    def __setitem__(self, k, v):

        k = self.locate(k)
        sr = self.val_to_series(v)

        nodes = self.dv.nodes[k]
        if len(nodes) != sr.nrows:
            raise ValueError(f"{k} must have a row for each node")
        sr.index = nodes
        LayerDict.__setitem__(self, k, sr)


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
        return Capacity(self.dv, self.excel_file, full)

    @property
    def flow(self):
        # Return lowest capacity where the entire layer is full
        val = self.full.min
        key = [k for k,v in self.full.sums.items() if v == val][0]
        sr = self[key]
        layer = self.dv.layers[key]
        return Flow(val, sr, key, layer)



class Demand(Constraints):

    @property
    def full(self):
        full = {k:v for k,v in self.items() if v.isfull()}
        return Demand(self.dv, self.excel_file, full)

    @property
    def flow(self):
        # Return highest demand even if layer isn't full
        val = self.max
        key = [k for k,v in self.sums.items() if v == val][-1]
        sr = self[key]
        layer = self.dv.layers[key]
        return Flow(val, sr, key, layer)




saved = """
    def validate_constraints(self) -> bool:
        '''
        Ensures constraints don't conflict with each other
        '''
        if not self.flow_demand_layer or not self.flow_capacity_layer:
            return
        # PART 1: Find the capacity constraint with the lowest total capacity, and
        # make sure this total is >= the demand constraint with the maximum total demand
        if self.flow_capacity < self.flow_demand:
            raise InfeasibleLayerConstraint(
                f'{self.flow_capacity_layer} capacity is less than {self.flow_demand_layer} demand requirement'
            )

        # PART 2: For layers with multiple constraints, each node's demand must be less
        # than its capacity.
        constraint_index_intersection = list(self.capacity.keys() & self.demand.keys()) # '&' means intersection
        for i in constraint_index_intersection:
            sr = (self.capacity[i] - self.demand[i]).dropna()
            bad_nodes = tuple(sr[sr == False].index)
            if len(bad_nodes) > 0:
                raise InfeasibleLayerConstraint(
                    f"{plural(self.dv.layers[i])} {comma_sep(bad_nodes)}'s capacity is less than its demand"
                )
"""
