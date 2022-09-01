# Maintainer:     Ryan Young
# Last Modified:  Aug 31, 2022
import pandas as pd, numpy as np
from typing import List, Generator

from tsopt.layer_based import *
from tsopt.constants import *
from tsopt.exceptions import *
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
        self.dv = ModelConstants(layers, coefs, self)

        self.__capacity = Capacity(self)
        self.__demand = Demand(self)
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
