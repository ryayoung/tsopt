# Maintainer:     Ryan Young
# Last Modified:  Sep 01, 2022
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
        ):
        self.units = units if units != None else 'units'
        self.excel_file = excel_file
        if isinstance(self.excel_file, str):
            self.excel_file = pd.ExcelFile(self.excel_file)

        self.dv = ModelConstants(layers, cost, self)

        self.node = NodeConstraintsContainer(self, capacity, demand)


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
