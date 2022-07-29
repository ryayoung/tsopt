# Maintainer:     Ryan Young
# Last Modified:  Jul 29, 2022
import attr
import pandas as pd
import numpy as np
from typing import List, Generator

from tsopt.dv import ModelStructure
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

        self.dv = ModelStructure(layers, sizes)

        self.cost = cost

        self.__capacity, self.__demand = dict(), dict()
        self.capacity = capacity
        self.demand = demand

        self.cap = Constraints(self.dv, 'Capacity')

        self.validate_constraints()


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
    def cost(self): return self.__cost

    @cost.setter
    def cost(self, tables):
        if tables == None:
            self.__cost = self.dv.templates.edges(fill=1)
            return

        # Validation
        if type(tables) != list:
            tables = [tables]
        assert len(tables) == len(self.dv.layers)-1, "len(cost) must be 1 less than len(layers)"
        dfs = [self.process_file(table) for table in tables]

        for (i_curr, i_nxt), (curr, nxt) in staged(enumerate(dfs)):
            assert ncols(curr) == nrows(nxt), \
                    f"Number of columns in Stage {i_curr} costs ({ncols(curr)}) ' \
                    f'and rows in Stage {i_nxt} costs ({nrows(nxt)}) must match"

        # Update self.dv, and create cost dfs
        sizes = [nrows(df) for df in dfs] + [ncols(dfs[-1])]
        self.dv = ModelStructure(self.dv.layers, sizes)
        for df, (in_nodes, out_nodes) in zip(dfs, staged(self.dv.nodes)):
            df.index, df.columns = in_nodes, out_nodes

        self.__cost = dfs


    @property
    def capacity(self): return self.__capacity

    @capacity.setter
    def capacity(self, new):
        if type(new) != dict:
            new = {0: new}

        self.__capacity = dict()
        new = self.standardize_key_format(new)
        for k, sr in new.items():
            sr = self.process_file(sr)[0].astype(float)
            nodes, layr = self.dv.nodes[k], self.dv.layers[k]
            assert len(nodes) == nrows(sr), f"Capacity {k} must have a row for each {layr}"
            sr.index, sr.name = nodes, 'Capacity'
            self.__capacity[k] = sr

        first_layr = self.dv.layers[0]
        assert 0 in self.__capacity.keys(), f'Capacity constraint required for {first_layr}'
        assert isfull(self.__capacity[0]), f'Capacity constraint required for ALL nodes in {first_layr}'

        self.validate_constraints()


    @property
    def demand(self): return self.__demand

    @demand.setter
    def demand(self, new):
        if type(new) != dict:
            new = {len(self.dv)-1: new}

        self.__demand = dict()
        new = self.standardize_key_format(new)
        for k, sr in new.items():
            sr = self.process_file(sr)[0].astype(float)
            nodes, layr = self.dv.nodes[k], self.dv.layers[k]
            assert len(nodes) == nrows(sr), f"Demand {k} must have a row for each {layr}"
            sr.index, sr.name = nodes, 'Demand'
            self.__demand[k] = sr

        last_layr, last_layr_idx = self.dv.layers[-1], len(self.dv)-1
        assert last_layr_idx in self.__demand.keys(), f'Demand constraint required for {last_layr}'
        assert isfull(self.__demand[last_layr_idx]), f'Demand constraint required for ALL nodes in {last_layr}'

        self.validate_constraints()


    @property
    def capacity_sum(self) -> dict:
        return {k: df.values.sum() for k,df in self.capacity.items() if isfull(df)}

    @property
    def demand_sum(self) -> dict:
        return {k: df.values.sum() for k,df in self.demand.items() if isfull(df)}

    @property
    def flow_capacity(self) -> int:
        return min(self.capacity_sum.values()) if len(self.capacity_sum) > 0 else None

    @property
    def flow_demand(self) -> int:
        return max(self.demand_sum.values()) if len(self.demand_sum) > 0 else None

    @property
    def flow_capacity_layer(self) -> str:
        sums = self.capacity_sum
        if len(sums) > 0:
            matches = [ self.dv.layers[k] for k in sums.keys() if sums[k] == self.flow_capacity ]
            return matches[0]
        return None

    @property
    def flow_demand_layer(self) -> str:
        sums = self.demand_sum
        if len(sums) > 0:
            matches = [ self.dv.layers[k] for k in sums.keys() if sums[k] == self.flow_demand ]
            return matches[-1]
        return None

    @property
    def flow_capacity_layer_idx(self) -> str:
        if self.flow_capacity_layer != None:
            return self.dv.layers.index(self.flow_capacity_layer)
        return None

    @property
    def flow_demand_layer_idx(self) -> str:
        if self.flow_demand_layer != None:
            return self.dv.layers.index(self.flow_demand_layer)
        return None


    def con_bounds(self) -> dict:
        return {k: pd.concat([ self.demand.get(k), self.capacity.get(k) ], axis=1)
            for k in self.dv.range() if k in self.demand or k in self.capacity
        }


    def sheet_format(self, df) -> list:
        """
        Returns location of headers and index col in dataframe of unknown format
        1. Headers (None, 0 or 1)
        2. Indexes (None or 0)
        --
        These values can be passed to pd.read_excel() and pd.read_csv() to
        accurately retrieve a data table from a source with unknown formatting
        """

        if 1 in df.shape:
            if df.shape[1] == 1 and not valid_dtype(df.iloc[0,0]):
                return 0, None
            if df.shape[0] == 1 and not valid_dtype(df.iloc[0,0]):
                return None, 0

            return None, None

        if not valid_dtype(df.iloc[1,1]):
            return (1 if not valid_dtype(df.iloc[2, 0]) else 0), 0

        return [(0 if not valid_dtype(df.iloc[0,1]) else None),
                (0 if not valid_dtype(df.iloc[1,0]) else None)]


    def read_file(self, table:str, idx=None, hdr=None) -> pd.DataFrame:
        """
        Param 'table' will be a filename if reading from csv or its own excel file,
        or a sheet name if reading from excel file declared upon object creation
        """
        if table.endswith('.csv'):
            return pd.read_csv(table, index_col=idx, header=hdr)
        if table.endswith('.xlsx'):
            return pd.read_excel(table, index_col=idx, header=hdr)
        if self.excel_file:
            return pd.read_excel(self.excel_file, table, index_col=idx, header=hdr)

        raise ValueError(f"Can't read file {table}")


    def process_file(self, table: str or pd.DataFrame) -> pd.DataFrame:
        '''
        Takes the name of an excel table or csv file,
        and returns a dataframe formatted properly, except
        for columns and indexes
        '''
        if type(table) == pd.DataFrame:
            return table
        if type(table) in [list, tuple]:
            return pd.DataFrame(table)
        # Start by reading in the file with headers=None, index_col=None.
        # This way ALL provided data will be placed INSIDE the dataframe as values
        df = self.read_file(table)

        # Find the headers and indexes, if any exist
        headers, indexes = self.sheet_format(df)

        # Now read the file again, passing index and header locations
        df = self.read_file(table, idx=indexes, hdr=headers)

        # Finally, remove empty columns and rows caused by the user
        # accidentally putting spaces in surrounding cells in excel file
        df = df.replace(' ', np.NaN)
        df = strip_null_ending_cols_and_rows(df)
        df.index.name = None
        return df


    def standardize_key_format(self, vals:dict) -> dict:
        result = dict()
        for key, val in vals.items():
            if type(key) == str:
                assert key in self.dv.layers, f"Key, '{key}' is not a valid layer"
                result[self.dv.layers.index(key)] = val
            elif type(key) == int:
                assert 0 <= key <= len(self.dv)-1, f"Key, '{key}' must represent a valid layer"
                result[key] = val
            else:
                raise AssertionError(f"Invalid key, '{key}'. Must use int index of layer, or name of layer")

        return result


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



    def __len__(self):
        return len(self.dv)


class VecDict(dict):
    '''
    A dictionary containing a series for each layer in a model.
    - Keys represent layer index. Values must have valid index for that layer.
    - Keys must be int, and values must be pd.Series'''
    def __init__(self, dv, *args):
        self.__dv = dv
        dict.__init__(self, *args)

    @property
    def dv(self):
        return self.__dv

    def __setitem__(self, k, v):
        if not isinstance(k, int):
            raise ValueError('Key in constraint dict must be int')
        if not isinstance(v, pd.Series):
            raise ValueError('Value in constraint dict must be pd.Series')
        if not 0 <= k < len(self.dv):
            raise ValueError(f'Key {k} does not represent a valid layer')
        if tuple(v.index) != tuple(self.dv.nodes[k]):
            raise ValueError(f"Constraint series {k} index must match nodes in {self.dv.layers[k]}")
        dict.__setitem__(self, k, v)



class Constraints(VecDict):
    def __init__(self, dv, name:str=None, excel_file=None, *args):
        self.__excel_file = excel_file
        self.__name = name if name else ""
        if len(args) == 0:
            args = tuple( [ { i: vec.rename(name) for i,vec in enumerate(dv.templates.vectors()) } ] )

        VecDict.__init__(self, dv, *args)

    @property
    def name(self): return self.__name

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


    def locate(self, k):
        if k == -1: return len(self.dv)-1
        if k in self: return k
        elif k in self.dv.layers:
            return self.dv.layers.index(k)
        raise ValueError(f"Key {k} invalid")


    def __getitem__(self, k):
        return VecDict.__getitem__(self, self.locate(k))


    def val_to_series(self, v):
        if type(v) == pd.Series:
            return v
        elif type(v) == pd.DataFrame:
            return v[v.columns[0]]
        elif type(v) in [list, tuple]:
            return pd.DataFrame(v)[0]
        elif type(v) == str:
            return self.process_file(v)
        else:
            raise ValueError(f'Invalid data type for constraint dict value {k}')


    def __setitem__(self, k, v):
        k = self.locate(k)

        sr = val_to_series(v)

        nodes = self.dv.nodes[k]
        if len(nodes) != nrows(sr):
            raise ValueError(f"Capacity {k} must have a row for each node")
        sr.index, sr.name = nodes, self.name

        VecDict.__setitem__(self, k, sr)


    def set_from_dict(self, data):
        ...

    def sheet_format(self):
        ...

    def read_file(self):
        ...

    def process_file(self):
        ...



