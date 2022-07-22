# Maintainer:     Ryan Young
# Last Modified:  Jul 21, 2022
import pandas as pd
import numpy as np
import textwrap

from tsopt.dv import DV
from tsopt.exceptions import InfeasibleLayerConstraint


class SourceData:
    """
    Responsible for processing user data into everthing needed to build a pyomo model
    for transportation problems.
    """
    def __init__(self,
            cost: list,
            capacity: str or pd.DataFrame or dict,
            demand: str or pd.DataFrame or dict,
            layers: list,
            excel_file=None,
        ):

        if type(excel_file) == str:
            self.excel_file = pd.ExcelFile(excel_file)
        elif type(excel_file) == pd.ExcelFile:
            self.excel_file = excel_file
        else:
            self.excel_file = None

        # Process user data into dataframes, handling unknown table formatting
        self.cost = self.process_cost_data(cost, layers)

        # Determine number of nodes in each layer based on cost table dimensions
        sizes = [df.shape[0] for df in self.cost] + [self.cost[-1].shape[1]]
        self.dv = DV(layers, sizes)

        # Set columns and indexes according to node names
        # Indexes are input nodes, columns are output nodes
        for i, df in enumerate(self.cost):
            df.index     = self.dv.nodes[i]
            df.columns   = self.dv.nodes[i+1]

        self.capacity = self.process_capacity_data(capacity)
        self.demand = self.process_demand_data(demand)

        self.capacity_sums = self.sum_constraint_data(self.capacity)
        self.demand_sums = self.sum_constraint_data(self.demand)

        self.final_demand_total = self.demand_sums[len(self.cost)]
        self.initial_capacity_total = self.capacity_sums[0]
        self.flow_capacity = min(self.capacity_sums.values())
        self.flow_demand = max(self.demand_sums.values())

        self.validate_constraints()


    # @property
    # def constraint_data(self):
        # all_dfs = [item for sublist in [self.capacity.values(), self.demand.values()] for item in sublist]
        # return {df.columns[0]: df for df in all_dfs}


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
            if df.shape[1] == 1 and not self.valid_dtype(df.iloc[0,0]):
                return 0, None
            if df.shape[0] == 1 and not self.valid_dtype(df.iloc[0,0]):
                return None, 0

            return None, None

        if not self.valid_dtype(df.iloc[1,1]):
            return (1 if not self.valid_dtype(df.iloc[2, 0]) else 0), 0

        return [(0 if not self.valid_dtype(df.iloc[0,1]) else None),
                (0 if not self.valid_dtype(df.iloc[1,0]) else None)]


    def valid_dtype(self, val) -> bool:
        """
        A foolproof way to determine if a value is a number.
        We need this (instead of type() or isnumeric())
        because pandas and numpy are annoying.
        """
        try:
            float(val)
            return True
        except ValueError:
            return False


    def read_file(self, table:str, idx=None, hdr=None) -> pd.DataFrame:
        """
        Param 'table' will be a filename if reading from csv or its own excel file,
        or a sheet name if reading from excel file declared upon object creation
        """
        if table.endswith('.csv'):
            return pd.read_csv(table, index_col=idx, header=hdr)
        if table.endswith('.xlsx'):
            return pd.read_excel(table, index_col=idx, header=hdr)

        return pd.read_excel(self.excel_file, table, index_col=idx, header=hdr)


    def process_file(self, table: str or pd.DataFrame) -> pd.DataFrame:
        '''
        Takes the name of an excel table or csv file,
        and returns a dataframe formatted properly, except
        for columns and indexes
        '''
        if type(table) == pd.DataFrame:
            return table
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
        df = self.strip_null_ending_cols_and_rows(df)
        df.index.name = None
        return df


    def strip_null_ending_cols_and_rows(self, df) -> pd.DataFrame:
        col_mask = df.notna().any(0)[::-1].cumsum()[::-1].astype(bool)
        row_mask = df.notna().any(1)[::-1].cumsum()[::-1].astype(bool)
        return df.loc[:,col_mask].T.loc[:,row_mask].T


    def process_cost_data(self, tables:list, layers:list) -> list:
        '''
        Given a list of strings or dataframes representing cost data,
        process them into dataframes and validate.
        '''
        if type(tables) != list:
            tables = [tables]
        assert len(tables) == len(layers)-1, \
                "Number of layers must be one more than number of cost stages"
        dfs = [self.process_file(table) for table in tables]

        for i in range(0, len(dfs)-1):
            assert dfs[i].shape[1] == dfs[i+1].shape[0], \
                    f"Number of columns in Stage {i} costs ({dfs[i].shape[1]}) ' \
                    f'and rows in Stage {i+1} costs ({dfs[i+1].shape[0]}) must match"

        return dfs


    def process_capacity_data(self, tables:dict or str) -> dict:
        '''
        Given a dict of capacity tables, keyed by layer number or name, process
        their values into dataframes, validate, and format rows and cols
        '''
        if type(tables) != dict:
            tables = {0: tables}

        assert (0 in tables or self.dv.layers[0] in [k.capitalize() for k in tables.keys() if type(k) == str]), \
                'Must provide a capacity constraint table for first layer'

        result = self.standardize_key_format(tables)
        for key, table in result.items():
            df = self.process_file(table)
            assert df.shape[1] == 1, f"Capacity table {key} must have only one column"
            assert len(self.dv.nodes[key]) == df.shape[0], \
                    f"Number of rows in Capacity {key} and Costs {key} must match"

            # Set index and cols based on node and layer names
            df.index = self.dv.nodes[key]
            df = df[0].astype('float')
            df.name = f'Capacity: {self.dv.layers[key]}'
            result[key] = df

        return result


    def process_demand_data(self, tables:dict or str) -> dict:
        '''
        Given a dict of demand tables, keyed by layer number or name, process
        their values into dataframes, validate, and format rows and cols
        '''
        if type(tables) != dict:
            tables = {len(self.cost): tables}

        assert (len(self.cost) in tables or self.dv.layers[-1] in [k.capitalize() for k in tables.keys() if type(k) == str]), \
                'Must provide a demand constraint table for first layer'

        result = self.standardize_key_format(tables)
        for key, table in result.items():
            df = self.process_file(table)
            assert df.shape[1] == 1, f"Demand table {key} must have only one column"
            assert len(self.dv.nodes[key]) == df.shape[0], \
                    f"Number of rows in Demand {key} and Costs {key} must match"

            # Set index and cols based on node and layer names
            df.index = self.dv.nodes[key]
            df = df[0].astype('float')
            df.name = f'Demand: {self.dv.layers[key]}'
            result[key] = df

        return result


    def standardize_key_format(self, vals:dict) -> dict:
        result = dict()
        for key, val in vals.items():
            if type(key) == str:
                key = key.capitalize()
                assert key in self.dv.layers, f"Key, '{key}' is not a valid layer"
                result[self.dv.layers.index(key)] = val
            elif type(key) == int:
                assert 0 <= key <= len(self.dv.layers)-1, f"Key, '{key}' must represent a valid layer"
                result[key] = val
            else:
                raise AssertionError(f"Invalid key, '{key}'. Must use int index of layer, or name of layer")

        assert len(vals.keys()) == len(result.keys()), 'Multiple of the same type of constraint on one layer'
        return result


    def sum_constraint_data(self, tables:dict) -> dict:
        result = dict()
        for key, df in tables.items():
            result[key] = df.values.sum()
        return result


    def validate_constraints(self) -> bool:
        '''
        Ensures constraints don't conflict with each other
        '''
        # PART 1: Find the capacity constraint with the lowest total capacity, and
        # make sure this total is >= the demand constraint with the maximum total demand
        if self.flow_capacity < self.flow_demand:
            min_cap_layr = self.dv.layers[
                [i for i in self.capacity_sums.keys() if self.capacity_sums[i] == self.flow_capacity][0]]
            max_dem_layr = self.dv.layers[
                [i for i in self.demand_sums.keys() if self.demand_sums[i] == self.flow_demand][0]]
            raise InfeasibleLayerConstraint(
                f'{min_cap_layr} capacity is less than {max_dem_layr} demand requirement'
            )

        # PART 2: For layers with multiple constraints, each node's demand must be less
        # than its capacity.
        constraint_index_intersection = list(self.capacity.keys() & self.demand.keys()) # '&' means intersection
        for i in constraint_index_intersection:
            cap = self.capacity[i]
            dem = self.demand[i]
            for node in cap.index:
                assert dem[node] <= cap[node], \
                        f'Node {node} is constrained to a capacity lower than its demand requirement'



    def plural(self, word:str) -> str:
        if word.endswith('y'):
            return word[:-1] + 'ies'
        if word.endswith('s'):
            return word + 'es'
        return word + 's'


    def comma_sep(self, iterable, max_len=5) -> str:
        if len(iterable) == 1:
            return str(iterable[0])
        elif len(iterable) <= max_len:
            main = iterable[:-1]
            last = iterable[-1]
            return ', '.join(main) + f' and {last}'
        else:
            return ', '.join(iterable[:max_len]) + f' (+{len(iterable)-max_len} more)'


    def dedent_wrap(self, s, prefix_mask='InfeasibleEdgeConstraint: ') -> str:
        s = prefix_mask + ' '.join(s.split())
        wraps = textwrap.wrap(textwrap.dedent(s.strip()), width=80)
        wraps[0] = wraps[0].replace(prefix_mask, '')
        return '\n'.join(wraps)


    def range_layer(self, start=0, end=0):
        return range(start, len(self)+end)

    def range_stage(self, start=0, end=0):
        return range(start, len(self.cost)+end)

    def range_node(self, idx, start=0, end=0):
        return range(start, len(self.dv.nodes[idx])+end)


    def __len__(self):
        return len(self.dv.layers)






