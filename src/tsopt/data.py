# Maintainer:     Ryan Young
# Last Modified:  Jul 16, 2022
import pandas as pd
import numpy as np

from tsopt.dv import DV


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
            min_output: str or pd.DataFrame or dict = None,
            max_input: str or pd.DataFrame or dict = None,
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
        if min_output != None:
            self.min_out = self.process_min_out_data(min_output)
        else:
            self.min_out = {}

        if max_input != None:
            self.max_in = self.process_max_in_data(max_input)
        else:
            self.max_in = {}

        self.validate_constraints()


    @property
    def constraint_data(self):
        all_dfs = [item for sublist in [self.capacity.values(), self.demand.values(), self.min_out.values(), self.max_in.values()] for item in sublist]
        return {df.columns[0]: df for df in all_dfs}


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
        df = df.dropna(axis=1, how='all')
        df = df.dropna(axis=0, how='all')
        df.index.name = None
        return df


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

        assert (0 in tables or self.dv.layers[0] in [k.capitalize() for k in tables.keys()]), 'Must provide a capacity constraint table for first layer'

        final = dict()
        for key, table in tables.items():
            if type(key) == str:
                key = key.capitalize()
                assert key in self.dv.layers, f"Invalid key for capacity, {key}"
                key = self.dv.layers.index(key)
            assert type(key) == int, 'Must use integer as key in capacity dictionary'
            assert 0 <= key < len(self.cost), \
                    f'Capacity constraints can only be made for layers {self.dv.layers[0]} through {self.dv.layers[-2]}'
            df = self.process_file(table)
            assert df.shape[1] == 1, f"Capacity table {key} must have only one column"
            assert len(self.dv.nodes[key]) == df.shape[0], \
                    f"Number of rows in Capacity {key} and Costs {key} must match"

            # Set index and cols based on node and layer names
            df.index = self.dv.nodes[key]
            df.columns = [f'Capacity: {self.dv.layers[key]}']
            final[key] = df

        return final


    def process_demand_data(self, tables:dict or str) -> dict:
        '''
        Given a dict of demand tables, keyed by layer number or name, process
        their values into dataframes, validate, and format rows and cols
        '''
        if type(tables) != dict:
            tables = {len(self.cost): tables}

        assert (len(self.cost) in tables or self.dv.layers[-1] in [k.capitalize() for k in tables.keys()]), 'Must provide a demand constraint table for first layer'

        final = dict()
        for key, table in tables.items():
            if type(key) == str:
                key = key.capitalize()
                assert key in self.dv.layers, f"Invalid key for demand, {key}"
                key = self.dv.layers.index(key)
            assert type(key) == int, 'Must use integer as key in demand dictionary'
            assert 0 < key <= len(self.cost), \
                    f"Demand constraints only allowed for layers 1 through {len(self.cost)}"
            df = self.process_file(table)
            assert df.shape[1] == 1, f"Demand table {key} must have only one column"
            assert len(self.dv.nodes[key]) == df.shape[0], \
                    f"Number of rows in Demand {key} and Costs {key} must match"

            # Set index and cols based on node and layer names
            df.index = self.dv.nodes[key]
            df.columns = [f'Demand: {self.dv.layers[key]}']
            final[key] = df

        return final


    def process_min_out_data(self, tables:dict or str) -> dict:
        '''
        Given a dict of minimum output tables, keyed by layer number or name,
        process their values into dataframes, validate, and format rows and cols
        '''
        if type(tables) != dict:
            tables = {0: tables}

        final = dict()
        for key, table in tables.items():
            if type(key) == str:
                key = key.capitalize()
                assert key in self.dv.layers, f'Invalid key for min output, {key}'
                key = self.dv.layers.index(key)
            assert type(key) == int, 'Must use integer as key in min output dictionary'
            assert 0 <= key < len(self.cost), \
                    f'Min output constraints can only be made for layers {self.dv.layers[0]} through {self.dv.layers[-2]}'
            df = self.process_file(table)
            assert df.shape[1] == 1, f'Min output table {key} must have only one column'
            assert len(self.dv.nodes[key]) == df.shape[0], \
                    f'Number of rows in min output {key} and costs {key} must match'

            # Set index and cols based on node and layer names
            df.index = self.dv.nodes[key]
            df.columns = [f'Min Output: {self.dv.layers[key]}']
            final[key] = df

        return final


    def process_max_in_data(self, tables:dict or str) -> dict:
        '''
        Given a dict of maximum input tables, keyed by layer number or name,
        process their values into dataframes, validate, and format rows and cols
        '''
        if type(tables) != dict:
            tables = {0: tables}

        final = dict()
        for key, table in tables.items():
            if type(key) == str:
                key = key.capitalize()
                assert key in self.dv.layers, f'Invalid key for max input, {key}'
                key = self.dv.layers.index(key)
            assert type(key) == int, 'Must use integer as key in max input dictionary'
            assert 1 <= key < len(self.cost)-1, \
                    f'Max input constraints can only be made for layers {self.dv.layers[1]} through {self.dv.layers[-1]}'
            df = self.process_file(table)
            assert df.shape[1] == 1, f'Max input table {key} must have only one column'
            assert len(self.dv.nodes[key]) == df.shape[0], \
                    f'Number of rows in max input {key} and costs {key} must match'

            # Set index and cols based on node and layer names
            df.index = self.dv.nodes[key]
            df.columns = [f'Max Input: {self.dv.layers[key]}']
            final[key] = df

        return final


    def validate_constraints(self) -> bool:
        '''
        Make sure that every layer with a capacity constraint has a
        total capacity (all nodes) greater than or equal to the total
        demand required by any later demand constraints
        '''
        for i, cap in self.capacity.items():
            later_demands = {k: v for k, v in self.demand.items() if k > i}
            for j, dem in later_demands.items():
                assert dem.iloc[:, 0].sum() <= cap.iloc[:, 0].sum(), \
                        f'{self.dv.layers[i]} capacity is less than {self.dv.layers[j]} demand. ' \
                        f'Total capacity at a layer must be greater than or equal to total demand ' \
                        f'at all following layers.'








