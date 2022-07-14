# Maintainer:     Ryan Young
# Last Modified:  Jul 11, 2022
import pandas as pd
import numpy as np
import re
from typing import List


class DV:
    def __init__(self, layers: list, sizes:list):
        self.__layers = [layer.capitalize() for layer in layers]
        self.__abbrevs = ["".join([s[0] for s in re.split("[ -\._]", layer)]).upper() for layer in layers]
        self.__sizes = sizes

        assert len(set(self.__abbrevs)) == len(self.__abbrevs), \
                "\nLayer names must start with different letters.\n" \
                "You can work around this by using multiple words for layer names\n' \
                'For instance, 'manufacturing center' becomes 'MC')"

        assert [not s[0].isdigit() for s in self.__abbrevs], \
                f"Layer names must not start with a number."

    @property
    def layers(self):
        return self.__layers

    @property
    def abbrevs(self):
        return self.__abbrevs

    @property
    def nodes(self):
        try:
            return self.__nodes
        except AttributeError:
            if self.__sizes != None:
                self.__nodes = [
                        [f'{abbrev}{i+1}' for i in range(0, length)]
                    for abbrev, length in zip(self.abbrevs, self.sizes)
                ]
                return self.__nodes

    @property
    def sizes(self):
        return self.__sizes

    @sizes.setter
    def sizes(self, new):
        assert len(new) == len(self.__layers), 'Must provide a size for each layer'
        del self.__nodes
        self.__sizes = new



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

        sizes = [df.shape[0] for df in self.cost] + [self.cost[-1].shape[1]]
        self.dv = DV(layers, sizes)

        # Set columns and indexes according to node names
        # Indexes are input nodes, columns are output nodes
        for i, df in enumerate(self.cost):
            df.index     = self.dv.nodes[i]
            df.columns   = self.dv.nodes[i+1]

        self.capacity = self.process_capacity_data(capacity)
        self.demand = self.process_demand_data(demand)

        self.validate_capacity_demand()


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
        Param 'table' will be a filename if reading from csv,
        or a sheet name if reading from excel
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
        Given a dict of capacity tables, keyed by layer number, process
        their values into dataframes, validate, and format rows and cols
        '''
        if type(tables) != dict:
            tables = {0: tables}

        assert 0 in tables, 'Must provide a capacity constraint table for first layer'

        final = dict()
        for key, table in tables.items():
            assert type(key) == int, 'Must use integer as key in capacity dictionary'
            assert 0 <= key < len(self.cost), \
                    f"Capacity constraints can only be made for layers 0 through {len(self.cost)-1}"
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
        Given a dict of demand tables, keyed by layer number, process
        their values into dataframes, validate, and format rows and cols
        '''
        if type(tables) != dict:
            tables = {len(self.cost): tables}

        assert len(self.cost) in tables, 'Must provide a demand constraint table for final layer'

        final = dict()
        for key, table in tables.items():
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


    def validate_capacity_demand(self) -> bool:
        # Any total layer capacity must be greater than
        # or equal to all demand that follows
        for i, cap in self.capacity.items():
            later_demands = {k: v for k, v in self.demand.items() if k > i}
            for j, dem in later_demands.items():
                assert dem.iloc[:, 0].sum() <= cap.iloc[:, 0].sum(), \
                        f'{self.dv.layers[i]} capacity is less than {self.dv.layers[j]} demand. ' \
                        f'Total capacity at a layer must be greater than or equal to total demand ' \
                        f'at all following layers. 








