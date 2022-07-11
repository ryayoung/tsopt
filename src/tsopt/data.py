# Maintainer:     Ryan Young
# Last Modified:  Jul 11, 2022
import pandas as pd
import numpy as np
import re
from typing import List

class SourceData:
    """
    Responsible for processing user data into everthing needed to build a pyomo model
    for 3 stage transportation problems.
    """
    def __init__(self,
            # stage_costs: List[str or pd.DataFrame],
            stage_costs: list,
            capacity: str or pd.DataFrame,
            demand: str or pd.DataFrame,
            layers: list,
            excel_file=None, from_csv=False,
        ):

        self.from_csv = from_csv
        self.excel_file = pd.ExcelFile(excel_file) if excel_file else None

        self.layers  = [stage.capitalize() for stage in layers]

        # Abbreviate stage names with the first letter of each word (space, period, dash, or underscore delimited)
        # So stage 'manufacturing-plant' is abbreviated as 'MP'
        self.layer_abbrevs = ["".join([s[0] for s in re.split("[ -\._]", stage)]).upper() for stage in self.layers]

        # Process user data into dataframes, handling unknown table formatting
        cost_names = [f'cost{i}' for i in range(0, len(stage_costs))]
        for name, table in zip(
                cost_names + ['capacity', 'demand'],
                stage_costs + [capacity, demand]):

            if (self.excel_file or self.from_csv) and type(table) == str:
                df = self.process_file(table)
            else:
                df = table

            setattr(self, name, df)

        # Put cost table instance variables into a list
        self.costs = []
        for i in range(0, len(stage_costs)):
            self.costs.append(getattr(self, f'cost{i}'))

        # Assign default names to capacity and demand
        self.capacity.columns = ["Capacity"]

        self.demand.columns = ["Requirement"]

        assert len(self.layer_abbrevs) == len(self.costs) + 1, \
                f"You provided {len(self.costs)} cost stages, so you must provide {len(self.costs)+1} layer names"

        assert len(set(self.layer_abbrevs)) == len(self.layer_abbrevs), \
                "Stage names must start with different letters.\n" \
                "You can work around this by using multiple words for stage names (Ex: 'manufacturing center' becomes 'MC')"

        assert [not s[0].isdigit() for s in self.layer_abbrevs], \
                f"Stage names must not start with a number."

        for i in range(0, len(self.costs) - 1):
            assert self.costs[i].shape[1] == self.costs[i+1].shape[0], \
                    f"Number of columns in Stage {i} costs ({self.costs[i].shape[1]}) and rows in Stage {i+1} costs ({self.costs[i+1].shape[0]}) must match"

        assert self.costs[0].shape[0] == self.capacity.shape[0], \
                f"Number of rows in Stage 0 costs ({self.costs[0].shape[0]}) and rows in Capacity ({self.capacity.shape[0]}) must match"

        assert self.costs[-1].shape[1] == self.demand.shape[0], \
                f"Number of rows in Stage {len(self.costs)-1} costs ({self.costs[-1].shape[1]}) and rows in Demand ({self.demand.shape[0]}) must match"

        assert self.capacity.shape[1] == 1, \
                f"Capacity table should have only one column. {self.capacity.shape[1]} were given."

        assert self.demand.shape[1] == 1, \
                f"Demand table should have only one column {self.demand.shape[1]} were given."

        """
        Set row and col names for our dataframes.
        Naming convention: '<stage abbrev> + <location #>'
        So if stage 2 is 'trap-house', and there are 3 locations,
        then we want a list like '["TH1", "TH2", "TH3"]' to use in dataframes
        """
        self.capacity.index     = [f"{self.layer_abbrevs[0]}{n}" for n in range(1, self.costs[0].shape[0] + 1)]
        self.demand.index       = [f"{self.layer_abbrevs[-1]}{n}" for n in range(1, self.costs[-1].shape[1] + 1)]
        for i, df in enumerate(self.costs):
            df.index     = [f'{self.layer_abbrevs[i]}{n}' for n in range(1, df.shape[0] + 1)]
            df.columns   = [f'{self.layer_abbrevs[i+1]}{n}' for n in range(1, df.shape[1] + 1)]


    def sheet_format(self, df):
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


    def valid_dtype(self, val):
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


    def read_file(self, table, idx=None, hdr=None):
        """
        Param 'table' will be a filename if reading from csv,
        or a sheet name if reading from excel
        """
        if self.from_csv == False:
            return pd.read_excel(self.excel_file, table, index_col=idx, header=hdr)
        else:
            return pd.read_csv(table, index_col=idx, header=hdr)


    def process_file(self, table):
        '''
        Takes the name of an excel table or csv file,
        and returns a dataframe formatted properly, except
        for columns and indexes
        '''
        # Start by reading in the file with headers=None, index_col=None.
        # This way ALL provided data will be placed INSIDE the dataframe as values
        df = self.read_file(table)

        # Find the headers and indexes, if any exist
        headers, indexes = self.sheet_format(df)

        # Now read the file again, passing index and header locations
        df = self.read_file(table, idx=indexes, hdr=headers)

        df.index.name = None
        return df






