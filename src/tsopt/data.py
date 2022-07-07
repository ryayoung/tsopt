# Maintainer:     Ryan Young
# Last Modified:  Jul 06, 2022
import pandas as pd
import numpy as np
import re

class SourceData:
    """
    Responsible for processing user data into everthing needed to build a pyomo model
    for 3 stage transportation problems.
    """
    def __init__(self,
            tbl_c1, tbl_c2, tbl_capacity, tbl_demand,
            from_csv=False, excel_file=None, stages=["plant", "distributor", "warehouse"]
        ):

        self.from_csv = from_csv
        self.excel_file = pd.ExcelFile(excel_file) if excel_file else None

        self.stages  = [stage.capitalize() for stage in stages]

        # Abbreviate stage names with the first letter of each word (space, period, dash, or underscore delimited)
        # So stage 'manufacturing-plant' is abbreviated as 'MP'
        self.stage_abbrevs = ["".join([s[0] for s in re.split("[ -\._]", stage)]).upper() for stage in self.stages]

        # Process user data into dataframes, handling unknown table formatting
        for saved_df, tblname in zip(
                ["c1", "c2", "capacity", "demand"],
                [tbl_c1, tbl_c2, tbl_capacity, tbl_demand]):

            # Start by reading in the file with headers=None, index_col=None.
            # This way ALL provided data will be placed INSIDE the dataframe as values
            df = self.read_file(tblname)

            # Find the headers and indexes, if any exist
            headers, indexes = self.sheet_format(df)

            # Now read the file again, passing index and header locations
            df = self.read_file(tblname, idx=indexes, hdr=headers)

            df.index.name = None
            setattr(self, saved_df, df)

        # Assign default names to capacity and demand tables if none given
        if not type(self.capacity.columns[0]) == str:
            self.capacity.columns = ["Capacity"]

        if not type(self.demand.columns[0]) == str:
            self.demand.columns = ["Requirement"]

        assert self.stage_abbrevs[0] != self.stage_abbrevs[1] != self.stage_abbrevs[2], \
                "Stage names must start with different letters.\n" \
                "You can work around this by using multiple words for stage names (Ex: 'manufacturing center' becomes 'MC')"

        assert [not s[0].isdigit() for s in self.stage_abbrevs], \
                f"Stage names must not start with a number. You provided {[s[0].isdigit() for s in self.stage_abbrevs]}"

        assert self.c1.shape[1] == self.c2.shape[0], \
                f"Number of columns in Stage 1 costs ({self.c1.shape[1]}) and rows in Stage 2 costs ({self.c2.shape[0]}) must match"

        assert self.c1.shape[0] == self.capacity.shape[0], \
                f"Number of rows in Stage 1 costs ({self.c1.shape[0]}) and rows in Capacity ({self.capacity.shape[0]}) must match"

        assert self.c2.shape[1] == self.demand.shape[0], \
                f"Number of rows in Stage 1 costs ({self.c2.shape[1]}) and rows in Demand ({self.demand.shape[0]}) must match"

        assert self.capacity.shape[1] == 1, \
                f"Capacity table should have only one column. {self.capacity.shape[1]} were given."

        assert self.demand.shape[1] == 1, \
                f"Demand table should have only one column {self.demand.shape[1]} were given."

        """
        Set row and col names for our dataframes.
        Naming convention: '<stage abbrev> + <location #>'
        So if stage 2 is 'trap-house', and there are 3 locations,
        then we want a list like '["TH1", "TH2", "TH3"]' to use in dataframes
        These will become the DV indexes in a TransshipModel
        """
        self.c1.index       = [f"{self.stage_abbrevs[0]}{n}" for n in range(1, len(list(self.c1.index)) + 1)]
        self.c1.columns     = [f"{self.stage_abbrevs[1]}{n}" for n in range(1, len(list(self.c1.columns)) + 1)]
        self.c2.index       = [f"{self.stage_abbrevs[1]}{n}" for n in range(1, len(list(self.c1.columns)) + 1)]
        self.c2.columns     = [f"{self.stage_abbrevs[2]}{n}" for n in range(1, len(list(self.c2.columns)) + 1)]
        self.capacity.index = [f"{self.stage_abbrevs[0]}{n}" for n in range(1, len(list(self.c1.index)) + 1)]
        self.demand.index   = [f"{self.stage_abbrevs[2]}{n}" for n in range(1, len(list(self.c2.columns)) + 1)]


    def sheet_format(self, df):
        """
        Returns location of headers and index col in dataframe of unknown format
        1. Headers (None, 0 or 1)
        2. Indexes (None or 0)
        --
        These values can be passed to pd.read_excel() and pd.read_csv() to
        accurately retrieve a data table from a source with unknown formatting
        """
        valid_d = [np.int64, np.float64, int]

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






