# Maintainer:     Ryan Young
# Last Modified:  Aug 20, 2022
import pandas as pd
import numpy as np
import re
from typing import List, Generator
from tsopt.vector_util import *

class ModelStructure:
    '''
    Validates and holds the metadata that describes a network's layers and nodes.
    Holds a list of each layer name, a list of abbreviations for each layer,
    and a list of the names of nodes in each layer.
    --
    Uses strict rules for getting/setting values so that nobody accidentally
    changes something that must not change.
    '''
    def __init__(self, layers:list, coefs:list, excel_file=None):
        self.__excel_file = excel_file

        self.__layers = tuple(layers)
        self.__abbrevs = tuple(["".join([s[0] for s in re.split("[ -\._]", layer)]).upper() for layer in self.__layers])
        assert len(set(self.__abbrevs)) == len(self.__abbrevs), \
                "\nLayer names must start with different letters.\n" \
                "You can work around this by using multiple words for layer names\n' \
                'For instance, 'manufacturing center' becomes 'MC')"
        assert [not s[0].isdigit() for s in self.__abbrevs], \
                f"Layer names must not start with a number."

        if not isinstance(coefs, list):
            coefs = [coefs]
        if not len(coefs) == len(self) - 1:
            raise ValueError(f"Must have {len(self)-1} cost tables")
        dfs = []
        for i, c in enumerate(coefs):
            if isinstance(c, list) or isinstance(c, tuple):
                df = pd.DataFrame(np.ones(c))
            elif isinstance(c, str):
                df = raw_df_from_file(c, self.excel_file)
            else:
                df = pd.DataFrame(c)

            df = df.astype(float)
            df.index = [ f'{self.abbrevs[i]}{n+1}' for n in range(0,len(df.index)) ]
            df.columns = [ f'{self.abbrevs[i+1]}{n+1}' for n in range(0,len(df.columns)) ]
            dfs.append(df)

        sizes = [df.nrows for df in dfs] + [dfs[-1].ncols]
        self.__nodes = tuple( [ tuple([ f'{abb}{i+1}' for i in range(0, length) ])
                for abb, length in zip(self.abbrevs, sizes) ] )
        self.__templates = TemplateGenerator(self.nodes)

        self.__cost = Cost(self.nodes, dfs)


    @property
    def excel_file(self): return self.__excel_file
    @property
    def layers(self): return self.__layers
    @property
    def abbrevs(self): return self.__abbrevs
    @property
    def nodes(self): return self.__nodes
    @property
    def templates(self): return self.__templates

    @property
    def sizes(self):
        return tuple([len(n) for n in self.__nodes])

    @property
    def cost(self): return self.__cost


    def range(self, idx=None, start=0, end=0):
        if idx == None:
            return range(start, len(self)+end)
        return range(start, len(self.__nodes[idx])+end)


    def range_stage(self, start=0, end=0):
        return self.range(end=end-1)


    def __len__(self):
        return len(self.__layers)


class TemplateGenerator:
    def __init__(self, nodes):
        self.__nodes = nodes

    def edges(self, fill=np.nan):
        return [ pd.DataFrame(fill, index=idx, columns=cols)
            for idx,cols in staged(self.__nodes)
        ]

    def vectors(self, fill=np.nan):
        return [ pd.Series(fill, index=nodes)
            for nodes in self.__nodes
        ]

    def bounds(self, fill_min=np.nan, fill_max=np.nan):
        return [ pd.concat( [
                    pd.Series(fill_min, index=nodes, name='min'),
                    pd.Series(fill_max, index=nodes, name='max')
                ],
                axis=1)
            for nodes in self.__nodes
        ]



class Cost(list):
    def __init__(self, nodes, *args):
        self.__nodes = nodes
        list.__init__(self, *args)

    @property
    def nodes(self):
        return self.__nodes


    def validate_frame(self, k, df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Cost table {k} must be of type pd.DataFrame")
        if df.shape != (len(self.nodes[k]), len(self.nodes[k+1])):
            raise ValueError(f"Invalid shape for cost table {k}")
        if tuple(df.index) != self.nodes[k]:
            raise ValueError(f"Invalid index for cost table {k}")
        if tuple(df.columns) != self.nodes[k+1]:
            raise ValueError(f"Invalid columns for cost table {k}")


    def __getitem__(self, k):
        df = list.__getitem__(self, k)
        self.validate_frame(k, df)
        return df


    def __setitem__(self, k, v):
        if not 0-len(self) <= k <= len(self):
            raise ValueError(f"Invalid stage, {k}")
        self.validate_frame(v)

        list.__setitem__(self, k, v)



