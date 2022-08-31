# Maintainer:     Ryan Young
# Last Modified:  Aug 30, 2022
import pandas as pd
import numpy as np
import re
from typing import List, Generator
from tsopt.vector_util import *

class ModelConstants:
    '''
    Validates and stores everything that cannot be changed after model instance
    creation.
    ---
    - Names, abbreviations, and node names for each layer
    - Edge cost data
    '''
    def __init__(self, layers:list, coefs:list, excel_file=None):

        def layers_to_abbrevs(layers) -> tuple:
            split_layer_to_parts = lambda layer: re.split("[ -\._]", layer)
            first_char_each_part = lambda layer_parts: [s[0] for s in layer_parts]

            abbrev_parts = [ first_char_each_part(split_layer_to_parts(layer)) for layer in layers ]
            return tuple("".join(parts).upper() for parts in abbrev_parts)


        def validate_layer_names(abbrevs) -> None:
            assert len(set(abbrevs)) == len(abbrevs), \
                    "\nLayer names must start with different letters.\n" \
                    "You can work around this by using multiple words for layer names\n' \
                    'For instance, 'manufacturing center' becomes 'MC')"
            assert [not s[0].isdigit() for s in abbrevs], \
                    f"Layer names must not start with a number."


        def node_labels(layer_idx, num_of_nodes) -> tuple:
            abbrev = self.abbrevs[layer_idx]
            # 1-based
            return tuple(abbrev + str(n + 1) for n in range(0, num_of_nodes))


        def coef_data_from_user_input(value) -> pd.DataFrame:
            if isinstance(value, list) or isinstance(value, tuple):
                df = pd.DataFrame(np.ones(value))
            elif isinstance(value, str):
                df = raw_df_from_file(value, self.excel_file)
            else:
                df = pd.DataFrame(value)

            return df.astype(float)


        self.__excel_file = excel_file

        self.__layers = tuple(layers)
        self.__abbrevs = layers_to_abbrevs(self.layers)

        validate_layer_names(self.abbrevs)

        # Process cost data
        if not isinstance(coefs, list):
            coefs = [coefs]
        if not len(coefs) == len(self) - 1:
            raise ValueError(f"Must have {len(self)-1} cost tables")

        dfs = []
        for i, c in enumerate(coefs):
            df = coef_data_from_user_input(c)
            df.index = node_labels(i, len(df.index))
            df.columns = node_labels(i+1, len(df.columns))
            dfs.append(df)

        nodes = [ df.index for df in dfs ] + [ dfs[-1].columns ]
        self.__nodes = tuple( tuple(group) for group in nodes )
        self.__stage_nodes = tuple((inp, out) for inp, out in staged(nodes))

        self.__templates = TemplateCreator(self.nodes)

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
    def stage_nodes(self): return self.__stage_nodes
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



class TemplateCreator:
    def __init__(self, nodes):
        self.__nodes = nodes

    @property
    def nodes(self): return self.__nodes

    # formerly edges
    def stages(self, fill=np.nan):
        return [ pd.DataFrame(fill, index=idx, columns=cols)
            for idx,cols in self.stage_nodes
        ]

    # formerly vectors
    def layers(self, fill=np.nan):
        return [ pd.Series(fill, index=nodes)
            for nodes in self.nodes
        ]

    def layer_bounds(self, fill_min=np.nan, fill_max=np.nan):
        return [ pd.concat( [
                    pd.Series(fill_min, index=nodes, name='min'),
                    pd.Series(fill_max, index=nodes, name='max')
                ],
                axis=1)
            for nodes in self.nodes
        ]
