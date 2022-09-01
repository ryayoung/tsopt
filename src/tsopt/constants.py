# Maintainer:     Ryan Young
# Last Modified:  Aug 31, 2022
import pandas as pd
import numpy as np
import re
from typing import List, Generator
from tsopt.vector_util import *
from tsopt.stage_based import *

class ModelConstants:
    '''
    Validates and stores everything that cannot be changed after model instance
    creation.
    ---
    - Names, abbreviations, and node names for each layer
    - Edge cost data
    '''
    def __init__(self, layers:list, coefs:list, mod):

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
            return tuple(abbrev + str(n + self.node_label_offset()) for n in range(0, num_of_nodes))


        def coef_data_from_user_input(value) -> pd.DataFrame:
            if isinstance(value, list) or isinstance(value, tuple):
                df = pd.DataFrame(np.ones(value))
            elif isinstance(value, str):
                df = raw_df_from_file(value, self.mod.excel_file)
            else:
                df = pd.DataFrame(value)

            return df.astype(float)


        self.__mod = mod

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

        self.__nodes = nodes_from_stage_dfs(dfs)
        self.__stage_nodes = tuple((inp, out) for inp, out in staged(self.nodes))

        self.__templates = TemplateCreator(self.nodes)

        self.__cost = Costs(dfs)


    @property
    def mod(self): return self.__mod
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

    @staticmethod
    def node_label_offset():
        return 0


    def range(self, idx=None, start=0, end=0):
        if idx == None:
            return range(start, len(self)+end)
        return range(start, len(self.__nodes[idx])+end)


    def range_stage(self, start=0, end=0):
        return self.range(end=end-1)


    def __len__(self):
        return len(self.__layers)




class TemplateCreator:
    def __init__(self, nodes):
        self.__nodes = nodes

    @property
    def nodes(self): return self.__nodes

    # formerly edges
    def stages(self, fill=np.nan):
        return [ pd.DataFrame(fill, index=idx, columns=cols)
            for idx,cols in staged(self.nodes)
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



temp = '''
    def validate_frame(self, k, df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Cost table {k} must be of type pd.DataFrame")
        if df.shape != (len(self.nodes[k]), len(self.nodes[k+1])):
            raise ValueError(f"Invalid shape for cost table {k}")
        if tuple(df.index) != self.nodes[k]:
            raise ValueError(f"Invalid index for cost table {k}")
        if tuple(df.columns) != self.nodes[k+1]:
            raise ValueError(f"Invalid columns for cost table {k}")
'''
