# Maintainer:     Ryan Young
# Last Modified:  Sep 26, 2022
import pandas as pd
import numpy as np
import re
from typing import List, Generator
from tsopt.util import *
from tsopt.types import *
from tsopt.edges import *
from tsopt.nodes import *

class ModelConstants:
    '''
    Validates and stores everything that cannot be changed after model instance
    creation.
    ---
    - Names, abbreviations, and node names for each layer
    - Edge cost data
    '''
    def __init__(self, mod, layers:list, coefs:list):

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


        self._mod = mod

        self._layers = tuple(layers)
        self._abbrevs = layers_to_abbrevs(self.layers)

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

        self._cost = StageEdges(self.mod, dfs)
        self._nodes = nodes_from_stage_dfs(dfs)
        self._stage_nodes = tuple((inp, out) for inp, out in staged(self.nodes))
        self._stage_edges = tuple( tuple( tuple( (inp, out) for out in outs ) for inp in inps )
            for inps, outs in self.stage_nodes
        )

    @property
    def mod(self): return self._mod
    @property
    def layers(self): return self._layers
    @property
    def abbrevs(self): return self._abbrevs
    @property
    def nodes(self): return self._nodes
    @property
    def stage_nodes(self): return self._stage_nodes
    @property
    def stage_edges(self): return self._stage_edges
    @property
    def cost(self): return self._cost

    @property
    def sizes(self):
        return tuple([len(n) for n in self._nodes])

    @staticmethod
    def node_label_offset():
        return 0


    def layer_index(self, val) -> int:
        ''' from int, layer name, or abbrev '''
        if isinstance(val, int):
            return val
        try:
            layers = [l.lower() for l in self.layers]
            return layers.index(val.lower())
        except Exception:
            abbrevs = [a.lower() for a in self.abbrevs]
            return abbrevs.index(val.lower())


    def node_str_to_layer_and_node_indexes(self, node) -> (int, int):
        ''' "B4" -> (1, 4), or "A3" -> (0, 3)'''
        node = node.lower()
        abb, node_idx = re.search(r"([a-zA-Z]+)(\d+)", node).groups()
        node_idx = int(node_idx)
        layer_idx = self.abbrevs.index(abb.upper())

        if node_idx+1 > len(self.nodes[layer_idx]):
            raise ValueError(f"Node index doesnt exist, {node}")

        return (layer_idx, node_idx)


    def range(self, idx=None, start=0, end=0):
        if idx == None:
            return range(start, len(self)+end)
        return range(start, len(self._nodes[idx])+end)


    def range_flow(self):
        return self.range(start=1, end=-1)


    def range_stage(self, start=0, end=0):
        return self.range(end=end-1)


    def template_stages(self, fill=np.nan):
        # formerly edges
        return [ pd.DataFrame(fill, index=idx, columns=cols)
            for idx,cols in staged(self.nodes)
        ]


    def template_layers(self, fill=np.nan):
        # formerly vectors
        return [ NodeSR(fill, index=nodes)
            for nodes in self.nodes
        ]


    def template_layer_bounds(self, fill_min=np.nan, fill_max=np.nan):
        return [ pd.concat( [
                    NodeSR(fill_min, index=nodes, name='min'),
                    NodeSR(fill_max, index=nodes, name='max')
                ],
                axis=1)
            for nodes in self.nodes
        ]


    def __len__(self):
        return len(self._layers)
