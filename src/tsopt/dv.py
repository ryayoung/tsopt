# Maintainer:     Ryan Young
# Last Modified:  Jul 29, 2022
import pandas as pd
import numpy as np
import re
from typing import List, Generator
from tsopt.vector_util import staged

class ModelStructure:
    '''
    Validates and holds the metadata that describes a network's layers and nodes.
    Holds a list of each layer name, a list of abbreviations for each layer,
    and a list of the names of nodes in each layer.
    --
    Uses strict rules for getting/setting values so that nobody accidentally
    changes something that must not change.
    '''
    def __init__(self, layers: list, sizes:list=None):
        if sizes == None:
            sizes = len(layers) * [1]
        self.__layers = tuple(layers)
        self.__sizes = tuple(sizes)

        self.update_derived_fields()


    @property
    def layers(self): return self.__layers
    @property
    def sizes(self): return self.__sizes
    @property
    def abbrevs(self): return self.__abbrevs
    @property
    def nodes(self): return self.__nodes
    @property
    def templates(self): return self.__templates

# 
    # @layers.setter
    # def layers(self, new):
        # self.__layers = tuple(new)
        # self.update_derived_fields()
# 
    # @sizes.setter
    # def sizes(self, new):
        # self.__sizes = tuple(new)
        # self.update_derived_fields()


    def update_derived_fields(self):
        ''' Called when layers or sizes is changed '''
        self.__abbrevs = tuple(["".join([s[0] for s in re.split("[ -\._]", layer)]).upper() for layer in self.__layers])
        assert len(set(self.__abbrevs)) == len(self.__abbrevs), \
                "\nLayer names must start with different letters.\n" \
                "You can work around this by using multiple words for layer names\n' \
                'For instance, 'manufacturing center' becomes 'MC')"

        assert [not s[0].isdigit() for s in self.__abbrevs], \
                f"Layer names must not start with a number."

        self.__nodes = tuple( [ [ f'{abb}{i+1}' for i in range(0, length) ]
                for abb, length in zip(self.__abbrevs, self.__sizes) ] )
        self.__templates = self.TemplateGenerator(self.__nodes)


    def range(self, idx=None, start=0, end=0):
        if idx == None:
            return range(start, len(self)+end)
        return range(start, len(self.__nodes[idx])+end)


    def range_stage(self, start=0, end=0):
        return self.range(end=end-1)


    def __len__(self):
        return len(self.__layers)


    def __eq__(self, other):
        if self.layers != other.layers: return False
        if self.sizes != other.sizes: return False


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

