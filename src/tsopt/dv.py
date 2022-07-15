# Maintainer:     Ryan Young
# Last Modified:  Jul 11, 2022
import re

class DV:
    '''
    Validates and holds the metadata that describes a network's layers and nodes.
    Holds a list of each layer name, a list of abbreviations for each layer,
    and a list of the names of nodes in each layer.
    --
    Uses strict rules for getting/setting values so that nobody accidentally
    changes something that must not change.
    '''
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
