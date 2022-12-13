# Maintainer:     Ryan Young
# Last Modified:  Dec 06, 2022

import pandas as pd


class ListData(list):
    def __init__(self, mod, data=None):
        self._mod = mod
        if not data:
            data = self.default_template
        super().__init__(data)

    @property
    def mod(self):
        return self._mod

    @property
    def default_template(self):
        ...

    def _repr_html_(self):
        return "".join([item._repr_html_() for item in self])


