# Maintainer:     Ryan Young
# Last Modified:  Sep 26, 2022

import pandas as pd


class ListData(list):
    dtype = None
    def __init__(self, mod, data=None):
        self._mod = mod
        if not data:
            data = self.default_template
        if self.cls_dtype:
            data = [self.cls_dtype(i) for i in data]
        super().__init__(data)

    @property
    def mod(self):
        return self._mod

    @property
    def default_template(self):
        return []

    @property
    def cls_dtype(self):
        return self.__class__.dtype


    def set_element_dtype(self, val):
        if self.cls_dtype:
            return self.cls_dtype(val)
        return val


    def set_element_format(self, idx, v):
        return v

    def __setitem__(self, idx, val):
        val = self.set_element_format(idx, val)
        val = self.set_element_dtype(val)
        super().__setitem__(idx, val)


    def _repr_html_(self):
        return "".join([item._repr_html_() for item in self])



class LayerList(ListData):
    def __len__(self):
        return len(self.mod)


class StageList(ListData):
    def __len__(self):
        return len(self.mod) - 1


