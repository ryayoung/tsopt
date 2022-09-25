# Maintainer:     Ryan Young
# Last Modified:  Sep 24, 2022
import pandas as pd
from dataclasses import dataclass


class NetworkValues:
    def __init__(self, mod, demand=None, capacity=None):
        self._mod = mod
        self._demand = demand
        self._capacity = capacity

    @property
    def mod(self): return self._mod
    @property
    def demand(self): return self._demand
    @property
    def capacity(self): return self._capacity

    @property
    def empty(self):
        return self.demand == None and self.capacity == None



class ModSR(pd.Series):
    def _repr_html_(self):
        df = pd.DataFrame(self, columns=[self.name if self.name else ""])
        return df._repr_html_()

class ModDF(pd.DataFrame):
    pass


@dataclass
class FlowVal:
    idx: int
    val: float

    def __repr__(self):
        return self.val.__repr__()

@dataclass
class FlowSeries:
    idx: int
    sr: pd.Series or ModSR

    def __repr__(self):
        return self.sr.__repr__()


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

    def __getitem__(self, idx):
        v = super().__getitem__(idx)
        if self.cls_dtype:
            if type(v) != self.cls_dtype:
                raise ValueError(f'Value at {idx} is incorrect type')
        return v


    def __setitem__(self, idx, val):
        val = self.set_element_format(idx, val)
        val = self.set_element_dtype(val)
        super().__setitem__(idx, val)



class LayerList(ListData):
    def __len__(self):
        return len(self.mod)


class StageList(ListData):
    def __len__(self):
        return len(self.mod) - 1



class ConstraintsContainer:
    '''
    Important trick: We want this class's children to inherit the init method below.
    This isn't possible, however, since the capacity and demand variables will
    be of different types depending on the child type (NodeConstraintsContainer would
    hold NodeCapacity and NodeDemand, for instance, whereas EdgeConstr... would hold
    EdgeCapacity etc.). To solve this, we store the class types as class variables,
    and access them dynamically inside the init method using 'self.__class__'. This
    way, child classes can replace their redundant init methods with simple class
    variable declarations, capacity_type and demand_type.
    '''
    capacity_type = None
    demand_type = None

    def __init__(self, mod, capacity=None, demand=None):
        self._mod = mod

        cls = self.__class__
        if capacity or not cls.capacity_type: self._capacity = capacity
        else: self._capacity = cls.capacity_type(mod)

        if demand or not cls.demand_type: self._demand = demand
        else: self._demand = cls.demand_type(mod)

    @property
    def mod(self): return self._mod
    @property
    def capacity(self): return self._capacity
    @property
    def demand(self): return self._demand

