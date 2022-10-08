# Maintainer:     Ryan Young
# Last Modified:  Oct 07, 2022
import pandas as pd
from dataclasses import dataclass

class NodeSR(pd.Series):
    def _repr_html_(self):
        df = pd.DataFrame(self, columns=[self.name if self.name else ""])
        return df._repr_html_()

    def pop(self):
        updated = self.iloc[:-1]
        return NodeSR(updated)


class NodeBoundsDF(pd.DataFrame):
    pass


class EdgeDF(pd.DataFrame):

    def pop(self, axis=0):
        if axis == 0:
            updated = self.iloc[:-1]
        elif axis == 1:
            updated = self.iloc[:, :-1]
        return EdgeDF(updated)


class EdgeMeltedDF(pd.DataFrame):
    pass

class EdgeMeltedBoundsDF(pd.DataFrame):
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
    sr: NodeSR

    def __repr__(self):
        return self.sr.__repr__()

    def _repr_html_(self):
        return self.sr._repr_html_()

