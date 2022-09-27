# Maintainer:     Ryan Young
# Last Modified:  Sep 26, 2022
import pandas as pd
from dataclasses import dataclass

class NodeSR(pd.Series):
    def _repr_html_(self):
        df = pd.DataFrame(self, columns=[self.name if self.name else ""])
        return df._repr_html_()


class NodeBoundsDF(pd.DataFrame):
    pass


class EdgeDF(pd.DataFrame):
    pass


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

