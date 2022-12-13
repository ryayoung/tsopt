from dataclasses import dataclass
from pandas import Series, DataFrame
from typing import TypeAlias

SomethingReallyGood = 5

Frame: TypeAlias = DataFrame | Series


@dataclass
class Outline:
    layers: tuple[str, ...]
    shape: tuple[int, ...]
    abbrevs: tuple[str, ...]
    nodes: tuple[tuple[str, ...], ...]
    """
    TODO:
        Default templates
        Index finding methods
        Range/stage methods
    """


class ModelAttr:
    """ """

    def __init__(self, outline: Outline) -> None:
        self.outline = outline

    @property
    def layers(self) -> tuple[str, ...]:
        return self.outline.layers

    @property
    def shape(self) -> tuple[int, ...]:
        return self.outline.shape

    @property
    def abbrevs(self) -> tuple[str, ...]:
        return self.outline.abbrevs

    @property
    def nodes(self) -> tuple[tuple[str, ...], ...]:
        return self.outline.nodes


class FlowVal(float):
    """ """

    def __new__(cls, value, idx):
        return float.__new__(cls, value)

    def __init__(self, value, idx) -> None:
        float.__init__(value)
        self.idx = idx


class BoundsDF(ModelAttr, DataFrame):
    """ """

    def __init__(self, outline: Outline, *args, **kwargs) -> None:
        super().__init__(outline)
        DataFrame.__init__(*args, **kwargs)

    def diff(self):
        if "cap" not in self.columns or "dem" not in self.columns:
            raise ValueError(
                "Bound DF must have columns 'dem' and 'cap' to calculate difference"
            )
        df = self.copy()
        df["val"] = df.cap - df.dem
        df = df.drop(columns=["cap", "dem"])
        return df
