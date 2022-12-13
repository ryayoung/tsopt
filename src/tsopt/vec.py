from tsopt.base import (
    ModelAttr,
    Outline,
    Frame,
)
from pandas import Series, DataFrame, concat
from typing import Self


class _Concatable(list):
    """
    Rules
        Elements must be instances of DataFrame or Series
    """

    def concat(self) -> Frame:
        return concat([i for i in self])


class _NullFilterable(list):
    """
    Rules
        Elements must be instances of DataFrame or Series
    """

    def __init__(self, outline, data) -> None:
        self.outline = outline
        list.__init__(data)

    def notnull(self) -> Self:
        return self.__class__(self.outline, [item.dropna() for item in self])


class _FullFilterable(list):
    """
    Rules
        Elements must be instances of DataFrame or Series
    """

    def __init__(self, outline, data) -> None:
        self.outline = outline
        list.__init__(data)

    def full_only(self) -> Self:
        def check_full(item: Frame) -> Frame:
            if isinstance(item, DataFrame):
                return DataFrame() if item.isna().any().any() else item
            elif isinstance(item, Series):
                return Series() if item.isna().any() else item

        return self.__class__(self.outline, [check_full(item) for item in self])


class Vec(ModelAttr, list):
    """
    Rules
        Length should never change.
        Length should never be 0.
    """

    def __init__(
        self, outline: Outline, data: list | None = None, fill_val: float = 0.0
    ) -> None:
        super().__init__(outline)
        if data is None:
            list_data = self.default_template(fill_val)
        else:
            list_data = data
        list.__init__(list_data)

    def default_template(*args, **_) -> list:
        return []

    def _repr_html_(self):
        return "".join([item._repr_html() for item in self])

    def __sub__(self, other) -> Self:
        if not isinstance(other, self.__class__):
            raise TypeError(f"Can't subtract type {other.__class__.__name__} from self")
        if not len(self) == len(other):
            raise ValueError("Can't subtract two vecs of different length")
        if len(self) == 0:
            return self
        for elem_self, elem_other in zip(self, other):
            if not hasattr(elem_self, "__sub__") or not hasattr(elem_other, "__sub__"):
                raise ValueError("Each element in self and other must be subtractable")
        return self.__class__(
            self.outline,
            [elem_self - elem_other for elem_self, elem_other in zip(self, other)],
        )


class NodeVec(Vec, _Concatable, _NullFilterable, _FullFilterable):
    """ """

    def agg(self) -> Series:
        ...


class NodeBoundVec(Vec, _Concatable):
    """ """

    def diff(self) -> NodeVec:
        ...

    def agg(self) -> DataFrame:
        ...


class EdgeMeltVec(Vec, _Concatable, _NullFilterable, _FullFilterable):
    """ """

    def __sub__(self, other) -> Self:
        # subtract the 'val' column
        ...


class EdgeVec(Vec, _FullFilterable):
    """ """

    def melt(self) -> EdgeMeltVec:
        ...

    def full(self) -> Self:
        ...

    def agg(self) -> NodeVec:
        ...


class EdgeMeltBoundVec(Vec, _Concatable):
    """ """

    def diff(self) -> EdgeMeltVec:
        ...
