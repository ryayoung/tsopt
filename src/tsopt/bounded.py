from typing import TypeAlias, overload, Any
from tsopt.base import FlowVal, Frame, ModelAttr, Outline
from tsopt.vec import EdgeMeltVec, EdgeMeltBoundVec, NodeBoundVec, Vec
from numpy import nan

NotVec: TypeAlias = Frame | FlowVal | float


class Bounds(ModelAttr):
    """"""

    @overload
    def __init__(self, outline: Outline, dem: None, cap: None) -> None:
        ...
    @overload
    def __init__(self, outline: Outline, dem: NotVec, cap: NotVec) -> None:
        ...
    @overload
    def __init__(self, outline: Outline, dem: Vec, cap: Vec) -> None:
        ...
    def __init__(self, outline, dem = None, cap = None):
        super().__init__(outline)
        if dem is None and cap is None:
            self._init_template()
        elif dem is not None and cap is not None:
            self.dem = dem
            self.cap = cap
        else:
            raise ValueError("Both 'dem' and 'cap' must be passed")

    def _init_template(self) -> None:
        self.dem = 0.0
        self.cap = nan

    def diff(self) -> Vec | NotVec:
        return self.cap - self.dem  # All possible contents should have a valid __sub__


class VecBounds(Bounds):
    """"""

    @overload
    def __init__(self, outline: Outline, dem: None, cap: None) -> None:
        ...
    @overload
    def __init__(self, outline: Outline, dem: Vec, cap: Vec) -> None:
        ...
    def __init__(self, outline, dem = None, cap = None):
        super().__init__(outline, dem, cap)

    def _init_template(self) -> None:
        self.dem = Vec(self.outline, data=[])
        self.cap = Vec(self.outline, data=[], fill_val = nan)


class BoundedEdge(VecBounds):
    ...


class BoundedEdgeMelt(VecBounds):
    ...


class BoundedNode(VecBounds):
    ...


class BoundedLayer(Bounds):
    def __init__(self, outline: Outline, dem: FlowVal, cap: FlowVal) -> None:
        super().__init__(outline, dem, cap)

class BoundedFlow(Bounds):
    def __init__(self, outline: Outline, dem: FlowVal, cap: FlowVal) -> None:
        super().__init__(outline, dem, cap)

class BoundedNet(Bounds):
    ...







