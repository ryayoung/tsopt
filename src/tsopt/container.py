# Maintainer:     Ryan Young
# Last Modified:  Oct 04, 2022
from dataclasses import dataclass
from tsopt.edges import *
from tsopt.nodes import *
from tsopt.values import *


class Container:
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
    dem_type = None
    cap_type = None

    def __init__(self, mod, demand=None, capacity=None):
        self._mod = mod

        cls = self.__class__

        if not cls.dem_type:
            self._dem = demand
        elif demand:
            self._dem = cls.dem_type(mod, demand)
        else:
            self._dem = cls.dem_type(mod)

        if not cls.cap_type:
            self._cap = capacity
        elif capacity:
            self._cap = cls.cap_type(mod, capacity)
        else:
            self._cap = cls.cap_type(mod)

    @property
    def mod(self): return self._mod
    @property
    def dem(self): return self._dem
    @property
    def cap(self): return self._cap

    @property
    def bounds(self):
        pass

    def __len__(self):
        return len(self.cap)

    def _repr_html_(self):
        return self.bounds._repr_html_()



@dataclass
class NetworkValuesContainer:
    dem: float
    cap: float

    @property
    def empty(self):
        return self.dem == None and self.cap == None



class LayerValuesContainer(Container):
    dem_type = LayerDemandValues
    cap_type = LayerCapacityValues

    @property
    def bounds(self):
        return pd.DataFrame(zip(self.dem, self.cap),
                    index=self.mod.dv.abbrevs, columns=['dem','cap'])

    @property
    def diff(self):
        diffs = [self.cap[i] - self.dem[i] for i in range(0, len(self))]
        return LayerValues(self.mod, diffs)



class NodesContainer(Container):
    dem_type = NodeDemand
    cap_type = NodeCapacity

    @property
    def layer(self):
        return LayerValuesContainer(self.mod, self.dem.layer, self.cap.layer)

    @property
    def bounds(self):
        return LayerNodeBounds(self.mod, self.dem, self.cap)


    @property
    def diff(self):
        diffs = [self.cap[i] - self.dem[i] for i in range(0, len(self))]
        return LayerNodes(self.mod, diffs)

    @property
    def true_diff(self):
        diffs = [self.cap.true[i] - self.dem.true[i] for i in range(0, len(self))]
        return LayerNodes(self.mod, diffs)



class EdgesContainer(Container):
    dem_type = EdgeDemand
    cap_type = EdgeCapacity

    @property
    def diff(self):
        diffs = [self.cap[i] - self.dem[i] for i in range(0, len(self))]
        return EdgeConstraints(self.mod, diffs)

    @property
    def node(self):
        return NodesContainer(self.mod, self.dem.node, self.cap.node)

    @property
    def bounds(self):
        ''' Returns MELTED bounds. Index: input. Cols: [output, demand, capacity]'''
        return StageEdgeBoundsMelted(self.mod, self.dem.melted, self.cap.melted)

    def get_node_diff(self, new):
        cap_updates = self.cap.nodes_by_layer(new).values()
        dem_updates = self.dem.nodes_by_layer(new).values()
        diffs = [cap - dem for cap,dem in zip(cap_updates, dem_updates)]
        return NodeConstraints(self.mod, diffs)

    @property
    def true_diff(self):
        def stage_diff(i):
            return self.cap.true[i] - self.dem.true[i]
        return EdgeConstraints(self.mod, [stage_diff(i) for i in range(0, len(self.cap))])


    # How can we get the diffs by stage conveniently?
    @property
    def node_diff(self):
        return self.get_node_diff(True)

    @property
    def true_node_diff(self):
        return self.get_node_diff(False)
