# Maintainer:     Ryan Young
# Last Modified:  Sep 26, 2022
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
    demand_type = None
    capacity_type = None

    def __init__(self, mod, demand=None, capacity=None):
        self._mod = mod

        cls = self.__class__

        if not cls.demand_type:
            self._demand = demand
        elif demand:
            self._demand = cls.demand_type(mod, demand)
        else:
            self._demand = cls.demand_type(mod)

        if not cls.capacity_type:
            self._capacity = capacity
        elif capacity:
            self._capacity = cls.capacity_type(mod, capacity)
        else:
            self._capacity = cls.capacity_type(mod)

    @property
    def mod(self): return self._mod
    @property
    def demand(self): return self._demand
    @property
    def capacity(self): return self._capacity

    @property
    def bounds(self):
        pass

    def __len__(self):
        return len(self.capacity)

    def _repr_html_(self):
        return self.bounds._repr_html_()



@dataclass
class NetworkValuesContainer:
    demand: float
    capacity: float

    @property
    def empty(self):
        return self.demand == None and self.capacity == None



class LayerValuesContainer(Container):
    demand_type = LayerDemandValues
    capacity_type = LayerCapacityValues

    @property
    def bounds(self):
        return pd.DataFrame(zip(self.demand, self.capacity),
                    index=self.mod.dv.abbrevs, columns=['dem','cap'])

    @property
    def diff(self):
        diffs = [self.capacity[i] - self.demand[i] for i in range(0, len(self))]
        return LayerValues(self.mod, diffs)



class NodesContainer(Container):
    demand_type = NodeDemand
    capacity_type = NodeCapacity

    @property
    def layer(self):
        return LayerValuesContainer(self.mod, self.demand.layer, self.capacity.layer)

    @property
    def bounds(self):
        return LayerNodeBounds(self.mod, self.demand, self.capacity)


    @property
    def diff(self):
        diffs = [self.capacity[i] - self.demand[i] for i in range(0, len(self))]
        return LayerNodes(self.mod, diffs)

    @property
    def true_diff(self):
        diffs = [self.capacity.true[i] - self.demand.true[i] for i in range(0, len(self))]
        return LayerNodes(self.mod, diffs)



class EdgesContainer(Container):
    demand_type = EdgeDemand
    capacity_type = EdgeCapacity

    @property
    def diff(self):
        diffs = [self.capacity[i] - self.demand[i] for i in range(0, len(self))]
        return EdgeConstraints(self.mod, diffs)

    @property
    def node(self):
        return NodesContainer(self.mod, self.demand.node, self.capacity.node)

    @property
    def bounds(self):
        ''' Returns MELTED bounds. Index: input. Cols: [output, demand, capacity]'''
        return StageEdgeBoundsMelted(self.mod, self.demand.melted, self.capacity.melted)

    def get_node_diff(self, new):
        cap_updates = self.capacity.nodes_by_layer(new).values()
        dem_updates = self.demand.nodes_by_layer(new).values()
        diffs = [cap - dem for cap,dem in zip(cap_updates, dem_updates)]
        return NodeConstraints(self.mod, diffs)

    @property
    def true_diff(self):
        def stage_diff(i):
            return self.capacity.true[i] - self.demand.true[i]
        return EdgeConstraints(self.mod, [stage_diff(i) for i in range(0, len(self.capacity))])


    # How can we get the diffs by stage conveniently?
    @property
    def node_diff(self):
        return self.get_node_diff(True)

    @property
    def true_node_diff(self):
        return self.get_node_diff(False)
