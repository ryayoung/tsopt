# Maintainer:     Ryan Young
# Last Modified:  Oct 07, 2022
import pandas as pd
import numpy as np
import pulp as pl
from copy import copy

from tsopt.constants import *
from tsopt.types import *
from tsopt.edges import *
from tsopt.nodes import *
from tsopt.container import *
from tsopt.solved import *

class Model:

    def __init__(self,
            layers: list[str] = [],
            shape: list[int] = [],
            excel_file: str|pd.ExcelFile = None,
            units: str|None = 'units',
        ):

        self._layers = []
        self.layers = layers
        self._shape = []
        self.shape = shape
        self.nodes = [self.get_nodes(i) for i in range(len(self.layers))]

        self.excel_file = pd.ExcelFile(excel_file)
        self.units = units
        self.pl_mod = None
        self.costs = StageEdges(self)
        self.con = ConstraintsContainer(self)


    @property
    def layers(self): return self._layers
    @layers.setter
    def layers(self, new:list): self.update_layers(new)


    @property
    def shape(self): return self._shape
    @shape.setter
    def shape(self, new:list):
        # Can't add or remove shape this way
        assert len(new) == len(self._shape), "Incorrect length"
        for layer_idx, new_dim in enumerate(new):
            self.update_dimension(layer_idx, new_dim)


    @property
    def abbrevs(self):
        return [self.get_abbrev(l) for l in self._layers]
    @property
    def stage_nodes(self):
        return list(staged(self.nodes))
    @property
    def stage_edges(self):
        return [[[(i,o) for o in outs ] for i in inps ] for inps, outs in self.stage_nodes]


    def get_abbrev(self, layer):
        return "".join([ s[0].upper() for s in re.split("[ -\._]", layer) ])


    def get_nodes(self, idx) -> list:
        length = self.shape[idx]
        abbrev = self.abbrevs[layer_idx]
        return [abbrev + str(n + Global.base) for n in range(0, length)]


    def refresh_format(self):
        # Call this after updating layers or shape
        curr = copy(self.nodes)
        new = [self.get_nodes(i) for i in range(len(self))]
        self.nodes = new

        # Push/pop layers
        diff_layers = len(new) - len(curr)
        for _ in range(abs(diff_layers)):
            if diff_layers > 0:
                self.costs.push()
                self.con.push()
            elif diff_layers < 0:
                self.costs.pop()
                self.con.pop()

        # Push/pop nodes
        diff_nodes = [len(n) - len(c) for n,c in zip(new, curr)]
        for i, diff in enumerate(diff_nodes):
            if diff > 0:
                self.costs.push_nodes(layer=i, amount=diff)
                self.con.push_nodes(layer=i, amount=diff)
            elif diff < 0:
                self.costs.pop_nodes(layer=i, amount=diff)
                self.con.pop_nodes(layer=i, amount=diff)

        # Refresh where existing layer changed node format but not maybe not shape
        changed = [i for i in range(min(len(new), len(curr))) if new[i][0] != curr[i][0]]
        for i in changed:
            self.costs.refresh_nodes(i)
            self.con.refresh_nodes(i)


    def set_layers(self, new:list):
        curr = copy(self._layers)
        diff = len(new) - len(curr)
        if new == curr:
            return

        self._validate_layer_names(new)
        self._layers = new

        if diff > 0: self._shape += [1 for _ in range(diff)]
        elif diff < 0: self._shape = self._shape[:diff]
        self.refresh_format()


    def push_layer(self, new:str, size:int=1):
        self._validate_layer_names(self.layers + [new])
        self._layers += [new]
        self._shape += [size]
        self.nodes += [self.get_nodes(len(self.layers)-1)]
        self.costs.push()
        self.con.push()

    def pop_layer(self):
        self._layers = self._layers[:-1]
        self._shape = self._shape[:-1]
        self.nodes = self.nodes[:-1]
        self.costs.pop()
        self.con.pop()


    def update_dimension(self, idx:int, new:int):
        curr = copy(self._shape[idx])
        if new == curr:
            return

        diff = new - curr
        self._shape[idx] = new

        self.refresh_format()


    def _validate_layer_names(self, layers):
        abbrevs = [self.get_abbrev(l) for l in layers]
        assert len(set(abbrevs)) == len(abbrevs), \
                "\nLayer names must start with different letters.\n" \
                "You can work around this by using multiple words for layer names\n' \
                'For instance, 'manufacturing center' becomes 'MC')"
        assert [not s[0].isdigit() for s in abbrevs], \
                f"Layer names must not start with a number."


    def layer_index(self, val) -> int:
        ''' from int, layer name, or abbrev '''
        if isinstance(val, int):
            return val
        try:
            layers = [l.lower() for l in self.layers]
            return layers.index(val.lower())
        except Exception:
            abbrevs = [a.lower() for a in self.abbrevs]
            return abbrevs.index(val.lower())


    def node_str_to_layer_and_node_indexes(self, node) -> (int, int):
        ''' "B4" -> (1, 4), or "A3" -> (0, 3)'''
        node = node.lower()
        abb, node_idx = re.search(r"([a-zA-Z]+)(\d+)", node).groups()
        node_idx = int(node_idx)
        layer_idx = self.abbrevs.index(abb.upper())

        if node_idx+1 > len(self.nodes[layer_idx]):
            raise ValueError(f"Node index doesnt exist, {node}")

        return (layer_idx, node_idx)


    def range(self, idx=None, start=0, end=0):
        if idx == None:
            return range(start, len(self)+end)
        return range(start, len(self._nodes[idx])+end)


    def range_flow(self):
        return self.range(start=1, end=-1)


    def range_stage(self, start=0, end=0):
        return self.range(end=end-1)


    def template_stages(self, fill=np.nan):
        # formerly edges
        return [ pd.DataFrame(fill, index=idx, columns=cols)
            for idx,cols in staged(self.nodes)
        ]


    def template_layers(self, fill=np.nan):
        # formerly vectors
        return [ NodeSR(fill, index=nodes)
            for nodes in self.nodes
        ]


    def template_layer_bounds(self, fill_min=np.nan, fill_max=np.nan):
        return [ pd.concat( [
                    NodeSR(fill_min, index=nodes, name='min'),
                    NodeSR(fill_max, index=nodes, name='max')
                ],
                axis=1)
            for nodes in self.nodes
        ]


    @property
    def net(self): return self.con.net
    @property
    def node(self): return self.con.node
    @property
    def edge(self): return self.con.edge


    def var(self, var_name):
        return getattr(self.pl_mod, var_name)

    def add(self, arg):
        self.pl_mod += arg


    def sum_outflow(self, idx, node):
        ''' for node B3, return sum(B3[C1], B3[C2], B3[C3]) '''
        output_nodes = self.dv.nodes[idx + 1]
        return sum(self.var(node)[out] for out in output_nodes)


    def sum_inflow(self, idx, node):
        ''' for node B3, return sum(A1[B3], A2[B3], A3[B3]) '''
        input_nodes = self.dv.nodes[idx - 1]
        return sum(self.var(inp)[node] for inp in input_nodes)


    def set_network_constraints(self):
        '''
        Flow through each layer of network.
        - Only one value for the whole network is needed,
          since each layer will have the same flow
        '''
        if self.net.empty:
            return
        # Measure flow in stage 0 since it must be the same in all stages
        total_flow = sum( self.sum_outflow(0, node) for node in self.dv.nodes[0] )

        if self.net.cap:
            self.add(total_flow <= self.net.cap)
        if self.net.dem:
            self.add(total_flow >= self.net.dem)


    def set_node_constraints(self):
        '''
        User-defined bounds on flow through each node.
        - This can be defined in one of two ways:
            1. Sum of node's edges with the next layer (outflow)
            2. Sum of node's edges with the previous layer (inflow)
        - It DOESN'T matter which option is used, as long as we don't attempt
          an inflow sum on layer 0, or outflow sum in layer -1 (impossible).
        - Inner function, sum_func(), will choose sum_outflow() when
          constraining nodes in the first layer, and sum_inflow() for
          all other layers.
        '''
        sum_func = lambda i, node: self.sum_outflow(i,node) if i == 0 else self.sum_inflow(i,node)

        for i, coefs in enumerate(self.node.cap.notnull):
            for node in coefs.index:
                self.add(sum_func(i, node) <= coefs[node])

        for i, coefs in enumerate(self.node.dem.notnull):
            for node in coefs.index:
                self.add(sum_func(i, node) >= coefs[node])


    def set_edge_constraints(self):
        '''
        Set bounds on individual edges.
        - User-defined values are stored in the same
          format as costs.
        '''
        for df in self.edge.dem.melted.notnull:
            for i, inp_node, out_node, val in df.itertuples():
                self.add(self.var(inp_node)[out_node] >= val)

        for df in self.edge.cap.melted.notnull:
            for i, inp_node, out_node, val in df.itertuples():
                self.add(self.var(inp_node)[out_node] <= val)


    def build(self):
        # NEW MODEL
        self.pl_mod = pl.LpProblem('Transhipment', pl.LpMinimize)

        # DECISION VARS
        for ins, outs in self.dv.stage_nodes:
            for node in ins:
                setattr(self.pl_mod, node, pl.LpVariable.dicts(node, list(outs), lowBound=0, upBound=None, cat='Integer'))

        # OBJECTIVE
        total_cost = 0
        for stg, df in enumerate(self.edge.bounds):
            for inp, out in zip(df.inp, df.out):
                total_cost += self.var(inp)[out] * self.dv.costs[stg].loc[inp, out]
        self.add(total_cost)

        # USER-DEFINED CONSTRAINTS
        self.set_network_constraints()
        self.set_node_constraints()
        self.set_edge_constraints()

        # FLOW CONSTRAINT
        '''
        In models with more than 2 layers, make sure that in middle
        layers (ones which take inflow and produce outflow), each
        node's inflow must equal its outflow.
        '''
        for flow in self.dv.range_flow():
            for node in self.dv.nodes[flow]:
                self.add(self.sum_inflow(flow, node) == self.sum_outflow(flow, node))


    def solve(self) -> SolvedModel:
        status = self.pl_mod.solve(pl.PULP_CBC_CMD(msg=0)) # Solve, silencing messages
        if pl.LpStatus[status] == 'Optimal':
            return SolvedModel(self)
        else:
            raise ValueError(f'Status: ', pl.LpStatus[status])


    def run(self) -> SolvedModel:
        self.build()
        return self.solve()


    def display(self):
        zipped = zip(self.dv.layers, [len(nodes) for nodes in self.dv.nodes])
        print(*[f'{length}x {plural(layer)}' for layer, length in zipped], sep=' -> ')
        print()
        print("---- COSTS ----")
        for i, df in enumerate(self.dv.costs):
            print(f'{self.dv.layers[i]} -> {self.dv.layers[i+1]}')
            display(df)

        print("---- NODE CONSTRAINTS ----")
        display(self.node)

        dfs = self.edge.bounds
        for i, df in enumerate(dfs):
            df.index = df.inp + " -> " + df.out
            dfs[i] = dfs[i].drop(columns=['inp', 'out'])
            dfs[i] = dfs[i][(~dfs[i].dem.isna()) | (~dfs[i].cap.isna())]

        dfs = [df for df in dfs if df.shape.rows > 0]
        if len(dfs) > 0:
            print("---- EDGE CONSTRAINTS ----")
            display(*dfs)


    def __len__(self):
        return len(self.layers)














