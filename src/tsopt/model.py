# Maintainer:     Ryan Young
# Last Modified:  Oct 07, 2022
import pandas as pd
import numpy as np
import pulp as pl
from dataclasses import dataclass

from tsopt.constants import *
from tsopt.types import *
from tsopt.edges import *
from tsopt.nodes import *
from tsopt.container import *
from tsopt.solved import *

class Model:

    def __init__(self,
            layers: list[str] = [],
            dimensions: list[int] = [],
            excel_file: str|pd.ExcelFile = None,
            units: str|None = 'units',
        ):

        self._layers = []
        self.layers = layers
        self._dimensions = []
        self.dimensions = dimensions

        self.excel_file = pd.ExcelFile(excel_file)
        self.units = units
        self.pl_mod = None
        self.con = ConstraintsContainer(self)

    @property
    def layers(self): return self._layers
    @property
    def dimensions(self): return self._dimensions

    @layers.setter
    def layers(self, new:list):
        for layer in new:
            self.push_layer(layer)

    @dimensions.setter
    def dimensions(self, new:list):
        assert len(new) == len(self._dimensions), "Incorrect length"
        for layer_idx, new_dim in enumerate(new):
            self.update_dimension(layer_idx, new_dim)


    def update_layers(self, new):
        curr = self._layers
        def layers_to_abbrevs(layers) -> tuple:
            split_layer_to_parts = lambda layer: re.split("[ -\._]", layer)
            first_char_each_part = lambda layer_parts: [s[0] for s in layer_parts]

            abbrev_parts = [ first_char_each_part(split_layer_to_parts(layer)) for layer in layers ]
            return tuple("".join(parts).upper() for parts in abbrev_parts)

        self.abbrevs = layers_to_abbrevs(new)

        def validate_layer_names(abbrevs) -> None:
            assert len(set(abbrevs)) == len(abbrevs), \
                    "\nLayer names must start with different letters.\n" \
                    "You can work around this by using multiple words for layer names\n' \
                    'For instance, 'manufacturing center' becomes 'MC')"
            assert [not s[0].isdigit() for s in abbrevs], \
                    f"Layer names must not start with a number."

        validate_layer_names(self.abbrevs)
        self._layers = new

        if len(new) > len(curr):
            self._dimensions.append(1)
        if len(new) < len(curr):
            self._dimensions.append(1)


    def update_dimension(self, idx, new):
        curr = self._dimensions[idx]
        self._dimensions[idx] = new
        if new == curr:
            return

        self.nodes[idx] = self._node_labels(idx, new)

        # Update constraints
        if new < curr:


    def _node_labels(self, idx, length) -> tuple:
        abbrev = self.abbrevs[layer_idx]
        return tuple(abbrev + str(n + Global.base) for n in range(0, length))


    def _layer_abbrev(self, layer) -> str:
        split_layer_to_parts = lambda layer: re.split("[ -\._]", layer)
        first_char_each_part = lambda layer_parts: [s[0] for s in layer_parts]

        abbrev_parts = first_char_each_part(split_layer_to_parts(layer))
        return "".join(abbrev_parts).upper()


    def _validate_layer_names(self, layers):
        abbrevs = self._layers_to_abbrevs(layers)
        assert len(set(abbrevs)) == len(abbrevs), \
                "\nLayer names must start with different letters.\n" \
                "You can work around this by using multiple words for layer names\n' \
                'For instance, 'manufacturing center' becomes 'MC')"
        assert [not s[0].isdigit() for s in abbrevs], \
                f"Layer names must not start with a number."


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
        return len(self.dv)














