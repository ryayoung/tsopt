# Maintainer:     Ryan Young
# Last Modified:  Sep 28, 2022
import pandas as pd
import numpy as np
import pyomo.environ as pe
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from copy import deepcopy

from tsopt.constants import *
from tsopt.types import *
from tsopt.edges import *
from tsopt.nodes import *
from tsopt.container import *
from tsopt.solved import *

class Model:
    """
    Pyomo wrapper for multi-stage transshipment optimization problems,
    where you have 2 or more location layers, and product transport between them.
    For example, you have 3 manufacturing plants, 2 distributors, and 5 warehouses,
    and you need to minimize cost from plant to distributor and from distributor to warehouse,
    while staying within capacity and meeting demand requirements.
    """
    solver = pe.SolverFactory('glpk')

    def __init__(self,
            layers: list,
            edge_costs: list,
            excel_file=None,
        ):

        self.units = 'units'
        self.excel_file = pd.ExcelFile(excel_file) if isinstance(excel_file, str) else excel_file

        self.dv = ModelConstants(self, layers, edge_costs)
        self.pe_mod = pe.ConcreteModel()

        # CONSTRAINTS
        self.network = NetworkValuesContainer(None, None)
        self.node = NodesContainer(self)
        self.edge = EdgesContainer(self)


    @property
    def constraint_objects(self):
        return {str(c): c for c in self.pe_mod.component_objects(pe.Constraint)}


    def var(self, var_name):
        return getattr(self.pe_mod, var_name)


    def add_constraint(self, name, constr):
        '''
        Dynamically set pe.Constraint variables on model, given a name
        '''
        setattr(self.pe_mod, name, pe.Constraint(expr=constr))


    def sum_outflows(self, idx, node):
        '''
        for node B3, return sum(B3[C1], B3[C2], B3[C3])
        '''
        output_nodes = self.dv.nodes[idx + 1]
        return sum(self.var(node)[out] for out in output_nodes)


    def sum_inflows(self, idx, node):
        '''
        for node B3, return sum(A1[B3], A2[B3], A3[B3])
        '''
        input_nodes = self.dv.nodes[idx - 1]
        return sum(self.var(inp)[node] for inp in input_nodes)


    def set_network_constraints(self):
        # Measure the flow in stage 0 since it must be the same in all stages
        if self.network.empty:
            return

        total_flow = sum( self.sum_outflows(0, node) for node in self.dv.nodes[0] )

        nw = self.network
        if nw.capacity: self.add_constraint('capacity', nw.capacity >= total_flow)
        if nw.demand: self.add_constraint('demand', nw.demand <= total_flow)


    def set_node_constraints(self):
        not_null = lambda srs: [sr[sr.notnull()] for sr in srs]
        sum_func = lambda i, node: self.sum_outflows(i,node) if i == 0 else self.sum_inflows(i,node)
        add = lambda name, expr: self.add_constraint(f'{node}_{type}', expr)

        for i, coefs in enumerate(not_null(self.node.capacity)):
            for node in coefs.index:
                add(f'capacity_{node}', coefs[node] >= sum_func(i, node))

        for i, coefs in enumerate(not_null(self.node.demand)):
            for node in coefs.index:
                add(f'demand_{node}', coefs[node] <= sum_func(i, node))


    def set_flow_constraints(self):
        '''
        Equal flow for all stages.
        '''
        for flow in self.dv.range_flow():
            for node in self.dv.nodes[flow]:
                expr = self.sum_inflows(flow, node) == self.sum_outflows(flow, node)
                self.add_constraint(f'flow_{node}', expr)


    def set_edge_constraints(self):
        not_null = lambda dfs: [df[~df.val.isna()] for df in dfs]
        edge = lambda inp, out: self.var(inp)[out]

        for df in not_null(self.edge.demand.melted):
            for i, inp_node, out_node, val in df.itertuples():
                self.add_constraint(f'demand_{inp_node}_{out_node}',
                        val <= edge(inp_node, out_node))

        for df in not_null(self.edge.capacity.melted):
            for i, inp_node, out_node, val in df.itertuples():
                self.add_constraint(f'capacity_{inp_node}_{out_node}',
                        val >= edge(inp_node, out_node))


    def build(self) -> pe.ConcreteModel:
        # DECISION VARS
        for ins, outs in self.dv.stage_nodes:
            for node in ins:
                setattr(self.pe_mod, node, pe.Var(outs, domain=pe.NonNegativeReals))

        # OBJECTIVE
        total_cost = 0
        for stg, df in enumerate(self.edge.bounds):
            for inp, out in zip(df.inp, df.out):
                total_cost += self.dv.cost[stg].loc[inp, out] * self.var(inp)[out]

        self.pe_mod.obj = pe.Objective(expr = total_cost, sense=pe.minimize)

        self.set_network_constraints()
        self.set_node_constraints()
        self.set_flow_constraints()
        self.set_edge_constraints()


    def solve(self) -> SolvedModel:
        self.build()
        success = Model.solver.solve(self.pe_mod)
        termination_condition = success.solver.termination_condition
        if termination_condition == 'optimal':
            return SolvedModel(self, success)
        else:
            print(f'Status: ', status)
            print(f'Term Condition: ', termination_condition)
            return success


    def display(self):
        zipped = zip(self.dv.layers, [len(nodes) for nodes in self.dv.nodes])
        print(*[f'{length}x {plural(layer)}' for layer, length in zipped], sep=' -> ')
        print()
        print("---- COSTS ----")
        for i, df in enumerate(self.dv.cost):
            print(f'{self.dv.layers[i]} -> {self.dv.layers[i+1]}')
            display(df)

        print("---- NODE CONSTRAINTS ----")
        display(self.node)

        print("---- EDGE CONSTRAINTS ----")
        dfs = self.edge.bounds
        for i, df in enumerate(dfs):
            df.index = df.inp + " -> " + df.out
            dfs[i] = dfs[i].drop(columns=['inp', 'out'])
            dfs[i] = dfs[i][(~dfs[i].dem.isna()) | (~dfs[i].cap.isna())]

        dfs = [df for df in dfs if df.shape.rows > 0]

        display(*dfs)


    def __len__(self):
        return len(self.dv)














