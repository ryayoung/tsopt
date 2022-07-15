# Maintainer:     Ryan Young
# Last Modified:  Jul 11, 2022
import pandas as pd
import numpy as np
import pyomo.environ as pe
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

from tsopt.data import SourceData
from tsopt.solution import Solution



class Model(SourceData):
    """
    Pyomo wrapper for multi-stage transshipment optimization problems,
    where you have 2 or more location layers, and product transport between them.
    For example, you have 3 manufacturing plants, 2 distributors, and 5 warehouses,
    and you need to minimize cost from plant to distributor and from distributor to warehouse,
    while staying within capacity and meeting demand requirements.
    """
    solver = pe.SolverFactory('glpk')

    def __init__(self, cell_constraints=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mod = self.new_model()


    @property
    def constraints(self):
        return {str(c): c for c in self.mod.component_objects(pe.Constraint)}


    def new_model(self) -> pe.ConcreteModel:
        return pe.ConcreteModel()


    def add_decision_var(self, name:str, variable:pe.Var or any, domain=pe.NonNegativeReals, **kwargs) -> pe.Var:
        '''
        Dynamically set pe.Var decision variables on model, given a name
        '''
        if type(variable) == pe.Var:
            setattr(self.mod, name, variable)
        else:
            setattr(self.mod, name, pe.Var(variable, domain=domain, **kwargs))
        return getattr(self.mod, name)


    def add_constraint(self, name, constr, prefix=None, **kwargs) -> pe.Constraint:
        '''
        Dynamically set pe.Constraint variables on model, given a name
        '''
        name = name if prefix == None else f'{prefix}_{name}'
        if type(constr) == pe.Constraint:
            setattr(self.mod, name, constr, **kwargs)
        else:
            setattr(self.mod, name, pe.Constraint(expr=constr))
        return getattr(self.mod, name)


    def set_decision_vars(self):
        '''
        Each edge between each pair of nodes is a decision variable.
        Model variables are named after the input node
        '''
        for stage in range(0, len(self.cost)):
            for node in self.dv.nodes[stage]:
                self.add_decision_var(node, self.dv.nodes[stage+1])


    def set_objective(self):
        '''
        The sum of all edges total cost (sumproduct of unit cost and quantity)
        '''
        total_cost = 0
        for stage, df in enumerate(self.cost):
            for node_in in self.dv.nodes[stage]:
                for node_out in self.dv.nodes[stage+1]:
                    total_cost += df.loc[node_in, node_out] * getattr(self.mod, node_in)[node_out]
        self.mod.obj = pe.Objective(expr = total_cost, sense=pe.minimize)


    def set_capacity_constraints(self):
        '''
        Make sure each decision variable is <= its capacity.
        Capacity constraints are valid on any layer except the last.
        '''
        for i, coefs in self.capacity.items():
            input_nodes = self.dv.nodes[i]
            output_nodes = self.dv.nodes[i+1]
            for node_in in input_nodes:
                capacity = coefs.loc[node_in, coefs.columns[0]]
                expr = capacity >= sum(getattr(self.mod, node_in)[node_out] for node_out in output_nodes)
                self.add_constraint(node_in, expr, 'capacity')


    def set_demand_constraints(self):
        '''
        Make sure each decision variable is >= its demand.
        Demand constraints are valid on any layer except the first
        '''
        for i, coefs in self.demand.items():
            input_nodes = self.dv.nodes[i-1]
            output_nodes = self.dv.nodes[i]
            for node_out in output_nodes:
                demand = coefs.loc[node_out, coefs.columns[0]]
                expr = demand <= sum(getattr(self.mod, node_in)[node_out] for node_in in input_nodes)
                self.add_constraint(node_out, expr, 'demand')


    def set_flow_constraints(self):
        '''
        Equal flow for all stages.
        A flow is a layer which takes input from a previous layer and
        outputs to the next layer. If there are 3 layers, then there must
        be one flow: the middle layer.
        '''
        flow_layer_indexes = range(1, len(self.dv.nodes)-1)
        for flow in flow_layer_indexes:
            for node_curr in self.dv.nodes[flow]:
                input_nodes = self.dv.nodes[flow-1]
                output_nodes = self.dv.nodes[flow+1]
                inflow = sum([getattr(self.mod, node_in)[node_curr] for node_in in input_nodes])
                outflow = sum([getattr(self.mod, node_curr)[node_out] for node_out in output_nodes])
                expr = inflow == outflow
                name = f'{node_curr}_{self.dv.abbrevs[flow+1]}'
                self.add_constraint(name, expr, 'flow')


    def build_model(self) -> pe.ConcreteModel:
        self.mod = self.new_model()
        self.set_decision_vars()
        self.set_objective()
        self.set_capacity_constraints()
        self.set_demand_constraints()
        self.set_flow_constraints()
        return self.mod


    def solve_model(self) -> Solution:
        success = Model.solver.solve(self.mod)
        status = success.solver.status
        termination_condition = success.solver.termination_condition
        if status != 'ok':
            print('Error: Status:', status)
        if termination_condition != 'optimal':
            print('Error: Termination condition:', termination_condition)

        self.solution = Solution(self.dv, self.cost, self.mod, self.constraints, success, status, termination_condition)
        return self.solution


    def run(self) -> Solution:
        self.build_model()
        return self.solve_model()


    def display(self):
        for i, df in self.capacity.items():
            print(f'Output capacity from {self.dv.layers[i]}')
            display(df)
        for i, df in self.demand.items():
            print(f'Demand required from {self.dv.layers[i]}')
            display(df)
        for i, df in enumerate(self.cost):
            print(f'{self.dv.layers[i]} to {self.dv.layers[i+1]} costs')
            display(df)


    def print_dv_indexes(self):
        for i, nodes in enumerate(self.dv.nodes):
            print(f"{self.dv.layers[i]}:\n- ", end="")
            print(*nodes, sep=", ")

