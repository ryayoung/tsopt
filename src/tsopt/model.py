# Maintainer:     Ryan Young
# Last Modified:  Jul 16, 2022
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


    def add_edge_constraint(self, stage, loc:tuple or list, sign:str, value) -> list:
        '''
        Constrain a single edge between two nodes.
        '''
        assert 0 <= stage <= len(self.cost)-1, f'Invalid stage, {stage}, for edge constraint {loc}'
        assert len(loc) == 2, 'edge constraint loc argument needs two elements: input and output'
        if type(loc) == tuple:
            loc = list(loc)

        for i, item in enumerate(loc):
            if type(item) != list:
                loc[i] = [item]

        assert sign in ['<=', '>='], 'Must use <= or >= sign in custom edge constraints'

        inputs, outputs = self.validate_edge_constraint(stage, loc, sign, value)
        constraints = []

        for inp in inputs:
            for out in outputs:
                if sign == '<=':
                    expr = getattr(self.mod, inp)[out] <= value
                    name = f'edge_max_{value}_{inp}_{out}'
                elif sign == '>=':
                    expr = getattr(self.mod, inp)[out] >= value
                    name = f'edge_min_{value}_{inp}_{out}'
                if not hasattr(self.mod, name):
                    constraints.append(self.add_constraint(name, expr))

        return constraints


    def validate_edge_constraint(self, stage, loc, sign, value) -> (list, list):
        # STEP 1: make sure the keys are valid and convert them to strings if needed
        for i in range(0, len(loc)):
            if loc[i] == [None]:
                loc[i] = self.dv.nodes[stage+i]
            else:
                for node_idx, node in enumerate(loc[i]):
                    if type(node) == int:
                        assert 0 <= node <= len(self.dv.nodes[i+stage])-1, f'Node {node} not found in layer {i+stage}'
                        loc[i][node_idx] = self.dv.nodes[i+stage][node]
                    if type(node) == str:
                        assert node in self.dv.nodes[i+stage], f'Node {node} not found in layer {self.dv.layers[i+stage]}'

        inputs, outputs = loc[0], loc[1]

        # STEP 2: Make sure we aren't creating a mathematically impossible model
        df_all = self.cost[stage].copy()
        df_all[:] = np.NaN
        for inp in inputs:
            for out in outputs:
                df_all.loc[inp, out] = value

        df_in = df_all.sum(axis=1).to_frame()
        df_in = df_in[df_in.index.isin(df_all.dropna(axis=0).index)]
        df_out = df_all.sum().to_frame()
        df_out = df_out[df_out.index.isin(df_all.dropna(axis=1).columns)]
        dfs = [df_in, df_out]

        fully_constrained_by = None
        if len(dfs[0].index) == len(df_all.index) and len(dfs[1].index) == len(df_all.columns):
            fully_constrained_by = df_all.copy().values.sum()

        if sign == '<=':
            sign = -1
        else:
            sign = 1

        if fully_constrained_by != None:
            if sign == '<=':
                assert fully_constrained_by >= self.demand[len(self.cost)].values.sum(), \
                        'Edge constraint makes it impossible to hit the final demand requirement'
            elif sign == '>=':
                assert fully_constrained_by <= self.capacity[0].values.sum(), \
                        'Edge constraint requires a flow volume that exceeds the initial capacity available'

        for i, df in enumerate(dfs):
            if not df.empty:
                for node in df.index:
                    val = df.loc[node, df.columns[0]]
                    layr = i+stage
                    if sign == '>=':
                        if layr in self.capacity:
                            assert val <= self.capacity[layr].loc[node, self.capacity[layr].columns[0]], \
                                    f'Node {node} is constrained at a value greater than its capacity'
                        if layr in self.max_in:
                            assert val <= self.max_in[layr].loc[node, self.max_in[layr].columns[0]], \
                                    f'Node {node} is constrained at a value greater than its max input'
                        if layr in self.demand:
                            assert self.capacity[0].values.sum() >= val + self.demand[layr][self.demand[layr].index != node].values.sum(), \
                                    f'Node {node} is constrained at a value ({val}) high enough that the other {self.dv.layers[layr]} ' \
                                    f"nodes can't meet their demand requirements without exceeding initial flow capacity"
                    elif sign == '<=':
                        if layr in self.demand:
                            assert val >= self.demand[layr].loc[node, self.demand[layr].columns[0]], \
                                    f'Node {node} is constrained at a value less than its demand'
                        if layr in self.min_out:
                            assert val >= self.min_out[layr].loc[node, self.min_out[layr].columns[0]], \
                                    f'Node {node} is constrained at a value less than its minimum output'

        return inputs, outputs


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
        i.e. Maximum node output
        --
        Make sure each decision variable is <= its capacity.
        Capacity constraints are valid on any layer except the last.
        '''
        for i, coefs in self.capacity.items():
            input_nodes = self.dv.nodes[i]
            output_nodes = self.dv.nodes[i+1]
            for node_in in input_nodes:
                val = coefs.loc[node_in, coefs.columns[0]]
                expr = val >= sum(getattr(self.mod, node_in)[node_out] for node_out in output_nodes)
                self.add_constraint(node_in, expr, 'capacity')


    def set_min_out_constraints(self):
        '''
        i.e. Minimum node output
        --
        Make sure each decision variable is outputting >= its minimum.
        Min. out constraints are valid on any layer except the last.
        '''
        for i, coefs in self.min_out.items():
            input_nodes = self.dv.nodes[i]
            output_nodes = self.dv.nodes[i+1]
            for node_in in input_nodes:
                val = coefs.loc[node_in, coefs.columns[0]]
                expr = val <= sum(getattr(self.mod, node_in)[node_out] for node_out in output_nodes)
                self.add_constraint(node_in, expr, 'min_out')


    def set_demand_constraints(self):
        '''
        i.e. Minimum node input from prev layer
        --
        Make sure each decision variable is >= its demand.
        Demand constraints are valid on any layer except the first
        '''
        for i, coefs in self.demand.items():
            input_nodes = self.dv.nodes[i-1]
            output_nodes = self.dv.nodes[i]
            for node_out in output_nodes:
                val = coefs.loc[node_out, coefs.columns[0]]
                expr = val <= sum(getattr(self.mod, node_in)[node_out] for node_in in input_nodes)
                self.add_constraint(node_out, expr, 'demand')


    def set_max_in_constraints(self):
        '''
        i.e. Maximum node input from prev layer
        --
        Make sure each decision variable is <= its maximum input.
        Max-in constraints are valid on any layer except the first
        '''
        for i, coefs in self.max_in.items():
            input_nodes = self.dv.nodes[i-1]
            output_nodes = self.dv.nodes[i]
            for node_out in output_nodes:
                val = coefs.loc[node_out, coefs.columns[0]]
                expr = val >= sum(getattr(self.mod, node_in)[node_out] for node_in in input_nodes)
                self.add_constraint(node_out, expr, 'max_in')


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
        self.set_min_out_constraints()
        self.set_max_in_constraints()
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
        zipped = zip(self.dv.layers, [len(nodes) for nodes in self.dv.nodes])
        print(*[f'{length}x {self.plural(layer)}' for layer, length in zipped], sep=' -> ')
        print()
        print("-------------- COSTS ---------------")
        for i, df in enumerate(self.cost):
            print(f'{self.dv.layers[i]} -> {self.dv.layers[i+1]}')
            display(df)
        print("----------- CONSTRAINTS ------------")
        for i, df in self.capacity.items():
            display(df)
        for i, df in self.demand.items():
            display(df)
        for i, df in self.min_out.items():
            display(df)
        for i, df in self.max_in.items():
            display(df)


    def print_dv_indexes(self):
        for i, nodes in enumerate(self.dv.nodes):
            print(f"{self.dv.layers[i]}:\n- ", end="")
            print(*nodes, sep=", ")


    def plural(self, word:str) -> str:
        if word.endswith('y'):
            return word[:-1] + 'ies'
        if word.endswith('s'):
            return word + 'es'
        return word + 's'
