# Maintainer:     Ryan Young
# Last Modified:  Sep 24, 2022
import pandas as pd
import numpy as np
import pyomo.environ as pe
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from copy import deepcopy

from tsopt.basic_types import *
from tsopt.constants import *
from tsopt.data import *
from tsopt.stage_based import *
from tsopt.layer_based import *
from tsopt.exceptions import InfeasibleConstraint, InfeasibleEdgeConstraint
from tsopt.solution import *
import tsopt.text_util as txt
from tsopt.vector_util import *
from tsopt.layer import *
from tsopt.solved_model import *


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
        self.network = NetworkValues(None, None)
        self.node = NodeConstraintsContainer(self)
        self.edge = EdgeConstraintsContainer(self)
        self.pe_mod = pe.ConcreteModel()


    @property
    def constraint_objects(self):
        return {str(c): c for c in self.pe_mod.component_objects(pe.Constraint)}


    def add_decision_var(self, name:str, variable:pe.Var or any, domain=pe.NonNegativeReals, **kwargs) -> pe.Var:
        '''
        Dynamically set pe.Var decision variables on model, given a name
        '''
        if type(variable) == pe.Var:
            setattr(self.pe_mod, name, variable)
        else:
            setattr(self.pe_mod, name, pe.Var(variable, domain=domain, **kwargs))
        return getattr(self.pe_mod, name)


    def add_constraint(self, name, constr, prefix=None, **kwargs) -> pe.Constraint:
        '''
        Dynamically set pe.Constraint variables on model, given a name
        '''
        name = f'{prefix}_{name}' if prefix else name
        setattr(self.pe_mod, name, pe.Constraint(expr=constr))
        return getattr(self.pe_mod, name)


    # def load_edge_constraint(self, filename, sign, stage=0, *args, **kwargs):
        # df = raw_df_from_file(filename, self.dv.excel_file)
        # df = df.replace(-1, np.nan)
        # return self.add_edge_constraint(df, sign, stage, *args, **kwargs)


    # def loc_to_df(self, loc, stg, value) -> pd.DataFrame:
        # assert value != None, 'Must provide a constraint value when adding edge constraints by index'
        # assert len(loc) in [1,2], 'edge constraint loc argument needs one or two elements'
        # if type(loc) == tuple:
            # loc = list(loc)
        # elif type(loc) == int:
            # loc = [loc, loc]
# 
        # for i, item in enumerate(loc):
            # if item == None:
                # loc[i] = [None]
            # if type(item) != list:
                # if type(item) == tuple:
                    # loc[i] = list(item)
                # elif type(item) == str or type(item) == int:
                    # loc[i] = [item]
                # elif item != None:
                    # raise ValueError(f"Invalid data type for 'loc' parameter in add_edge_constraint")
# 
        # for i in range(0, len(loc)):
            # if loc[i] == [None]:
                # loc[i] = self.dv.nodes[stg+i]
            # else:
                # for node_idx, node in enumerate(loc[i]):
                    # if type(node) == int:
                        # assert 0 <= node <= len(self.dv.nodes[i+stg])-1, f'Node {node} not found in layer {i+stg}'
                        # loc[i][node_idx] = self.dv.nodes[i+stg][node]
                    # if type(node) == str:
                        # assert node in self.dv.nodes[i+stg], f'Node {node} not found in layer {self.dv.layers[i+stg]}'
# 
        # df = self.dv.template_stages()[stg]
        # for inp in loc[0]:
            # for out in loc[1]:
                # df.loc[inp, out] = value
# 
        # return df


    # def add_edge_constraint(self, loc:tuple or list or int or pd.DataFrame, sign:str, stage=0, value=None, force=False) -> pd.DataFrame:
        # '''
        # Constrain a single edge between two nodes.
        # '''
        # assert 0 <= stage <= len(self.dv.cost)-1, f'Invalid stage, {stage}, for edge constraint {loc}'
# 
        # if not sign in ['<=', '>=', 'max', 'min']:
            # raise ValueError("Invalid value for 'sign' parameter. Pyomo constraints require '<=' or '>='")
# 
        # if sign == '<=':
            # sign = 'max'
        # elif sign == '>=':
            # sign = 'min'
# 
        # if not isinstance(loc, pd.DataFrame):
            # df = self.loc_to_df(loc, stage, value)
        # else:
            # df = loc
# 
        # correct_dimensions = self.dv.cost[stage].shape
        # assert df.shape == correct_dimensions, \
            # f'Edge constraint dataframe dimensions {df.shape} do not match the required dimensions '\
            # f'{correct_dimensions} for stage {stage}.'
# 
        # df.index, df.columns = self.dv.stage_nodes[stage]
# 
# 
        # if sign == 'max':
            # self.edge_capacity[stage].update(df)
            # return self.edge_capacity[stage]
        # else:
            # self.edge_demand[stage].update(df)
            # return self.edge_demand[stage]
# 
# 
    # def set_edge_constraints(self) -> list:
        # constraints = []
        # for stg, constrs in enumerate(self.edge.bounds):
            # for name, df in constrs.items():
# 
                # Remove entirely null rows or entirely null columns to avoid useless iteration
                # df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
# 
                # for inp in df.index:
                    # for out in df.columns:
                        # val = df.loc[inp, out]
                        # if pd.isna(val): # Skip nulls
                            # continue
# 
                        # constr_name = f'edge_{name}_{inp}_{out}' # name of model instance var
# 
                        # if hasattr(self.pe_mod, constr_name): # replace existing constraint on that edge
                            # delattr(self.pe_mod, constr_name)
                        # edge = getattr(self.pe_mod, inp)[out]
# 
                        # if name == 'max':
                            # expr = edge <= val
                        # elif name == 'min':
                            # expr = edge >= val
# 
                        # constraints.append(self.add_constraint(constr_name, expr))
# 
        # return constraints


    def set_decision_vars(self):
        '''
        Each edge between each pair of nodes is a decision variable.
        Model variables are named after the input node
        '''
        for stage in self.dv.range_stage():
            for node in self.dv.nodes[stage]:
                self.add_decision_var(node, self.dv.nodes[stage+1])


    def set_objective(self):
        '''
        The sum of all edges total cost (sumproduct of unit cost and quantity)
        '''
        total_cost = 0
        for stage, df in enumerate(self.dv.cost):
            for node_in in self.dv.nodes[stage]:
                for node_out in self.dv.nodes[stage+1]:
                    total_cost += df.loc[node_in, node_out] * getattr(self.pe_mod, node_in)[node_out]
        self.pe_mod.obj = pe.Objective(expr = total_cost, sense=pe.minimize)


    def set_network_constraints(self):
        # Measure the flow in stage 0
        if self.network.empty:
            return
        inputs, outputs = self.dv.stage_nodes[0]
        total_flow = sum([
                sum(getattr(self.pe_mod, node_in)[node_out] for node_out in outputs)
            for node_in in inputs
        ])
        if self.network.capacity:
            expr = self.network.capacity >= total_flow
            self.add_constraint('network_capacity', expr)
        if self.network.demand:
            expr = self.network.demand <= total_flow
            self.add_constraint('network_demand', expr)


    def pe_node_flow_sum(self, direction:str, target_node:str, layer_idx:int):
        """
        Say we have layers, [A, B, C], with 3 nodes each.
        Target node is B3. We want to sum the units flowing through it. Since layer B
        is surrounded by other layers on either side, it doesn't matter if we use the
        input or output nodes for this calculation. If it were A, we would need to use
        output nodes, and for C, we would need to use input nodes.
        --
        If direction == "in":
            return sum(
                A1[B3],
                A2[B3],
                A3[B3]
            )
        If direction == "out":
            return sum(
                B3[C1],
                B3[C2],
                B3[C3]
            )
        """
        if direction == "in":
            input_nodes = self.dv.nodes[layer_idx - 1]
            return sum(getattr(self.pe_mod, node_in)[target_node] for node_in in input_nodes)

        if direction == "out":
            output_nodes = self.dv.nodes[layer_idx + 1]
            return sum(getattr(self.pe_mod, target_node)[node_out] for node_out in output_nodes)


    def set_node_constraints(self):
        '''
        CAPACITY
        --
        Make sure each decision variable is <= its capacity.
        Capacity constraints are valid on any layer except the last.
        '''
        for i, coefs in enumerate(self.node.capacity):
            coefs = coefs[coefs.notnull()]
            direction = "in" if i == len(self)-1 else "out"
            for curr in tuple(coefs.index):
                expr = coefs[curr] >= self.pe_node_flow_sum(direction, curr, i)
                self.add_constraint(curr, expr, 'capacity')
        '''
        DEMAND
        --
        If layer > 0: calculated as minimum input from all previous nodes
        If layer == 0: calculated as minimum output to all following nodes
        '''
        for i, coefs in enumerate(self.node.demand):
            coefs = coefs[coefs.notnull()]
            direction = "out" if i == 0 else "in"
            for curr in tuple(coefs.index):
                expr = coefs[curr] <= self.pe_node_flow_sum(direction, curr, i)
                self.add_constraint(curr, expr, 'demand')



    def set_flow_constraints(self):
        '''
        Equal flow for all stages.
        A flow is a layer which takes input from a previous layer and
        outputs to the next layer. If there are 3 layers, then there must
        be one flow: the middle layer.
        '''
        for flow in self.dv.range(start=1, end=-1):
            for curr in self.dv.nodes[flow]:
                input_nodes = self.dv.nodes[flow-1]
                output_nodes = self.dv.nodes[flow+1]
                inflow = sum([getattr(self.pe_mod, inp)[curr] for inp in input_nodes])
                outflow = sum([getattr(self.pe_mod, curr)[out] for out in output_nodes])
                expr = inflow == outflow
                name = f'{curr}_{self.dv.abbrevs[flow+1]}'
                self.add_constraint(name, expr, 'flow')


    def set_edge_constraints(self):
        pass


    def build(self) -> pe.ConcreteModel:

        self.set_decision_vars()
        self.set_objective()
        self.set_network_constraints()
        self.set_node_constraints()
        self.set_flow_constraints()
        self.set_edge_constraints()

        return self


    def solve(self) -> Solution:
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
        print(*[f'{length}x {txt.plural(layer)}' for layer, length in zipped], sep=' -> ')
        print()
        print("-------------- COSTS ---------------")
        for i, df in enumerate(self.dv.cost):
            print(f'{self.dv.layers[i]} -> {self.dv.layers[i+1]}')
            display(df)

        print("----------- CONSTRAINTS ------------")
        for i, df in self.constraint_df_bounds.items():
            print(f'{self.dv.layers[i]}:')
            df.columns = ['Demand', 'Capacity']
            display(df)

        print("--------- EDGE CONSTRAINTS ---------")
        for stg, constrs in enumerate(self.edge.bounds):
            for name, df in constrs.items():
                if df.isnull().values.all():
                    continue
                print(f'\n{name.upper()}: {self.dv.layers[stg]} -> {self.dv.layers[stg+1]}')
                display(df.fillna(-1).astype(int).replace(-1, '-'))


    def print_dv_indexes(self):
        for i, nodes in enumerate(self.dv.nodes):
            print(f"{self.dv.layers[i]}:\n- ", end="")
            print(*nodes, sep=", ")




    def __len__(self):
        return len(self.dv)














