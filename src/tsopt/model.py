# Maintainer:     Ryan Young
# Last Modified:  Aug 31, 2022
import pandas as pd
import numpy as np
import pyomo.environ as pe
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from copy import deepcopy

from tsopt.constants import *
from tsopt.data import *
from tsopt.stage_based import *
from tsopt.exceptions import InfeasibleConstraint, InfeasibleEdgeConstraint
from tsopt.solution import Solution
import tsopt.text_util as txt
from tsopt.vector_util import *



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

        self.pe_mod = pe.ConcreteModel()

        self.reset_edge_constraints()


    def reset_edge_constraints(self):
        self.edge = EdgeConstraintsContainer(self)


    @property
    def mod_constraints(self):
        return {str(c): c for c in self.pe_mod.component_objects(pe.Constraint)}


    @property
    def edge_constraints(self):
        return {
            stage: {
                'min': self.edge.demand[stage],
                'max': self.edge.capacity[stage],
            }
            for stage in self.dv.range_stage()
        }


    @staticmethod
    def combine_if(sr1, sr2, func) -> pd.Series:
        def pick(a,b):
            if not np.isnan(a) and not np.isnan(b):
                return func(a,b)
            if not np.isnan(a):
                return a
            return b
        return sr1.combine(sr2, pick)


    def edge_lowered_capacity_by_stage(self, new=True) -> list:
        ''' Show nodes whose edges are fully constrained by edge upper bounds
        to a value lower than their original capacity '''

        def capacity_lowered(stage):
            sums1, sums2 = self.edge_capacity[stage].sums(full=True, concat=False)
            cap1, cap2 = self.capacity.stage(stage)
            if new:
                sums1[sums1 >= cap1] = np.nan
                sums2[sums2 >= cap2] = np.nan
            else:
                sums1 = self.combine_if(sums1, cap1, min)
                sums2 = self.combine_if(sums2, cap2, min)
            return [sums1, sums2]

        return [capacity_lowered(i) for i in self.dv.range_stage()]


    def edge_raised_demand_by_stage(self, new=True) -> list:
        ''' Show nodes which have edges constrained to a lower bound totaling
        to greater than the node's demand '''

        def demand_raised(stage):
            sums1, sums2 = self.edge_demand[stage].sums(concat=False)
            dem1, dem2 = self.demand.stage(stage)
            if new:
                sums1[sums1 <= dem1] = np.nan
                sums2[sums2 <= dem2] = np.nan
            else:
                sums1 = self.combine_if(sums1, dem1, max)
                sums2 = self.combine_if(sums2, dem2, max)
            return [sums1, sums2]

        return [demand_raised(i) for i in self.dv.range_stage()]


    def edge_lowered_capacity_by_layer(self, new=True) -> LayerDict:
        stages = self.edge_lowered_capacity_by_stage(new)

        inputs = stages[0][0]
        outputs = stages[-1][1]
        flows = [ self.combine_if(s1[1], s2[0], min) for s1,s2 in staged(stages) ]

        all_layers = [inputs] + flows + [outputs]
        ldict = {i: sr for i,sr in enumerate(all_layers)}

        return LayerDict(self.dv, ldict)


    def edge_raised_demand_by_layer(self, new=True) -> LayerDict:
        stages = self.edge_raised_demand_by_stage(new)

        inputs = stages[0][0]
        outputs = stages[-1][1]
        flows = [ self.combine_if(s1[1], s2[0], max) for s1,s2 in staged(stages) ]

        all_layers = [inputs] + flows + [outputs]
        ldict = {i: sr for i,sr in enumerate(all_layers)}

        return LayerDict(self.dv, ldict)


    def blank_stage_df(self, val, stage) -> pd.DataFrame:
        return pd.DataFrame(val,
                index=self.dv.cost[stage].index,
                columns=self.dv.cost[stage].columns,
        ).astype('float')


    def blank_constraint_df(self, val, layer, name) -> pd.Series:
        return pd.Series(val,
                index=self.dv.nodes[layer],
                name=name,
        ).astype('float')


    def new_model(self) -> pe.ConcreteModel:
        self.edge_capacity = StageDFs(self.dv.templates.stages())
        self.edge_demand = StageDFs(self.dv.templates.stages())
        return pe.ConcreteModel()


    def new_edge_constraints(self) -> dict:
        return {
            stage: {
                'min': self.blank_stage_df(np.nan, stage),
                'max': self.blank_stage_df(np.nan, stage),
            }
            for stage in self.dv.range_stage()
        }


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
        name = name if prefix == None else f'{prefix}_{name}'
        if type(constr) == pe.Constraint:
            setattr(self.pe_mod, name, constr, **kwargs)
        else:
            setattr(self.pe_mod, name, pe.Constraint(expr=constr))
        return getattr(self.pe_mod, name)


    def load_edge_constraint(self, filename, sign, stage=0, *args, **kwargs):
        df = raw_df_from_file(filename, self.dv.excel_file)
        df = df.replace(-1, np.nan)
        return self.add_edge_constraint(df, sign, stage, *args, **kwargs)


    def loc_to_df(self, loc, stg, value) -> pd.DataFrame:
        assert value != None, 'Must provide a constraint value when adding edge constraints by index'
        assert len(loc) in [1,2], 'edge constraint loc argument needs one or two elements'
        if type(loc) == tuple:
            loc = list(loc)
        elif type(loc) == int:
            loc = [loc, loc]

        for i, item in enumerate(loc):
            if item == None:
                loc[i] = [None]
            if type(item) != list:
                if type(item) == tuple:
                    loc[i] = list(item)
                elif type(item) == str or type(item) == int:
                    loc[i] = [item]
                elif item != None:
                    raise ValueError(f"Invalid data type for 'loc' parameter in add_edge_constraint")

        for i in range(0, len(loc)):
            if loc[i] == [None]:
                loc[i] = self.dv.nodes[stg+i]
            else:
                for node_idx, node in enumerate(loc[i]):
                    if type(node) == int:
                        assert 0 <= node <= len(self.dv.nodes[i+stg])-1, f'Node {node} not found in layer {i+stg}'
                        loc[i][node_idx] = self.dv.nodes[i+stg][node]
                    if type(node) == str:
                        assert node in self.dv.nodes[i+stg], f'Node {node} not found in layer {self.dv.layers[i+stg]}'

        df = self.dv.templates.stages()[stg]
        for inp in loc[0]:
            for out in loc[1]:
                df.loc[inp, out] = value

        return df


    def add_edge_constraint(self, loc:tuple or list or int or pd.DataFrame, sign:str, stage=0, value=None, force=False) -> pd.DataFrame:
        '''
        Constrain a single edge between two nodes.
        '''
        assert 0 <= stage <= len(self.dv.cost)-1, f'Invalid stage, {stage}, for edge constraint {loc}'

        if not sign in ['<=', '>=', 'max', 'min']:
            raise ValueError("Invalid value for 'sign' parameter. Pyomo constraints require '<=' or '>='")

        if sign == '<=':
            sign = 'max'
        elif sign == '>=':
            sign = 'min'

        if not isinstance(loc, pd.DataFrame):
            df = self.loc_to_df(loc, stage, value)
        else:
            df = loc

        correct_dimensions = self.dv.cost[stage].shape
        assert df.shape == correct_dimensions, \
            f'Edge constraint dataframe dimensions {df.shape} do not match the required dimensions '\
            f'{correct_dimensions} for stage {stage}.'

        df.index, df.columns = self.dv.stage_nodes[stage]


        if sign == 'max':
            self.edge_capacity[stage].update(df)
            return self.edge_capacity[stage]
        else:
            self.edge_demand[stage].update(df)
            return self.edge_demand[stage]


    def set_edge_constraints(self) -> list:
        constraints = []
        for stg, constrs in self.edge_constraints.items():
            for name, df in constrs.items():

                # Remove entirely null rows or entirely null columns to avoid useless iteration
                df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')

                for inp in df.index:
                    for out in df.columns:
                        val = df.loc[inp, out]
                        if pd.isna(val): # Skip nulls
                            continue

                        constr_name = f'edge_{name}_{inp}_{out}' # name of model instance var

                        if hasattr(self.pe_mod, constr_name): # replace existing constraint on that edge
                            delattr(self.pe_mod, constr_name)
                        edge = getattr(self.pe_mod, inp)[out]

                        if name == 'max':
                            expr = edge <= val
                        elif name == 'min':
                            expr = edge >= val

                        constraints.append(self.add_constraint(constr_name, expr))

        return constraints


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


    def set_capacity_constraints(self):
        '''
        i.e. Maximum node output
        --
        Make sure each decision variable is <= its capacity.
        Capacity constraints are valid on any layer except the last.
        '''
        final_layr_idx = len(self)-1
        for i, coefs in self.capacity.items():
            coefs = coefs[coefs.notnull()]
            curr_nodes = tuple(coefs.index)
            if i < final_layr_idx:
                output_nodes = self.dv.nodes[i+1]
                for node_curr in curr_nodes:
                    val = coefs[node_curr]
                    expr = val >= sum(getattr(self.pe_mod, node_curr)[node_out] for node_out in output_nodes)
                    self.add_constraint(node_curr, expr, 'capacity')
            elif i == final_layr_idx:
                input_nodes = self.dv.nodes[i-1]
                for node_curr in curr_nodes:
                    val = coefs[node_curr]
                    expr = val >= sum(getattr(self.pe_mod, node_in)[node_curr] for node_in in input_nodes)
                    self.add_constraint(node_curr, expr, 'capacity')


    def set_demand_constraints(self):
        '''
        Lower bound for units passing through nodes
        --
        If layer > 0: calculated as minimum input from all previous nodes
        If layer == 0: calculated as minimum output to all following nodes
        '''
        for i, coefs in self.demand.items():
            coefs = coefs[coefs.notnull()]
            curr_nodes = tuple(coefs.index)
            if i > 0:
                input_nodes = self.dv.nodes[i-1]
                for node_curr in curr_nodes:
                    val = coefs[node_curr]
                    expr = val <= sum(getattr(self.pe_mod, node_in)[node_curr] for node_in in input_nodes)
                    self.add_constraint(node_curr, expr, 'demand')
            elif i == 0:
                output_nodes = self.dv.nodes[i+1]
                for node_curr in curr_nodes:
                    val = coefs[node_curr]
                    expr = val <= sum(getattr(self.pe_mod, node_curr)[node_out] for node_out in output_nodes)
                    self.add_constraint(node_curr, expr, 'demand')


    def set_flow_constraints(self):
        '''
        Equal flow for all stages.
        A flow is a layer which takes input from a previous layer and
        outputs to the next layer. If there are 3 layers, then there must
        be one flow: the middle layer.
        '''
        for flow in self.dv.range(start=1, end=-1):
            for node_curr in self.dv.nodes[flow]:
                input_nodes = self.dv.nodes[flow-1]
                output_nodes = self.dv.nodes[flow+1]
                inflow = sum([getattr(self.pe_mod, inp)[curr] for inp in input_nodes])
                outflow = sum([getattr(self.pe_mod, curr)[out] for out in output_nodes])
                expr = inflow == outflow
                name = f'{node_curr}_{self.dv.abbrevs[flow+1]}'
                self.add_constraint(name, expr, 'flow')


    def build(self) -> pe.ConcreteModel:

        self.set_decision_vars()
        self.set_objective()
        self.set_capacity_constraints()
        self.set_demand_constraints()
        self.set_flow_constraints()
        self.set_edge_constraints()

        return self


    def solve(self) -> Solution:
        self.build()
        success = Model.solver.solve(self.pe_mod)
        status = success.solver.status
        termination_condition = success.solver.termination_condition
        self.solution = Solution(self.dv, self.dv.cost, self.pe_mod, self.mod_constraints, success, status, termination_condition)

        return self.solution


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
        for stg, constrs in self.edge_constraints.items():
            for name, df in constrs.items():
                if df.isnull().values.all():
                    continue
                print(f'\n{name.upper()}: {self.dv.layers[stg]} -> {self.dv.layers[stg+1]}')
                display(df.fillna(-1).astype(int).replace(-1, '-'))


    def print_dv_indexes(self):
        for i, nodes in enumerate(self.dv.nodes):
            print(f"{self.dv.layers[i]}:\n- ", end="")
            print(*nodes, sep=", ")


















