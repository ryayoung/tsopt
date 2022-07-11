import pandas as pd
import numpy as np
import pyomo.environ as pe
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

from tsopt.data import SourceData


@dataclass
class Solution:
    obj_val: float
    slack: dict
    quantities: list



class Model(SourceData):
    """
    Pyomo wrapper for multi-stage transshipment optimization problems,
    where you have 3 or more location layers, and product transport between them.
    For example, you have 3 manufacturing plants, 2 distributors, and 5 warehouses,
    and you need to minimize cost from plant to distributor and from distributor to warehouse,
    while staying within capacity and meeting demand requirements.
    """
    solver = pe.SolverFactory('glpk')

    def __init__(self, cell_constraints=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mod = self.new_model()
        self.DV = [list(df.index) for df in self.costs] + [list(self.costs[-1].columns)]


    def new_model(self) -> pe.ConcreteModel:
        model = pe.ConcreteModel()
        self.solution = Solution(0.0, dict(), [])
        return model


    def add_decision_var(self, name:str, variable:pe.Var or any, domain=pe.NonNegativeReals, **kwargs) -> pe.Var:
        if type(variable) == pe.Var:
            setattr(self.mod, name, variable)
        else:
            setattr(self.mod, name, pe.Var(variable, domain=domain, **kwargs))
        return getattr(self.mod, name)


    def add_constraint(self, name, constr, prefix=None, **kwargs) -> pe.Constraint:
        name = name if prefix == None else f'{prefix}_{name}'
        if type(constr) == pe.Constraint:
            setattr(self.mod, name, constr, **kwargs)
        else:
            setattr(self.mod, name, pe.Constraint(expr=constr))

        return getattr(self.mod, name)


    def set_decision_vars(self):
        for stage in range(0, len(self.costs)):
            for node in self.DV[stage]:
                self.add_decision_var(node, self.DV[stage+1])


    def set_objective(self):
        total_cost = 0
        for stage, df in enumerate(self.costs):
            for node_in in self.DV[stage]:
                for node_out in self.DV[stage+1]:
                    total_cost += df.loc[node_in, node_out] * getattr(self.mod, node_in)[node_out]
        self.mod.obj = pe.Objective(expr = total_cost, sense=pe.minimize)


    def set_capacity_constraints(self):
        first_stg_inputs = self.DV[0]
        first_stg_outputs = self.DV[1]
        for node_in in first_stg_inputs:
            capacity = self.capacity.loc[node_in, self.capacity.columns[0]]
            expr = capacity >= sum(getattr(self.mod, node_in)[node_out] for node_out in first_stg_outputs)
            self.add_constraint(node_in, expr, 'capacity')


    def set_demand_constraints(self):
        final_stg_inputs = self.DV[-2]
        final_stg_outputs = self.DV[-1]
        for node_out in final_stg_outputs:
            demand = self.demand.loc[node_out, self.demand.columns[0]]
            expr = demand <= sum(getattr(self.mod, node_in)[node_out] for node_in in final_stg_inputs)
            self.add_constraint(node_out, expr, 'demand')


    def set_flow_constraints(self):
        '''
        Equal flow for all stages.
        A flow is a layer which takes input from a previous layer and
        outputs to the next layer. If there are 3 layers, then there must
        be two stages, and one flow: the middle layer.
        '''
        flow_layer_indexes = range(1, len(self.DV)-1)
        for flow in flow_layer_indexes:
            for node_curr in self.DV[flow]:
                input_nodes = self.DV[flow-1]
                output_nodes = self.DV[flow+1]
                inflow = sum([getattr(self.mod, node_in)[node_curr] for node_in in input_nodes])
                outflow = sum([getattr(self.mod, node_curr)[node_out] for node_out in output_nodes])
                expr = inflow == outflow
                self.add_constraint(node_curr, expr, 'flow')


    def get_quantities(self) -> list:
        quantities = []
        stage_indexes = range(0, len(self.costs))
        for stage in stage_indexes:
            input_nodes = self.DV[stage]
            output_nodes = self.DV[stage+1]
            values = [
                    [getattr(self.mod, node_in)[node_out].value for node_out in output_nodes]
                for node_in in input_nodes
            ]
            quantities.append(pd.DataFrame(values, columns=output_nodes, index=input_nodes))

        return quantities


    def get_slack(self) -> dict:
        slack = dict()
        for c in self.mod.component_objects(pe.Constraint):
            slack[str(c).split("_")[1]] = c.slack()
        return slack


    def build_model(self) -> pe.ConcreteModel:
        self.mod = self.new_model()
        self.set_decision_vars()
        self.set_objective()
        self.set_capacity_constraints()
        self.set_demand_constraints()
        self.set_flow_constraints()
        return self.mod


    def solve_model(self) -> Solution:
        Model.solver.solve(self.mod)

        # Save DV outputs to dataframes
        quantities = self.get_quantities()

        # Save objective value
        obj_val = self.mod.obj.expr()

        # Save slack to dictionary with constraint suffix
        slack = self.get_slack()

        self.solution = Solution(obj_val, slack, quantities)
        return self.solution


    def run(self) -> Solution:
        self.build_model()
        return self.solve_model()


    def display(self):
        print(f'Output capacity from {self.layers[0]}')
        display(self.capacity)
        print(f'Demand required from {self.layers[-1]}')
        display(self.demand)
        for i, df in enumerate(self.costs):
            print(f'{self.layers[i]} to {self.layers[i+1]} costs')
            display(df)


    def print_dv_indexes(self):
        for i, nodes in enumerate(self.DV):
            print(f"{self.layers[i]}:\n- ", end="")
            print(*nodes, sep=", ")


    def print_slack(self):
        has_slack = [c for c in self.solution.slack.keys() if self.solution.slack[c] != 0]
        print(f"The following {len(has_slack)} constraints have slack:")
        print(*[f"{c}: {self.solution.slack[c]}" for c in has_slack], sep="\n")


    def print_result(self):
        print("OBJECTIVE VALUE")
        print(f"Minimized Cost: ${self.obj_val}\n")
        print("DECISION VARIABLE QUANTITIES")
        for i, df in enumerate(self.outputs):
            print(f'STAGE {i}: {self.layers[i]} -> {self.layers[i+1]}')
            display(self.outputs[i].copy().astype(np.int64))


    # PLOTTING ---------------------------------------------------------------
    def label_edges(self,
            sum_inflow:bool,
            sum_outflow:bool,
            inflow_nodes:list,
            outflow_nodes:list,
            inflow_abbrev:str,
            ):
        '''
        Helper function for plot_stage to create a column that
        accurately labels each edge
        '''
        if sum_outflow and sum_inflow:
            route = inflow_abbrev
        elif sum_inflow:
            route = [inflow_abbrev] * len(outflow_nodes)
        elif sum_outflow:
            route = inflow_nodes
        else:
            route = inflow_nodes * len(outflow_nodes)

        route = pd.Series(route) + '->' + pd.Series(outflow_nodes)
        return route


    def stage_quantity_table(self, stage, sum_inflow=False, sum_outflow=False):
        '''
        Summarizes stage quantities
        '''
        df = self.outputs[stage-1]
        label_pref = (self.stage_abbrevs[stage-1], self.stage_abbrevs[stage])

        if sum_outflow:
            df = df.sum(axis=1).to_frame().rename(columns={0:label_pref[1]})
        if sum_inflow:
            df = df.sum().to_frame().T.rename(index={0:label_pref[0]})

        df = pd.melt(df).rename(columns={'variable':'outflow_nodes', 'value':'units'})
        df['Route'] = self.label_edges(
                sum_inflow, sum_outflow, self.DV[stage-1], df.outflow_nodes, label_pref[0])
        return df


    def smart_width(self, size:tuple, rows) -> tuple:
        '''
        Changes first element of size tuple based on num of rows.
        Only has effect when number of rows is less than 3
        '''
        w, h = size
        if rows < 3:
            w = w // 1.5
        if rows < 2:
            w = w // 2
        return (w, h)


    def plot_stage_quantity(self, stage=None,
            sum_inflow=False,
            sum_outflow=False,
            dynamic_width=True,
            figure=dict(),
            legend=dict(),
            **kwargs
            ):
        # By default, do all stages
        if stage == None:
            stage = [i for i in range(1, len(self.outputs)+1)]

        # Do multiple stages
        if type(stage) == list:
            if 'ax' in kwargs:
                raise TypeError("Can't use subplots when passing list of stages. (Can't take 'ax' argument when 'stage' argument is a list)")
            for s in stage:
                self.plot_stage_quantity(s, sum_inflow, sum_outflow, dynamic_width, figure, legend, **kwargs)
            return

        # Defaults based on stage
        if legend == dict():
            legend = dict(title=self.layers[stage], loc='upper right')
        if figure == dict():
            figure = dict(figsize=(12,5))

        df = self.stage_quantity_table(stage, sum_inflow, sum_outflow)
        if dynamic_width and 'figsize' in figure:
            figure['figsize'] = self.smart_width(figure['figsize'], df.shape[0])

        if 'ax' not in kwargs and figure != None:
            plt.figure(**figure)

        optional = dict()
        if 'hue' not in kwargs and 'color' not in kwargs:
            if sum_inflow == False and sum_outflow == False:
                optional['hue'] = 'outflow_nodes'
            else:
                optional['color'] = '#1f77b4'

        result = sns.barplot(x='Route', y='units', data=df, dodge=False, **kwargs, **optional) \
            .set(xlabel=None, title=f'Quantity: {self.layers[stage-1]} -> {self.layers[stage]}')

        if sum_inflow == False and sum_outflow == False and legend != None:
            if 'ax' in kwargs:
                kwargs['ax'].legend(**legend)
            else:
                plt.legend(**legend)

        return result





