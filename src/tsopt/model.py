import pandas as pd
import numpy as np
import pyomo.environ as pe
import matplotlib.pyplot as plt
import seaborn as sns

from tsopt.data import SourceData


class Model(SourceData):
    """
    Pyomo wrapper for 3-stage transshipment optimization problems,
    where you have 3 location layers, and product transport between them.
    For example, you have 3 manufacturing plants, 2 distributors, and 5 warehouses,
    and you need to minimize cost from plant to distributor and from distributor to warehouse,
    while staying within capacity and meeting demand requirements.
    """
    solver = pe.SolverFactory('glpk')

    def __init__(self, cell_constraints=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Optional added constraint: list of dicts
        # [{sign: <str>, cell1: {stage: <int>, row: <int>, col: <int>}, cell2: {stage: <int>, row: <int>, col: <int>}}]
        # self.cell_constraints = cell_constraints

        # Set DV indexes.
        self.DV = [list(x) for x in [self.c1.index, self.c1.columns, self.c2.columns]]

        self.mod = pe.ConcreteModel()

        # Declare the model's pe.Var() decision variables.
        for x in self.DV[0]:
            setattr(self.mod, x, pe.Var(self.DV[1], domain = pe.NonNegativeReals))
        for d in self.DV[1]:
            setattr(self.mod, d, pe.Var(self.DV[2], domain = pe.NonNegativeReals))

        # Final decision variables stored when model is run
        self.obj_val = None
        self.slack = {} # Dict where each key is constraint name, val is slack

    def run(self):
        # TO SUPPRESS WARNING: Each time function is called, delete component objects
        for attr in list(self.mod.component_objects([pe.Constraint, pe.Objective])):
            self.mod.del_component(getattr(self.mod, str(attr)))

        # Objective function
        products = []
        for x in self.DV[0]:
            products += [self.c1.loc[x, i] * getattr(self.mod, x)[i] for i in self.DV[1]]
        for d in self.DV[1]:
            products += [self.c2.loc[d, i] * getattr(self.mod, d)[i] for i in self.DV[2]]

        self.mod.obj = pe.Objective(expr = sum(products), sense = pe.minimize)

        # Capacity constraint
        for x in self.DV[0]:
            setattr(self.mod, f"con_{x}", pe.Constraint(
                expr = sum(getattr(self.mod, x)[d] for d in self.DV[1])
                    <= self.capacity.loc[x, self.capacity.columns[0]]))

        # Demand constraint
        for w in self.DV[2]:
            setattr(self.mod, f"con_{w}", pe.Constraint(expr =
                sum(getattr(self.mod, d)[w] for d in self.DV[1])
                    >= self.demand.loc[w, self.demand.columns[0]]))

        # Equal flow for both stages
        for d in self.DV[1]:
            setattr(self.mod, f"con_{d}", pe.Constraint(expr =
                sum([getattr(self.mod, d)[w] for w in self.DV[2]])
                    == sum([getattr(self.mod, x)[d] for x in self.DV[0]])))

        Model.solver.solve(self.mod)

        # Save DV outputs to dataframes
        self.outputs = [
            pd.DataFrame([
                [getattr(self.mod, x)[d].value for d in self.DV[1]] for x in self.DV[0]],
                columns=self.DV[1], index=self.DV[0]
            ),
            pd.DataFrame([
                [getattr(self.mod, d)[w].value for w in self.DV[2]] for d in self.DV[1]],
                columns=self.DV[2], index=self.DV[1]
            ),
        ]

        # Save objective value
        self.obj_val = round(self.mod.obj.expr(), 2)

        # Save slack to dictionary with constraint suffix
        for c in self.mod.component_objects(pe.Constraint):
            self.slack[str(c).split("_")[1]] = c.slack()


    def display(self):
        st = self.layers
        descriptions = [f"{st[0]} to {st[1]} costs", f"{st[1]} to {st[2]} costs",
                        f"Output capacity from {st[0]}", f"Demand required from {st[2]}"]
        for description, df in zip(descriptions, [self.c1, self.c2, self.capacity, self.demand]):
            print(description)
            display(df)


    def print_dv_indexes(self):
        for i, stage in enumerate([self.DV[0], self.DV[1], self.DV[2]]):
            print(f"{self.layers[i]} stage locations: \n- ", end="")
            print(*stage, sep=", ")


    def print_slack(self):
        has_slack = [c for c in self.slack.keys() if self.slack[c] != 0]
        print(f"The following {len(has_slack)} constraints have slack:")
        print(*[f"{c}: {self.slack[c]}" for c in has_slack], sep="\n")


    def print_result(self):
        print("OBJECTIVE VALUE")
        print(f"Minimized Cost: ${self.obj_val}\n")
        print("DECISION VARIABLE QUANTITIES")
        print(f"{self.layers[0]} to {self.layers[1]}:")
        display(self.outputs[0].copy().astype(np.int64))
        print(f"{self.layers[1]} to {self.layers[2]}:")
        display(self.outputs[1].copy().astype(np.int64))


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





