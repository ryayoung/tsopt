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
        output_stage1 = pd.DataFrame()
        output_stage2 = pd.DataFrame()
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





