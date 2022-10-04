# Maintainer:     Ryan Young
# Last Modified:  Oct 04, 2022
import pandas as pd
import numpy as np
import pulp as pl
# import pyomo.environ as pe
# from pyomo.opt import SolverResults
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

from tsopt.edges import *
from tsopt.nodes import *


class SolvedModel:
    def __init__(self, mod):
        self.mod = mod
        self.dv = mod.dv
        self.pl_mod = mod.pl_mod

    # @property
    # def solver(self): return self.success.solver
    # @property
    # def status(self): return self.solver.status
    # @property
    # def termination_condition(self): return self.solver.termination_condition
    @property
    def obj_val(self):
        return pl.value(self.pl_mod.objective)

    # @property
    # def constraint_objects(self):
        # return {str(c): c for c in self.pl_mod.component_objects(pe.Constraint)}
    # @property
    # def slack(self):
        # return { str(key): val.slack() for key, val in self.constraint_objects.items() }
    @property
    def quantities(self):
        return StageEdges(self.mod, [ pd.DataFrame(columns=df.columns, index=df.index,
                data=[[getattr(self.pl_mod, inp)[outp].varValue for outp in df.columns] for inp in df.index])
            for df in self.dv.cost
        ])

    def display(self):
        print(f"MIN. COST: ${round(self.obj_val, 2):,}\n")
        print("FLOW QUANTITIES")
        for i, df in enumerate(self.quantities):
            print(f'{self.dv.layers[i]} -> {self.dv.layers[i+1]}')
            display(self.quantities[i].astype(np.int64))
