# Maintainer:     Ryan Young
# Last Modified:  Oct 07, 2022
import pandas as pd
import numpy as np
import pulp as pl
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

from tsopt.edges import *
from tsopt.nodes import *
from tsopt.solved.plots import *


class SolvedModel:
    def __init__(self, mod):
        self.mod = mod
        self.dv = mod.dv
        self.pl_mod = mod.pl_mod
        self.plot = QuantityPlots(self.mod, self.quantities)

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
        return EdgeQuantities(self.mod, [ pd.DataFrame(columns=df.columns, index=df.index,
                data=[[getattr(self.pl_mod, inp)[outp].varValue for outp in df.columns] for inp in df.index])
            for df in self.dv.costs
        ])

    def display(self):
        print(f"MIN. COST: ${round(self.obj_val, 2):,}\n")
        print("FLOW QUANTITIES")
        for i, df in enumerate(self.quantities):
            print(f'{self.dv.layers[i]} -> {self.dv.layers[i+1]}')
            display(self.quantities[i].astype(np.int64))


    # PLOTTING ---------------------------------------------------------------
    # @staticmethod
    # def label_edges(
            # sum_inflow:bool,
            # sum_outflow:bool,
            # inflow_nodes:list,
            # outflow_nodes:list,
            # inflow_abbrev:str,
            # ):
        # '''
        # Helper function for plot_stage to create a column that
        # accurately labels each edge
        # '''
        # if sum_outflow and sum_inflow:
            # route = inflow_abbrev
        # elif sum_inflow:
            # route = [inflow_abbrev] * len(outflow_nodes)
        # elif sum_outflow:
            # route = inflow_nodes
        # else:
            # route = inflow_nodes * len(outflow_nodes)
# 
        # route = pd.Series(route) + '-' + pd.Series(outflow_nodes)
        # return route


    def stage_quantity_table(self, stage, sum_inflow=False, sum_outflow=False):
        '''
        Summarizes stage quantities
        '''
        df = self.quantities.melted[stage]
        col_inp, col_out = 'inp', 'out'
        if sum_inflow or sum_outflow:
            cols = (col_inp, col_out)
            abbrevs = self.dv.abbrevs[stage:stage+2]
            if sum_outflow:
                cols, abbrevs = cols[::-1], abbrevs[::-1]

            df = df.groupby(cols[0]).sum(numeric_only=True).reset_index()
            df[cols[1]] = abbrevs[1]

        df['Route'] = df[col_inp] + '-' + df[col_out]
        df['outflow_nodes'] = df[col_out]
        df = df.drop(columns=[col_inp, col_out])
        return df
        # label_pref = (self.dv.abbrevs[stage], self.dv.abbrevs[stage+1])
# 
        # if sum_outflow:
            # df = df.sum(axis=1).to_frame().rename(columns={0:label_pref[1]})
        # if sum_inflow:
            # df = df.sum().to_frame().T.rename(index={0:label_pref[0]})
# 
        # df = pd.melt(df).rename(columns={'variable':'outflow_nodes', 'value':'units'})
        # df['Route'] = self.label_edges(
                # sum_inflow, sum_outflow, self.dv.nodes[stage], df.outflow_nodes, label_pref[0])
        # return df


    def smart_width(self, size:tuple, rows) -> tuple:
        '''
        Changes first element of size tuple based on num of rows.
        Only has effect when number of rows is less than 3
        '''
        w, h = size
        if rows < 4:
            w = w // 1.5 + 1
        if rows < 3:
            w = w // 2 + 1
        return (w, h)


    def plot_quantity(self, stage=None, *args, **kwargs):
        if stage == None:
            stage = [*self.dv.range_stage()]

        if is_list_or_tuple(stage):
            if 'ax' in kwargs:
                raise TypeError("Can't use subplots when passing list of stages. " \
                        "(Can't take 'ax' argument when 'stage' argument is a list)")
            for s in stage:
                self.plot_stage_quantity(s, *args, **kwargs)
        else:
            return self.plot_stage_quantity(stage, *args, **kwargs)


    def plot_stage_quantity(self,
            stage,
            sum_inflow=False,
            sum_outflow=False,
            dynamic_width=True,
            figure=dict(),
            legend=dict(),
            **kwargs
            ):

        layr_curr, layr_next = self.dv.layers[stage:stage+2]

        # Defaults based on stage
        if legend == dict():
            legend = dict(title=layr_next, loc='upper right')
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

        title_inflow = f'SUM({layr_curr})' if sum_inflow else layr_curr
        title_outflow = f'SUM({layr_next})' if sum_outflow else layr_next
        result = sns.barplot(x='Route', y='val', data=df, dodge=False, **kwargs, **optional) \
            .set(xlabel=None, title=f'Quantity: {title_inflow} -> {title_outflow}')

        if sum_inflow == False and sum_outflow == False and legend != None:
            if 'ax' in kwargs:
                kwargs['ax'].legend(**legend)
            else:
                plt.legend(**legend)

        return result
