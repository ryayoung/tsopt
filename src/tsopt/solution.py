# Maintainer:     Ryan Young
# Last Modified:  Jul 11, 2022

import pandas as pd
import numpy as np
import pyomo.environ as pe
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

from tsopt.data import DV



@dataclass
class Solution:
    dv: DV
    costs: list
    model: pe.ConcreteModel
    constraints: dict
    status: str
    termination_condition: str

    @property
    def obj_val(self):
        return self.model.obj.expr()

    @property
    def slack(self):
        return {
            str(key): val.slack() for key, val in self.constraints.items()
        }

    @property
    def quantities(self):
        return [pd.DataFrame(columns=df.columns, index=df.index,
                data=[[getattr(self.model, inp)[outp].value for outp in df.columns] for inp in df.index])
            for df in self.costs
        ]


    def display(self):
        print("OBJECTIVE VALUE")
        print(f"Minimized Cost: ${self.obj_val}\n")
        print("DECISION VARIABLE QUANTITIES")
        for i, df in enumerate(self.quantities):
            print(f'STAGE {i}: {self.dv.layers[i]} -> {self.dv.layers[i+1]}')
            display(self.quantities[i].copy().astype(np.int64))


    def show_slack(self):
        has_slack = [c for c in self.slack.keys() if self.slack[c] != 0]
        print(f"The following {len(has_slack)} constraints have slack:")
        print(*[f"{c}: {self.slack[c]}" for c in has_slack], sep="\n")


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

        route = pd.Series(route) + '-' + pd.Series(outflow_nodes)
        return route


    def stage_quantity_table(self, stage, sum_inflow=False, sum_outflow=False):
        '''
        Summarizes stage quantities
        '''
        df = self.quantities[stage-1]
        label_pref = (self.dv.abbrevs[stage-1], self.dv.abbrevs[stage])

        if sum_outflow:
            df = df.sum(axis=1).to_frame().rename(columns={0:label_pref[1]})
        if sum_inflow:
            df = df.sum().to_frame().T.rename(index={0:label_pref[0]})

        df = pd.melt(df).rename(columns={'variable':'outflow_nodes', 'value':'units'})
        df['Route'] = self.label_edges(
                sum_inflow, sum_outflow, self.dv.nodes[stage-1], df.outflow_nodes, label_pref[0])
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
            stage = [i for i in range(1, len(self.quantities)+1)]

        # Do multiple stages
        if type(stage) == list:
            if 'ax' in kwargs:
                raise TypeError("Can't use subplots when passing list of stages. (Can't take 'ax' argument when 'stage' argument is a list)")
            for s in stage:
                self.plot_stage_quantity(s, sum_inflow, sum_outflow, dynamic_width, figure, legend, **kwargs)
            return

        # Defaults based on stage
        if legend == dict():
            legend = dict(title=self.dv.layers[stage], loc='upper right')
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
            .set(xlabel=None, title=f'Quantity: {self.dv.layers[stage-1]} -> {self.dv.layers[stage]}')

        if sum_inflow == False and sum_outflow == False and legend != None:
            if 'ax' in kwargs:
                kwargs['ax'].legend(**legend)
            else:
                plt.legend(**legend)

        return result
