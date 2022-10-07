# Maintainer:     Ryan Young
# Last Modified:  Oct 04, 2022
import matplotlib.pyplot as plt
import seaborn as sns

from tsopt.edges import *
from tsopt.nodes import *
from tsopt.util import *


class QuantityPlots:
    def __init__(self, mod, vals):
        self.mod = mod
        self.vals = vals
        self.layers, self.abbrevs = self.mod.dv.layers, self.mod.dv.abbrevs
        self.node = NodeQuantityPlots(mod, vals.node)
        self.edge = EdgeQuantityPlots(mod, vals)



class NodeQuantityPlots:
    def __init__(self, mod, nodes):
        self.mod = mod
        self.layers = self.mod.dv.layers
        self.abbrevs = self.mod.dv.abbrevs
        self.nodes = nodes


    def show(self, layer):
        sr = self.nodes[layer]
        df = pd.DataFrame(sr).reset_index()
        df.columns = ['node', 'val']

        figsize = smart_width((10,4), df.shape.rows)
        plt.figure(figsize=figsize)

        result = sns.barplot(x='node', y='val', data=df, dodge=False, color='#1F77B4') \
            .set(xlabel=None, ylabel=self.mod.units, title=f'Quantity: {self.layers[layer]}')
        return result



class EdgeQuantityPlots:
    def __init__(self, mod, edges):
        self.mod = mod
        self.layers = self.mod.dv.layers
        self.abbrevs = self.mod.dv.abbrevs
        self.edges = edges
        self.in_out = ['inp', 'out']

    '''
    - Plot all edges in stage
    - Plot all edges for one node
    - Plot all edges for one node in a stage
    '''

    def plot_stage(self, stage, hue=0):
        '''
        Show all edges in a stage.
        - x-labels: A1-B3, etc.
        - Hue based on output nodes
        '''
        df = self.edges.melted[stage]

        order = binary_reverse(hue, self.in_out)
        layers = binary_reverse(hue, self.layers[stage:stage+2])

        df = df.sort_values(by=order)

        df['route'] = df.inp + '-' + df.out

        title = f'Quantity: {layers[0]} -> {layers[1]}'
        legend = dict(title=layers[0], loc='upper right')
        figsize = smart_width((12,5), df.shape.rows)
        plt.figure(figsize=figsize)

        result = sns.barplot(x='route', y='val', data=df, dodge=False, hue=order[0]) \
                .set(xlabel=None, ylabel=self.mod.units, title=title)

        plt.legend(**legend)

        return result


    def _stage_node_df(self, stage, node_loc, one_sided_label=True):
        '''
        Node format:
            - str: 'A1'
            - tuple or list: (layer:int, node:int)
                - Layer must be one of the two in current stage.
                  If passing int, 0 represents input layer
        --
        Returns layer:int (0 or 1), node label ('A1')
        - Layer represents index of self.in_out
        '''
        df = self.edges[stage]
        layer, node_idx = StageEdges.find_node(df, node_loc)
        node = f'{self.abbrevs[stage+layer]}{node_idx}'
        df = self.edges.melted[stage]
        df = df[df[self.in_out[layer]] == node]
        if one_sided_label:
            order = binary_reverse(layer, self.in_out)
            name_parts = binary_reverse(layer, ['->', df[order[1]]])
            df['route'] = name_parts[0] + name_parts[1]
        else:
            df['route'] = df[self.in_out[0]] + '-' + df[self.in_out[1]]
        return (df, layer, node)


    def plot_stage_node(self, stage, node_loc):
        df, layer, node = self._stage_node_df(stage, node_loc, one_sided_label=False)
        return df


        layers = binary_reverse(layer, self.layers[stage:stage+2])
        title_parts = binary_reverse(layer, [f'Node {node}', f''])
        title = f'Quantity: '
        figsize = smart_width((10,5), df.shape.rows)
        plt.figure(figsize=figsize)

        result = sns.barplot(x='route', y='val', data=df, dodge=False) \
                .set(xlabel=None, ylabel=self.mod.units, title=f'Quantity: {layers[0]} -> {layers[1]}')



        return df




    def plot_multi_stage_node(self, node):
        '''
        Node format:
            - str: 'A1'
            - tuple or list: (layer:?, node:int)
                - Any layer can be specified
        '''



















