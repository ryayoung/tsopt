# Maintainer:     Ryan Young
# Last Modified:  Jul 21, 2022
import pandas as pd
import numpy as np
import pyomo.environ as pe
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from copy import deepcopy

from tsopt.data import SourceData
from tsopt.exceptions import InfeasibleConstraint, InfeasibleEdgeConstraint
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

        self.mod = pe.ConcreteModel()

        self.edge_constraints = self.new_edge_constraints()


    @property
    def constraints(self):
        return {str(c): c for c in self.mod.component_objects(pe.Constraint)}


    def layer_bounds(self, layer:int, friendly_columns=False) -> pd.Series:
        cols = ['min', 'max'] if not friendly_columns else ['Demand', 'Capacity']
        df = pd.concat([
                self.demand.get(layer, self.blank_constraint_df(0, layer, 'min')),
                self.capacity.get(layer, self.blank_constraint_df(np.inf, layer, 'max'))
            ], axis=1)
        df.columns = cols
        return df


    def blank_stage_df(self, val, stage) -> pd.DataFrame:
        return pd.DataFrame(val,
                index=self.cost[stage].index,
                columns=self.cost[stage].columns,
        ).astype('float')


    def blank_constraint_df(self, val, layer, name) -> pd.Series:
        return pd.Series(val,
                index=self.dv.nodes[layer],
                name=name,
        ).astype('float')


    def new_model(self) -> pe.ConcreteModel:
        self.edge_constraints = self.new_edge_constraints()
        return pe.ConcreteModel()


    def new_edge_constraints(self) -> dict:
        return {
            stage: {
                'min': self.blank_stage_df(np.nan, stage),
                'max': self.blank_stage_df(np.nan, stage),
            }
            for stage in self.range_stage()
        }


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


    def load_edge_constraint(self, filename, sign, stage=0, *args, **kwargs):
        df = self.process_file(filename)
        return self.add_edge_constraint(df, sign, stage, *args, **kwargs)


    def add_edge_constraint(self, loc:tuple or list or int or pd.DataFrame, sign:str, stage=0, value=None, force=False) -> pd.DataFrame:
        '''
        Constrain a single edge between two nodes.
        '''
        assert 0 <= stage <= len(self.cost)-1, f'Invalid stage, {stage}, for edge constraint {loc}'

        if not sign in ['<=', '>=', 'max', 'min']:
            raise ValueError("Invalid value for 'sign' parameter. Pyomo constraints require '<=' or '>='")

        if sign == '<=':
            sign = 'max'
        elif sign == '>=':
            sign = 'min'

        if type(loc) == pd.DataFrame:
            dimensions = loc.shape
            correct_dimensions = self.cost[stage].shape
            assert dimensions == correct_dimensions, \
                f'Edge constraint dataframe dimensions {dimensions} do not match the required dimensions '\
                f'{correct_dimensions} for stage {stage}.'
            df = loc
            df.index = self.dv.nodes[stage]
            df.columns = self.dv.nodes[stage+1]
        else:
            assert value != None, 'Must provide a constraint value when adding edge constraints by index'
            assert len(loc) in [1,2], 'edge constraint loc argument needs one or two elements'
            if type(loc) == tuple:
                loc = list(loc)
            elif type(loc) == int:
                loc = [loc, loc]

            for i, item in enumerate(loc):
                if type(item) != list:
                    if type(item) == tuple:
                        loc[i] = list(item)
                    elif type(item) == str or type(item) == int:
                        loc[i] = [item]
                    elif item != None:
                        raise ValueError(f"Invalid data type for 'loc' parameter in add_edge_constraint")

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

            df = self.blank_stage_df(np.nan, stage)
            for inp in loc[0]:
                for out in loc[1]:
                    df.loc[inp, out] = value


        valid = self.validate_edge_constraint(stage, df, sign, force)
        if valid != True:
            return self.edge_constraints[stage][sign]

        self.edge_constraints[stage][sign].update(df)

        return self.edge_constraints[stage][sign]


    def validate_edge_constraint(self, stage, df_new, sign, force=False) -> bool:
        '''
        Make sure the model is feasible
        '''
        def edge_bounds_valid(bounds:dict) -> None:
            ''' Edge lower bound can't exceed its upper bound '''
            df_diff = bounds['max'] - bounds['min']
            bad_edges = ['-'.join(pair) for pair in df_diff[df_diff < 0].stack().index.tolist()]
            if len(bad_edges) == 0:
                return
            signs_long = ('lower', 'upper') if sign == 'min' else ('upper', 'lower')
            raise InfeasibleEdgeConstraint(
                f'The {signs_long[0]} bounds placed on edges between {self.dv.layers[stage]} and '\
                f'{self.dv.layers[stage+1]} conflict with existing {signs_long[1]} bounds on those edges. '\
                f'Lower bound exceeds upper bound on edge {self.comma_sep(bad_edges)}.'
            )


        if force == True:
            return True

        dfs_all = deepcopy(self.edge_constraints[stage])
        dfs_all[sign].update(df_new)
        edge_bounds_valid(dfs_all)
        df_all = dfs_all[sign]
        if sign == 'min':
            self.validate_min_edge_constraint(df_all, stage)
        elif sign == 'max':
            self.validate_max_edge_constraint(df_all, stage)


        # df_all = self.edge_constraints[stage][sign].copy()
        # df_all_other = self.edge_constraints[stage][sign_opp].copy()
        # df_all.update(df_new)

        # for i, df in enumerate(dfs_full_constr_nodes):
            # layr = i+stage
            # layr_name = self.dv.layers[layr]
            # if df.empty:
                # continue
            # if sign == 'max':
                # if layr in self.capacity:
                    # cap = self.capacity[layr]
                    # updated_layer_capacity = df.values.sum() + cap[~cap.index.isin(df.index)].values.sum()
                    # assert updated_layer_capacity >= self.flow_demand, \
                        # f'One or more nodes in {layr_name} have been fully-constrained to a maximum value low enough ' \
                        # f'that the total capacity of {layr_name} is too low to meet the overall flow demand of the network.'


        # changes = [ df_all.sum(axis=1), df_all.sum() ]
        # changes_opp = [ df_all.sum(), df_all.sum(axis=1) ]
        # bounds = [ self.layer_bounds(stage), self.layer_bounds(stage+1) ]
        # mins = [ bounds[0]['min'], bounds[1]['min'] ]
        # opp_maxs = [ bounds[1]['max'], bounds[0]['max'] ]
        # df_alls = [df_all, df_all.T]
        # layer_opps = [self.dv.layers[stage+1], self.dv.layers[stage]]
        # if sign == 'max':
            # for i, (demand, opp_cap, edge_cap, edge_cap_opp, dfa, layer_opp) in enumerate(zip(mins, opp_maxs, changes, changes_opp, df_alls, layer_opps)):
                # min_of_remaining = demand - edge_cap
                # nodes = demand.index
                # opp_nodes = opp_cap.index
                # for node in nodes:
                    # required = min_of_remaining[node]
                    # available = [node_opp for node_opp in opp_nodes if pd.isna(dfa.loc[node, node_opp])]
                    # capacity = opp_cap[opp_cap.index.isin(available)].sum()
                    # if capacity < required:
                        # raise InfeasibleEdgeConstraint(
                            # f'Custom edge constraints between {self.dv.layers[stage]} and {self.dv.layers[stage+1]} '\
                            # f'declaring maximum values on edges under node {node} are infeasible, '\
                            # f'because in order for the unconstrained edges between {layer_opp} '\
                            # f'and {node} to meet the total demand of {node} ({int(demand[node])} units), the sum '\
                            # f'of their values would exceed the sum of their respective capacities in {layer_opp} '\
                            # f'({int(capacity)} units). In this case, the unconstrained edges need at least '\
                            # f'{int(required)} units of flow to meet the demand of {node}. The node(s) lacking the '\
                            # f"capacity to provide this volume are {self.plural(layer_opp)} {self.comma_sep(available)}"
                        # )


        # changes = [ df_all.sum(axis=1), df_all.sum() ]
        # bounds = [ self.layer_bounds(stage), self.layer_bounds(stage+1) ]
        # old_max = [s['max'] for s in bounds]
        # old_min = [s['min'] for s in bounds]

        # if sign == 'max':
            # changes_inf = [ c.replace(0, np.inf) for c in changes ]
            # new_max = [ np.minimum(s1, s2) for s1, s2 in zip(changes_inf, old_max) ]
            # overages = [ list((new.sum() - old).loc[lambda x: x < 0].index)
                # for new, old in zip(new_max, old_max)
            # ]
            # for i, nodes in enumerate(overages):
                # if len(nodes) > 0:
                    # raise InfeasibleEdgeConstraint(
                        # f'Custom edge constraints between {self.dv.layers[stage]} and {self.dv.layers[stage+1]} '\
                        # f'lowered the maximum total capacity of {self.dv.layers[stage+i]} to be less than the '\
                        # f"maximum capacity of node(s) {', '.join(nodes[:10])}. Please increase the upper bound "\
                        # f'placed on edges within {self.dv.layers[stage+i]}, or increase the capacity of this layer.'
                    # )


        return True


    def validate_min_edge_constraint(self, edges:pd.DataFrame, stage:int) -> None:
        node_bounds = [ self.layer_bounds(stage), self.layer_bounds(stage+1)]
        layers = tuple(self.dv.layers[stage:stage+2])

        def sum_exceeds_flow_capacity() -> None:
            if edges.values.sum() > self.flow_capacity:
                raise InfeasibleEdgeConstraint(f'Lower bounds on edges between {layers[0]} and {layers[1]} are '\
                    f'infeasible. The maximum capacity of the network is limited to whichever layer has the lowest '\
                    f'total capacity. Therefore, while the sum of these lower bounds may or may not satisfy the '\
                    f'capacities of {layers[0]} and {layers[1]}, it exceeds the overall network capacity.'
                )

        def edges_within_node_capacity() -> None:
            '''
            Unlike edges_meet_node_demand in max edge constraints, it only takes one
            bad edge to exceed a node's capacity, so we must check all constrained nodes
            instead of just the fully constrained ones.
            '''
            node_capacities = [ bound['max'] for bound in node_bounds ]
            edge_sums = [ edges.sum(axis=1), edges.sum(axis=0) ]
            bad_nodes = [ [ n for n in dem.index if cap[n] < dem[n] ]
                    for dem,cap in zip(edge_sums, node_capacities) ]

            for nodes, layrs in zip(bad_nodes, (layers, layers[::-1])):
                if len(nodes) == 1:
                    raise InfeasibleEdgeConstraint(f"The sum of the lower bounds on {layrs[0]} {nodes[0]}'s edges "\
                            f"with {self.plural(layrs[1])} exceed {nodes[0]}'s capacity.")

                elif len(nodes) > 1:
                    raise InfeasibleEdgeConstraint(f"Lower bounds on edges that connect {self.plural(layrs[0])} "\
                        f"{self.comma_sep(nodes)} to {self.plural(layrs[1])} are infeasible. At each of these "\
                        f"{self.plural(layrs[0])}, the sum of its lower bounded edges with {self.plural(layrs[1])} "\
                        f"exceeds that {layrs[0]}'s capacity.")


        def inc_layer_demand_under_flow_cap() -> None:
            '''
            At this point, we know each node's edge bounds are valid against its own capacity. However,
            the capacities we're testing against might not be the ones limiting the flow of the model.
            Say we're setting lower bounds on Warehouse edges. Current total warehouse demand is 300
            units, total capacity is 400. W1 demand is 20. If we set 40 units worth of lower bounds on
            W1's edges, the above tests would pass. Demand increased, but is within W1's bounds. However,
            network capacity was actually 300 units, as constrained by a different layer.  Solution:
            New layer demand = MAX(node edge sum, original node demand). Compare this to network capacity.
            '''
            old = [s['min'] for s in node_bounds]
            added = [edges.sum(axis=1), edges.sum(axis=0)]
            new_demand = [ np.maximum(edge,node) for edge,node in zip(added, old)]
            nodes_over = [ list((old<new).index[old<new]) for new,old in zip(added, old)]

            for new, nodes, layrs in zip(new_demand, nodes_over, (layers, layers[::-1])):
                if new.sum() > self.flow_capacity:
                    outro = f"Total {layrs[0]} demand is increased to {new.sum()} units. "\
                        f"The network flow capacity is determined by whichever layer has the lowest "\
                        f"capacity, and all other layers will be limited to this flow rate. So, "\
                        f"although this demand may or may not be feasible within {layrs[0]}'s capacity, "\
                        f"it exceeds the network's flow capacity of {self.flow_capacity} units."

                    if len(nodes) == 1:
                        raise InfeasibleEdgeConstraint(
                            f"Lower bounds placed on edges connecting {layrs[0]} {nodes[0]} to {self.plural(layrs[1])} "\
                            f"increased {nodes[0]}'s minimum value past its original demand, thus changing "\
                            f"the demand of the whole layer. {outro}"
                        )
                    elif len(nodes) > 1:
                        raise InfeasibleEdgeConstraint(
                            f"Lower bounds placed on edges connecting {self.plural(layrs[0])} {self.comma_sep(nodes)} "\
                            f"to {self.plural(layrs[1])} are infeasible. For each of these {self.plural(layrs[0])}, the "\
                            f"sum of the lower bounds on its {layrs[1]} edges exceeds its original demand, "\
                            f"thus affecting the demand of the whole layer. {outro}"
                        )


        sum_exceeds_flow_capacity()
        edges_within_node_capacity()
        inc_layer_demand_under_flow_cap()




    def validate_max_edge_constraint(self, edges:pd.DataFrame, stage:int) -> None:
        node_bounds = [ self.layer_bounds(stage), self.layer_bounds(stage+1)]
        layers = tuple([s.lower() for s in self.dv.layers[stage:stage+2]])

        def sum_less_than_demand() -> None:
            ''' Unlike sum_exceeds_flow_capacity in min edge constraints, we only
            enforce this upper bound rule if ALL edges have been constrained
            '''
            if not edges.isnull().any(None) and edges.values.sum() < self.flow_demand:
                raise InfeasibleEdgeConstraint(f'Upper bounds were placed on all edges between {layers[0]} and '\
                    f'{layers[1]}. For this to be valid, the sum of the upper bounds on these edges must be '\
                    f"large enough to satisfy the demand of the network. The network's demand is determined "\
                    f'by whichever layer has the highest demand requirement. While the declared edge upper '\
                    f'bounds may or may not satisfy the demands of {layers[0]} and {layers[1]}, their sum '\
                    f'({edges.values.sum()} units) is insufficient to meet the network demand of ({self.flow_demand} units).')


        def edges_meet_node_demand() -> None:
            ''' Unlike edges_within_node_capacity in min edge constraints, we only
            enforce this rule for each node that has all of its edges constrained
            '''
            demands = [bound['min'] for bound in node_bounds]
            capacities = [edges.T.dropna(axis=1).sum(), edges.dropna(axis=1).sum()]
            bad = [[node for node in cap.index if dem[node] > cap[node]] for cap, dem in zip(capacities, demands)]
            for nodes, layrs in zip(bad, (layers, layers[::-1])):
                if len(nodes) == 1:
                    raise InfeasibleEdgeConstraint(f"Upper bounds were placed on ALL of {layrs[0]} {nodes[0]}'s edges "\
                        f"with {self.plural(layrs[1])}. For this to be valid, the sum of the upper bounds on those "\
                        f"edges must be high enough to meet {layrs[0]}'s demand, since there aren't any unconstrained "\
                        f"edges left to balance it out.")

                elif len(nodes) > 0:
                    raise InfeasibleEdgeConstraint(f'Upper bounds were placed on all edges connecting node(s) '\
                        f'{self.comma_sep(nodes)} to {layrs[1]}. For this to be valid, at each node, the sum '\
                        f'of the upper bounds of all its edges in a given stage must be high enough to meet '\
                        f"the node's demand, since there aren't any unconstrained edges left to balance it out.")


        def overflow_exceeds_cross_capacities() -> None:
            '''
            At this point, we've ensured that a node's edge upper bounds can't prevent
            its demand from being met, and tested this on the layer level as well.
            ---
            A non-obvious consequence of edge upper bounds is overflow. It's non-obvious
            because the constraints that ultimately determine the flow's feasibility
            exist in the layer opposite to that in which the redirection of flow is caused.
            When the majority of edges under a node are constrained to a relatively
            low upper bound, the remaining flow needed to satisfy the node's demand
            is forced through the remaining unconstrained edges, increasing their flow.
            From the previous validation checks, we already know this change in flow
            won't conflict with this node or its layer constraints.  However, the edges
            attempting to carry the extra flow must ALSO satisfy the constraints of the
            other nodes they're connected to on the opposite layer.  If the opposing
            nodes don't provide enough capacity for these edges to carry the needed
            flow, then a solution is infeasible.
            '''
            changes = [ edges.sum(axis=1), edges.sum() ]
            cross_cap = [ node_bounds[1]['max'], node_bounds[0]['max'] ]
            df_flops = [edges, edges.T]
            required = [b['min']-change for b, change in zip(node_bounds, changes)]
            cross_req = [ [ {
                        'node': k,
                        'req': v,
                        'from': cap[cap.index.isin([c for c in df.columns if pd.isna(df.loc[k,c])])]
                        }
                    for k, v in req.to_dict().items() if df.notna().any(1)[k] ]
                for req, df, cap in zip(required, df_flops, cross_cap)
            ]
            cross_req = [[dict(item, **{'avail': item['from'].sum()}) for item in layr] for layr in cross_req]
            cross_req_invalid = [[i for i in layr if i['avail'] < i['req']] for layr in cross_req]
            if any([len(l) > 0 for l in cross_req_invalid]):
                for meta, layrs in zip(cross_req_invalid, (layers, layers[::-1])):
                    if len(meta) == 0: continue
                    first = meta[0]
                    node, req = first['node'], first['req']
                    from_nodes, avail = list(first['from'].index), first['avail']
                    if len(from_nodes) == 1:
                        raise InfeasibleEdgeConstraint(self.dedent_wrap(f'''
                            Upper bounds were placed on all but one of {layrs[0]} {node}'s edges
                            with {self.plural(layrs[1])}. This means {node}'s only unconstrained edge, {layrs[1]}
                            {from_nodes[0]}, is forced to provide at least {req} units to meet {node}'s
                            demand. However, {from_nodes[0]} is limited to a capacity of {avail} units,
                            which is insufficient to meet the required demand.
                            '''))
                    raise InfeasibleEdgeConstraint(self.dedent_wrap(f'''
                        Upper bounds were placed on some of {layrs[0]} {node}'s edges with
                        {self.plural(layrs[1])}. These constraints may appear valid. However, to
                        meet {node}'s demand, its remaining unconstrained edges with {self.plural(layrs[1])}
                        {self.comma_sep(from_nodes)} need to provide at least {req} units. Those
                        {self.plural(layrs[1])} each have their own capacities that cannot
                        be exceeded, and unfortunately, the most they can provide together is {avail}
                        units. Therefore, a solution is infeasible.
                        '''))

            assignments = [list(set([tuple(node['from'].index) for node in layr])) for layr in cross_req]
            assignments = [{nodes: {
                    'sum_req': sum([node['req'] for node in cross_req[i] if tuple(node['from'].index) == nodes]),
                    'by': tuple([node['node'] for node in cross_req[i] if tuple(node['from'].index) == nodes]),
                    'avail': [node['avail'] for node in cross_req[i] if tuple(node['from'].index) == nodes][0]
                    } for nodes in layr }
                for i, layr in enumerate(assignments)
            ]
            assignments_invalid = [{k:v for k,v in layr.items() if v['sum_req'] > v['avail']} for layr in assignments]
            if all([len(l) == 0 for l in assignments_invalid]):
                return
            for conflicts, layrs in zip(assignments_invalid, (layers, layers[::-1])):
                if len(conflicts) == 0: continue
                nodes, meta = list(conflicts.items())[0]
                sum_req, by_nodes, avail = int(meta['sum_req']), meta['by'], int(meta['avail'])
                if len(nodes) == 1:
                    node = nodes[0]
                    raise InfeasibleEdgeConstraint(self.dedent_wrap(f'''
                        Upper bounds were placed on all but one of {self.plural(layrs[0])} {self.comma_sep(by_nodes)}'s
                        edges with {self.plural(layrs[1])}. This means that {node}, the only {layrs[1]} to have
                        unconstrained edges with each of these {self.plural(layrs[0])}, must carry at least {sum_req} units
                        to satisfy their demands. Unfortunately, {node}'s capacity of {avail} units is insufficient.
                    '''))
                raise InfeasibleEdgeConstraint(self.dedent_wrap(f'''
                    Upper bounds were placed on {self.plural(layrs[0])} {self.comma_sep(by_nodes)}'s edges
                    with {self.plural(layrs[1])}. The remaining unconstrained edges, which are between those
                    {self.plural(layrs[0])} and {self.plural(layrs[1])} {self.comma_sep(nodes)}, must
                    carry at least {sum_req} units of flow to meet the demands of the {len(by_nodes)}
                    {self.plural(layrs[0])}. Unfortunately, these {len(nodes)} {self.plural(layrs[1])}
                    can together carry only {avail} units.
                    '''))

        # def multiple_overflows_exceed_cross_capacities() -> None:





        sum_less_than_demand()
        edges_meet_node_demand()
        overflow_exceeds_cross_capacities()



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

                        if hasattr(self.mod, constr_name): # replace existing constraint on that edge
                            delattr(self.mod, constr_name)
                        edge = getattr(self.mod, inp)[out]

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
        for stage in self.range_stage():
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
        final_layr_idx = len(self)-1
        for i, coefs in self.capacity.items():
            if i < final_layr_idx:
                curr_nodes = self.dv.nodes[i]
                output_nodes = self.dv.nodes[i+1]
                for node_curr in curr_nodes:
                    val = coefs[node_curr]
                    expr = val >= sum(getattr(self.mod, node_curr)[node_out] for node_out in output_nodes)
                    self.add_constraint(node_curr, expr, 'capacity')
            elif i == final_layr_idx:
                input_nodes = self.dv.nodes[i-1]
                curr_nodes = self.dv.nodes[i]
                for node_curr in curr_nodes:
                    val = coefs[node_curr]
                    expr = val >= sum(getattr(self.mod, node_in)[node_curr] for node_in in input_nodes)
                    self.add_constraint(node_curr, expr, 'capacity')


    def set_demand_constraints(self):
        '''
        Lower bound for units passing through nodes
        --
        If layer > 0: calculated as minimum input from all previous nodes
        If layer == 0: calculated as minimum output to all following nodes
        '''
        for i, coefs in self.demand.items():
            if i > 0:
                input_nodes = self.dv.nodes[i-1]
                curr_nodes = self.dv.nodes[i]
                for node_curr in curr_nodes:
                    val = coefs[node_curr]
                    expr = val <= sum(getattr(self.mod, node_in)[node_curr] for node_in in input_nodes)
                    self.add_constraint(node_curr, expr, 'demand')
            elif i == 0:
                curr_nodes = self.dv.nodes[i]
                output_nodes = self.dv.nodes[i+1]
                for node_curr in curr_nodes:
                    val = coefs[node_curr]
                    expr = val <= sum(getattr(self.mod, node_curr)[node_out] for node_out in output_nodes)
                    self.add_constraint(node_curr, expr, 'demand')


    def set_flow_constraints(self):
        '''
        Equal flow for all stages.
        A flow is a layer which takes input from a previous layer and
        outputs to the next layer. If there are 3 layers, then there must
        be one flow: the middle layer.
        '''
        for flow in self.range_layer(1, -1):
            for node_curr in self.dv.nodes[flow]:
                input_nodes = self.dv.nodes[flow-1]
                output_nodes = self.dv.nodes[flow+1]
                inflow = sum([getattr(self.mod, node_in)[node_curr] for node_in in input_nodes])
                outflow = sum([getattr(self.mod, node_curr)[node_out] for node_out in output_nodes])
                expr = inflow == outflow
                name = f'{node_curr}_{self.dv.abbrevs[flow+1]}'
                self.add_constraint(name, expr, 'flow')


    def build_model(self) -> pe.ConcreteModel:

        self.set_decision_vars()
        self.set_objective()
        self.set_capacity_constraints()
        self.set_demand_constraints()
        self.set_flow_constraints()
        self.set_edge_constraints()

        return self


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
        bounds = {i: self.layer_bounds(i, friendly_columns=True) for i in self.range_layer()}
        for i, df in bounds.items():
            print(f'{self.dv.layers[i]}:')
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


