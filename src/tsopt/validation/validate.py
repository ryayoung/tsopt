# Maintainer:     Ryan Young
# Last Modified:  Aug 30, 2022

'''
What if, for validating upper bound, instead of using
infinity, use the network flow for each unconstrained edge!
'''

''' FROM model.py, add_edge_cosntraint()
valid = self.validate_edge_constraint(stage, df, sign, force)
if valid != True:
    return self.edge_constraints[stage][sign]
'''

def validate(mod) -> None:

    def layer_bounds(layer_idx:int, friendly_columns=False) -> pd.Series:
        cols = ['min', 'max'] if not friendly_columns else ['Demand', 'Capacity']
        df = mod.constraint_df_bounds[layer_idx]
        df.columns = cols
        return df

    def blank_stage_df(val, stage) -> pd.DataFrame:
        return pd.DataFrame(val,
                index=mod.cost[stage].index,
                columns=mod.cost[stage].columns,
        ).astype('float')


    def blank_constraint_df(val, layer, name) -> pd.Series:
        return pd.Series(val,
                index=mod.dv.nodes[layer],
                name=name,
        ).astype('float')


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
                f'Lower bound exceeds upper bound on edge {txt.comma_sep(bad_edges)}.'
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

        return True



    def validate_min_edge_constraint(self, edges:pd.DataFrame, stage:int) -> None:
        node_bounds = [ self.dv.templates.layer_bounds()[stage], self.dv.templates.layer_bounds()[stage+1]]
        # node_bounds = [ self.layer_bounds(stage), self.layer_bounds(stage+1)]
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
                            f"with {txt.plural(layrs[1])} exceed {nodes[0]}'s capacity.")

                elif len(nodes) > 1:
                    raise InfeasibleEdgeConstraint(f"Lower bounds on edges that connect {txt.plural(layrs[0])} "\
                        f"{txt.comma_sep(nodes)} to {txt.plural(layrs[1])} are infeasible. At each of these "\
                        f"{txt.plural(layrs[0])}, the sum of its lower bounded edges with {txt.plural(layrs[1])} "\
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
                            f"Lower bounds placed on edges connecting {layrs[0]} {nodes[0]} to {txt.plural(layrs[1])} "\
                            f"increased {nodes[0]}'s minimum value past its original demand, thus changing "\
                            f"the demand of the whole layer. {outro}"
                        )
                    elif len(nodes) > 1:
                        raise InfeasibleEdgeConstraint(
                            f"Lower bounds placed on edges connecting {txt.plural(layrs[0])} {txt.comma_sep(nodes)} "\
                            f"to {txt.plural(layrs[1])} are infeasible. For each of these {txt.plural(layrs[0])}, the "\
                            f"sum of the lower bounds on its {layrs[1]} edges exceeds its original demand, "\
                            f"thus affecting the demand of the whole layer. {outro}"
                        )


        sum_exceeds_flow_capacity()
        edges_within_node_capacity()
        inc_layer_demand_under_flow_cap()




    def validate_max_edge_constraint(self, edges:pd.DataFrame, stage:int) -> None:
        node_bounds = [ self.dv.templates.layer_bounds()[stage], self.dv.templates.layer_bounds()[stage+1]]
        # node_bounds = [ self.layer_bounds(stage), self.layer_bounds(stage+1)]
        layers = tuple([s for s in self.dv.layers[stage:stage+2]])

        def sum_less_than_demand() -> None:
            ''' Unlike sum_exceeds_flow_capacity in min edge constraints, we only
            enforce this upper bound rule if ALL edges have been constrained
            '''
            if not edges.isnull().any(None) and edges.values.sum() < self.flow_demand:
                raise InfeasibleEdgeConstraint(txt.dedent_wrap(f'''
                    Upper bounds were placed on all edges between {txt.plural(layers[0])} and
                    {txt.plural(layers[1])}. For this to be valid, the sum of the upper bounds on these edges must be
                    large enough to satisfy the demand of the network. The network's demand is determined
                    by whichever layer has the highest demand requirement (in this case, {self.flow_demand_layer}).
                    While the declared edge upper bounds may satisfy the demands of all {txt.plural(layers[0])}
                    and {txt.plural(layers[1])}, their sum ({int(edges.values.sum())} {self.units}) is insufficient
                    to meet the {txt.plural(self.flow_demand_layer)}' demand of ({int(self.flow_demand)} {self.units}).
                    '''))


        def edges_meet_node_demand() -> None:
            ''' Unlike edges_within_node_capacity in min edge constraints, we only
            enforce this rule for each node that has all of its edges constrained
            '''
            demands = [bound['min'] for bound in node_bounds]
            capacities = [edges.T.dropna(axis=1).sum(), edges.dropna(axis=1).sum()]
            bad = [[node for node in cap.index if dem[node] > cap[node]] for cap, dem in zip(capacities, demands)]
            for nodes, layrs in zip(bad, (layers, layers[::-1])):
                if len(nodes) == 1:
                    node = nodes[0]
                    raise InfeasibleEdgeConstraint(txt.dedent_wrap(f'''
                        Upper bounds were placed on ALL of {layrs[0]} {node}'s edges
                        with {txt.plural(layrs[1])}. For this to be valid, the sum of the upper bounds on those
                        edges must be high enough to meet {node}'s demand, since there aren't any unconstrained
                        edges left to balance it out.
                        '''))

                elif len(nodes) > 0:
                    raise InfeasibleEdgeConstraint(f'Upper bounds were placed on all edges connecting node(s) '\
                        f'{txt.comma_sep(nodes)} to {layrs[1]}. For this to be valid, at each node, the sum '\
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
                        raise InfeasibleEdgeConstraint(txt.dedent_wrap(f'''
                            Upper bounds were placed on all but one of {layrs[0]} {node}'s edges
                            with {txt.plural(layrs[1])}. This means {node}'s only unconstrained edge, {layrs[1]}
                            {from_nodes[0]}, is forced to provide at least {req} {self.units} to meet {node}'s
                            demand. However, {from_nodes[0]} is limited to a capacity of {avail} {self.units},
                            which is insufficient to meet the required demand.
                            '''))
                    raise InfeasibleEdgeConstraint(txt.dedent_wrap(f'''
                        Upper bounds were placed on some of {layrs[0]} {node}'s edges with
                        {txt.plural(layrs[1])}. These constraints may appear valid. However, to
                        meet {node}'s demand, its remaining unconstrained edges with {txt.plural(layrs[1])}
                        {txt.comma_sep(from_nodes)} need to provide at least {req} {self.units}. Those
                        {txt.plural(layrs[1])} each have their own capacities that cannot
                        be exceeded, and unfortunately, the most they can provide together is {avail}
                        {self.units}. Therefore, a solution is infeasible.
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
                    raise InfeasibleEdgeConstraint(txt.dedent_wrap(f'''
                        Upper bounds were placed on all but one of {txt.plural(layrs[0])} {txt.comma_sep(by_nodes)}'s
                        edges with {txt.plural(layrs[1])}. This means that {node}, the only {layrs[1]} to have
                        unconstrained edges with each of these {txt.plural(layrs[0])}, must carry at least {sum_req} {self.units}
                        to satisfy their demands. Unfortunately, {node}'s capacity of {avail} {self.units} is insufficient.
                    '''))
                raise InfeasibleEdgeConstraint(txt.dedent_wrap(f'''
                    Upper bounds were placed on {txt.plural(layrs[0])} {txt.comma_sep(by_nodes)}'s edges
                    with {txt.plural(layrs[1])}. The remaining unconstrained edges, where each of those
                    {txt.plural(layrs[0])} meets {txt.plural(layrs[1])} {txt.comma_sep(nodes)}, must
                    carry at least {sum_req} {self.units} to meet the demands of the {len(by_nodes)}
                    {txt.plural(layrs[0])}. Unfortunately, the {len(nodes)} {txt.plural(layrs[1])}
                    can together carry only {avail} {self.units}.
                    '''))

        sum_less_than_demand()
        edges_meet_node_demand()
        overflow_exceeds_cross_capacities()
