# -------------------------------------------------------------------------------------
# Implementation of the MILP method discussed in the paper:
# Exact Approaches to the Multi-agent Collective Construction Problem by E.Lam, et al.
# https://link.springer.com/chapter/10.1007/978-3-030-58475-7_43

# Adapted from the original code provided by E. Lam.
# -------------------------------------------------------------------------------------

import multiprocessing
from gurobipy import *
from math import ceil


def MILP(max_agents, T, X, Y, Z, structure, time_limit, threads, sol_height, sol_paths, sol_pickup, sol_delivery):

    # initialize the model
    model = Model('MILP')

    # define model parameters
    model.Params.Method = 3     # concurrent optimization
    model.Params.Threads = threads
    model.Params.TimeLimit = time_limit

    # create borders
    borders = set(
        [(x2, 0, 0) for x2 in range(X)] +
        [(x2, Y - 1, 0) for x2 in range(X)] +
        [(0, y2, 0) for y2 in range(1, Y - 1)] +
        [(X - 1, y2, 0) for y2 in range(1, Y - 1)]
    )

    # create variables that move agents from outside the grid to the borders
    agent = tupledict()     # tupledict is a subclass of dict where the keys are a tuplelist
    x = y = z = 'start'
    action = 'move'
    for t in range(T-3):
        for c in range(2):
            for (x2, y2, z2) in borders:
                # add actions that are available to the agent
                agent[t, c, x, y, z, action, x2, y2, z2] = model.addVar(vtype=GRB.BINARY)

    # create variables that move agents from the borders to the inside of the grid
    action = 'move'
    for t in range(1, T-2):
        for c in range(2):
            for x in range(X):
                for y in range(Y):
                    for z in range(Z):

                        # Move to neighbor cell
                        for (x2, y2) in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
                            for z2 in [z-1, z, z+1]:
                                if 0 <= x2 < X and 0 <= y2 < Y and 0 <= z2 < Z:
                                    agent[t, c, x, y, z, action, x2, y2, z2] = model.addVar(vtype=GRB.BINARY)

                        # Stay in the same cell
                        (x2, y2, z2) = (x, y, z)
                        agent[t, c, x, y, z, action, x2, y2, z2] = model.addVar(vtype=GRB.BINARY)

    # create variables that move agent from the border to the end vertex
    x2 = y2 = z2 = 'end'
    action = 'move'
    for t in range(2, T - 1):
        for c in range(2):
            for (x, y, z) in borders:
                assert (t, c, x, y, z, action, x2, y2, z2) not in agent
                agent[t, c, x, y, z, action, x2, y2, z2] = model.addVar(vtype=GRB.BINARY)

    # create variables that allows pickup action
    action = 'pickup'
    for t in range(1,T-2):
        c = 0
        for x in range(X):
            for y in range(Y):
                for z in range(Z-1):
                    for (x2, y2) in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
                        if 0 <= x2 < X and 0 <= y2 < Y:
                            z2 = z
                            agent[t, c, x, y, z, action, x2, y2, z2] = model.addVar(vtype=GRB.BINARY)

    # create variables that allows delivery action
    action = 'delivery'
    for t in range(1,T-2):
        c = 1
        for x in range(X):
            for y in range(Y):
                for z in range(Z-1):
                    for (x2,y2) in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]:
                        if 0 <= x2 < X and 0 <= y2 < Y:
                            z2 = z
                            agent[t, c, x, y, z, action, x2, y2, z2] = model.addVar(vtype=GRB.BINARY)

    # create variables that indicates height of the structures
    height = tupledict()
    for t in range(T - 1):
        for x in range(X):
            for y in range(Y):
                for z in range(Z):
                    for z2 in [z - 1, z, z + 1]:
                        if 0 <= z2 < Z:
                            height[t, x, y, z, z2] = model.addVar(vtype=GRB.BINARY)

    # process any pending model modifications / i.e., update the model to make sure all the variables are added
    model.update()

    # define the objective function (i.e., sum-of-costs)
    # add cost to every action taken
    for ((t, c, x, y, z, action, x2, y2, z2), var) in agent.items():
        if x != 'start':
            var.Obj = 1

    # define the constraint that height of borders is 0
    for t in range(T-1):
        for (x,y,z) in borders:
            model.addConstr(height[t, x, y, z, z] == 1)

    # define the constraint that height is 0 at the first two time steps
    t = 0
    for x in range(X):
        for y in range(Y):
            model.addConstr(height[t, x, y, 0, 0] == 1)

    # define the constraint that height is 0 at the last two time steps
    t = T-2
    for x in range(X):
        for y in range(Y):
            z = structure[x, y] if (x, y) in structure else 0
            model.addConstr(height[t, x, y, z, z] == 1)

    # define the constraint that exactly one height at each cell and time step
    for t in range(T-1):
        for x in range(X):
            for y in range(Y):
                # '*' means wildcard, i.e., any value
                model.addConstr(quicksum(height.select(t, x, y, '*', '*')) == 1)

    # define the constraint of changes in height
    for t in range(T-2):
        for x in range(X):
            for y in range(Y):
                for z in range(Z):
                    model.addConstr(
                        quicksum(height.select(t, x, y, '*', z))
                        ==
                        quicksum(height.select(t+1, x, y, z, '*')),
                    )

    # define the constraint of maximum number of agents
    for t in range(T):
        model.addConstr(
            quicksum(
                var for ((tt, c, x, y, z, action, x2, y2, z2), var) in agent.items() if tt == t)
            # Include start here because an agent takes one step to exit then come back into the map
            <= max_agents
        )

    # define the flow
    for t in range(T-2):
        for x in range(X):
            for y in range(Y):
                for z in range(Z):
                    model.addConstr(
                        quicksum(agent.select(t, 0, '*', '*', '*', 'move', x, y, z)) +
                        quicksum(agent.select(t, 1, x, y, z,'delivery', '*', '*', '*'))
                        ==
                        quicksum(agent.select(t+1, 0, x, y, z, 'move', '*', '*', '*')) +
                        quicksum(agent.select(t+1, 0, x, y, z, 'pickup', '*', '*', '*'))
                    )

    # Flow - carrying
    for t in range(T-2):
        for x in range(X):
            for y in range(Y):
                for z in range(Z):
                    model.addConstr(
                        quicksum(agent.select(t, 1, '*', '*', '*', 'move', x, y, z)) +
                        quicksum(agent.select(t, 0, x, y, z, 'pickup', '*', '*', '*'))
                        ==
                        quicksum(agent.select(t+1, 1, x, y, z, 'move', '*', '*', '*')) +
                        quicksum(agent.select(t+1, 1, x, y, z, 'delivery', '*', '*', '*'))
                    )

    # avoid agent vertex collision
    for t in range(1, T-1):
        for x in range(X):
            for y in range(Y):
                model.addConstr(
                    quicksum(agent.select(t, '*', x, y, '*', '*', '*', '*', '*')) +
                    quicksum(agent.select(t, '*', '*', '*', '*', 'pickup', x, y, '*')) +
                    quicksum(agent.select(t, '*', '*', '*', '*', 'delivery', x, y, '*'))
                    <= 1
                )

    # avoid agent edge collision
    for t in range(1,T-1):
        for x in range(X):
            for y in range(Y):
                for (x2,y2) in [(x+1,y),(x,y+1)]:
                    if 0 <= x2 < X and 0 <= y2 < Y:
                        model.addConstr(
                            quicksum(agent.select(t, '*', x, y, '*', 'move', x2, y2, '*')) +
                            quicksum(agent.select(t, '*', x2, y2, '*', 'move', x, y, '*'))
                            <= 1
                        )

    # height of flow
    for t in range(T-1):
        for x in range(X):
            for y in range(Y):
                for z in range(Z):
                    model.addConstr(
                        quicksum(height.select(t, x, y, z, '*'))
                        >=
                        quicksum(agent.select(t, '*', x, y, z, '*', '*', '*', '*'))
                    )

    # height decrease
    for t in range(T-1):
        for x in range(X):
            for y in range(Y):
                for z in range(Z-1):
                    model.addConstr(
                        height[t, x, y, z + 1, z]
                        ==
                        quicksum(agent.select(t, 0, '*', '*', '*', 'pickup', x, y, z))
                    )

    # height increase
    for t in range(T-1):
        for x in range(X):
            for y in range(Y):
                for z in range(Z-1):
                    model.addConstr(
                        height[t,x,y,z,z+1]
                        ==
                        quicksum(agent.select(t,1,'*','*','*','delivery',x,y,z))
                    )

    # Minimum number of agents
    model.addConstr(
        quicksum(agent.select('*','*','start','start','start','move','*','*','*'))
        >=
        sum(structure.values())
    )

    # Start at the first time step
    model.addConstr(
        quicksum(agent.select(0,'*','start','start','start','move','*','*','*')) >= 1
    )

    # Input warm start solution
    if len(sol_paths) > 0:

        # Set up warm start value for height variables
        for var in height.values():
            var.Start = 0
        for (t, x, y) in sol_height:
            if t < T - 1:
                z = sol_height[t, x, y]
                z2 = sol_height[t + 1, x, y]
                height[t, x, y, z, z2].Start = 1

        # Set up warm start value for agent variables
        for var in agent.values():
            var.Start = 0
        for (a, path) in enumerate(sol_paths):
            for (t, (c, x, y, z)) in enumerate(path):
                if x != '-':
                    if 0 <= t - 1 and t - 1 < T - 3 and path[t - 1][1] == '-' and path[t + 1][1] != '-':
                        if (a, t) in sol_pickup:
                            c = 0
                        elif (a, t) in sol_delivery:
                            c = 1
                        agent[t - 1, c, 'start', 'start', 'start', 'move', x, y, z].Start = 1

                    if (a, t) in sol_pickup:
                        (x2, y2) = sol_pickup[a, t]
                        z2 = z
                        agent[t, 0, x, y, z, 'pickup', x2, y2, z2].Start = 1
                    elif (a, t) in sol_delivery:
                        (x2, y2) = sol_delivery[a, t]
                        z2 = z
                        agent[t, 1, x, y, z, 'delivery', x2, y2, z2].Start = 1
                    elif t < len(path) - 1 and path[t + 1][1] != '-':
                        (_, x2, y2, z2) = path[t + 1]
                        agent[t, c, x, y, z, 'move', x2, y2, z2].Start = 1
                    elif 0 < t and t < len(path) - 1 and path[t - 1][1] != '-' and path[t + 1][1] == '-':
                        agent[t, c, x, y, z, 'move', 'end', 'end', 'end'].Start = 1

    # Solve
    model.optimize()
    print('')

    # Delete log file
    if os._exists('gurobi.log'):
        os.remove('gurobi.log')

    # Get statistics
    status = model.Status
    run_time = model.Runtime
    sol_count = model.SolCount

    # Process output
    if status == GRB.Status.INFEASIBLE:
        return ('Infeasible', run_time, -1)

    elif sol_count == 0:
        # print('Did not find any feasible solution')
        lb = int(ceil(max(0.0, model.ObjBound)))
        return ('Unknown', run_time, lb)

    else:
        # Get statistics
        lb = int(ceil(max(0.0, model.ObjBound)))
        ub = int(model.ObjVal)

        # Get heights
        sol_height = {
            (t+1,x,y): [z2 for z2 in range(Z) for var in height.select(t,x,y,'*',z2) if var.X > 1e-4]
            for t in range(T-1) for x in range(X) for y in range(Y)
        }
        for ((t,x,y),zs) in sol_height.items():
            assert len(zs) == 1
        sol_height = {key: val[0] for (key,val) in sol_height.items()}
        for x in range(X):
            for y in range(Y):
                sol_height[0,x,y] = 0

        # Get paths.
        N = int(sum(var.X for var in agent.select('*','*','start','start','start','move','*','*','*')))
        sol_paths = [[('-','-','-','-')] * T for _ in range(N)]
        a = 0
        for ((t,c,x,y,z,action,x2,y2,z2),var) in agent.items():
            if var.X > 1e-4 and x == 'start':
                sol_paths[a][t+1] = (c,x2,y2,z2)
                a += 1
        for a in range(N):
            t = min(t for (t,(c,x,y,z)) in enumerate(sol_paths[a]) if x != '-')
            (c,x,y,z) = sol_paths[a][t]

            while True:
                next = []

                # Find next move.
                for (x2,y2) in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]:
                    for z2 in [z-1,z,z+1]:
                        if 0 <= x2 and x2 < X and 0 <= y2 and y2 < Y and 0 <= z2 and z2 < Z:
                            if (t,c,x,y,z,'move',x2,y2,z2) in agent and agent[t,c,x,y,z,'move',x2,y2,z2].X > 1e-4:
                                next.append((c,x2,y2,z2))
                (x2,y2,z2) = (x,y,z)
                if (t,c,x,y,z,'move',x2,y2,z2) in agent and agent[t,c,x,y,z,'move',x2,y2,z2].X > 1e-4:
                    next.append((c,x2,y2,z2))
                (x2,y2,z2) = ('end','end','end')
                if (t,c,x,y,z,'move',x2,y2,z2) in agent and agent[t,c,x,y,z,'move',x2,y2,z2].X > 1e-4:
                    next.append((c,x2,y2,z2))

                # Find next pickup or delivery.
                z2 = z
                if c == 0:
                    for (x2,y2) in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]:
                        if 0 <= x2 and x2 < X and 0 <= y2 and y2 < Y:
                            if (t,c,x,y,z,'pickup',x2,y2,z2) in agent and agent[t,c,x,y,z,'pickup',x2,y2,z2].X > 1e-4:
                                next.append((1,x,y,z))
                else:
                    for (x2,y2) in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]:
                        if 0 <= x2 and x2 < X and 0 <= y2 and y2 < Y:
                            if (t,c,x,y,z,'delivery',x2,y2,z2) in agent and agent[t,c,x,y,z,'delivery',x2,y2,z2].X > 1e-4:
                                next.append((0,x,y,z))

                # Stop if reached the end.
                assert len(next) == 1
                (c,x,y,z) = next[0]
                if x == 'end':
                    break

                # Store the action.
                t += 1
                sol_paths[a][t] = (c,x,y,z)

        # Get pickups and deliveries.
        sol_pickup = {}
        sol_delivery = {}
        for a in range(N):
            for (t,(c,x,y,z)) in enumerate(sol_paths[a]):
                if x != '-' and t < T-2:
                    if z < Z-1:
                        z2 = z
                        if c == 0:
                            for (x2,y2) in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]:
                                if 0 <= x2 and x2 < X and 0 <= y2 and y2 < Y:
                                    if agent[t,c,x,y,z,'pickup',x2,y2,z2].X > 1e-4:
                                        assert (a,t) not in sol_pickup
                                        sol_pickup[a,t] = (x2,y2)
                        elif c == 1:
                            for (x2,y2) in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]:
                                if 0 <= x2 and x2 < X and 0 <= y2 and y2 < Y:
                                    if agent[t,c,x,y,z,'delivery',x2,y2,z2].X > 1e-4:
                                        assert (a,t) not in sol_delivery
                                        sol_delivery[a,t] = (x2,y2)

        # Return
        if status == GRB.Status.TIME_LIMIT or status == GRB.Status.INTERRUPTED:
            return ('Feasible', run_time, lb, ub, sol_paths, sol_pickup, sol_delivery, sol_height)
        elif status == GRB.Status.OPTIMAL:
            return ('Optimal', run_time, lb, ub, sol_paths, sol_pickup, sol_delivery, sol_height)
        else:
            print(f'Invalid Gurobi status {status}')
            exit()


if __name__ == "__main__":
    # leverage as many cores as possible for parallel processing
    threads = multiprocessing.cpu_count()
    print(f'Using {threads} threads')

    # create a toy instance first with all parameters hardcoded
    # TODO: interface with our 3D GridWorld environment to create test instance

    # set the upper bound for number of agents
    max_agents = 2

    # set the length and width of the grid
    X = 7
    Y = 7

    # initialize the desired structure, which is specified by a LIST of
    # (x, y, z) coordinates where (x, y) is the position of the block
    # looking from the top and z is the height of the structure at that position
    # (1, 2, 3) means a block at (1, 2) with height 3.
    # NOTE: hollow structure is not supported based on structure representation
    structure = dict()
    structure[(3, 3)] = 1
    structure[(3, 4)] = 1

    # Z = max(all heights in the structure) + 1 is the height of the GridWorld
    # You can set this to a larger value if you want to but this will increase
    # the size of the MILP problem.
    Z = max(structure.values()) + 1

    # initialize parameters for the MILP solver
    status = 'Unknown'
    run_time = 0.0
    sum_of_costs = -1
    sol_paths = {}
    sol_pickup = {}
    sol_delivery = {}
    sol_height = {}
    time_limit = 60.0

    # prioritize the makespan over the sum of costs
    for makespan in range(3, 10000):

        # solve the MILP problem
        iter_T = makespan + 1
        output = MILP(max_agents, iter_T, X, Y, Z, structure, time_limit, threads,
                      sol_height, sol_paths, sol_pickup, sol_delivery)
        (status, iter_run_time) = output[:2]
        run_time += iter_run_time
        time_limit -= iter_run_time

        if status in ['Feasible', 'Optimal']:

            T = iter_T
            (status, run_time, lb, ub, sol_paths, sol_pickup, sol_delivery, sol_height) = output
            sum_of_costs = sum(1 for path in sol_paths for (_, x, _, _) in path if x != '-')
            if sum_of_costs != ub:
                print(f'Error: calculated sum-of-costs {sum_of_costs} and objective value {ub} mismatch')
            break

        elif time_limit < 0:
            # Done
            break

print('====================================================================================================')
print(f'Status: {status}')
print(f'Run time: {run_time:.2f} seconds')
if status != 'Infeasible' and status != 'Unknown':
    print(f'Makespan: {makespan}')
    print(f'Sum-of-costs: {sum_of_costs}')
    print(f'Paths: {len(sol_paths)}')

    T = len(sol_paths[0])
    for path in sol_paths:
        assert len(path) == T
    agents = 0
    for t in range(T):
        count = 0;
        for path in sol_paths:
            count += (path[t][0] != '-')
        if t > 0:
            for path in sol_paths:
                count += (path[t-1][0] != '-' and path[t][0] == '-')
        if count > agents:
            agents = count
    print(f'Agents: {agents}')