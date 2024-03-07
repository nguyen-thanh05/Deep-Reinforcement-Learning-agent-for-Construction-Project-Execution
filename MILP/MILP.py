import multiprocessing
from gurobipy import *
from math import ceil
from termcolor import colored
import matplotlib.pyplot as plt
import numpy as np


def solve_mip(instance_name, A, T, X, Y, Z, structure, objective, time_limit, threads,
              sol_height, sol_paths, sol_pickup, sol_delivery):

    # Create model
    model = Model()

    # Set parameters
    model.Params.Method = 3    # Non-deterministic concurrent
    model.Params.Threads = threads
    model.Params.TimeLimit = time_limit

    # Create list of border vertices.
    border = set(
        [(x2, 0, 0)   for x2 in range(X)] +
        [(x2, Y-1, 0) for x2 in range(X)] +
        [(0, y2, 0)   for y2 in range(1, Y-1)] +
        [(X-1, y2, 0) for y2 in range(1, Y-1)]
    )

    # Agent move variables - move to the border from the start vertex
    agent = tupledict()
    x = y = z = 'start'
    action = 'move'
    for t in range(T-3):
        for c in range(2):
            for (x2, y2, z2) in border:
                assert (t, c, x, y, z, action, x2, y2, z2) not in agent
                agent[t, c, x, y, z, action, x2, y2, z2] = model.addVar(
                    vtype=GRB.BINARY,
                    name=f'agent({t}, {c}, {x}, {y}, {z}, {action}, {x2}, {y2}, {z2})'
                )

    # Agent move variables - intermediate time steps;
    action = 'move'
    for t in range(1, T-2):
        for c in range(2):
            for x in range(X):
                for y in range(Y):
                    for z in range(Z):

                        # Move to neighbor cell
                        for (x2, y2) in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
                            for z2 in [z-1, z, z+1]:
                                if 0 <= x2 and x2 < X and 0 <= y2 and y2 < Y and 0 <= z2 and z2 < Z:
                                    assert (t, c, x, y, z, action, x2, y2, z2) not in agent
                                    agent[t, c, x, y, z, action, x2, y2, z2] = model.addVar(
                                        vtype=GRB.BINARY,
                                        name=f'agent({t}, {c}, {x}, {y}, {z}, {action}, {x2}, {y2}, {z2})'
                                    )

                        # Wait
                        (x2, y2, z2) = (x, y, z)
                        assert (t, c, x, y, z, action, x2, y2, z2) not in agent
                        agent[t, c, x, y, z, action, x2, y2, z2] = model.addVar(
                            vtype=GRB.BINARY,
                            name=f'agent({t}, {c}, {x}, {y}, {z}, {action}, {x2}, {y2}, {z2})'
                        )

    # Agent move variables - move from the border to the end vertex
    x2 = y2 = z2 = 'end'
    action = 'move'
    for t in range(2, T-1):
        for c in range(2):
            for (x, y, z) in border:
                assert (t, c, x, y, z, action, x2, y2, z2) not in agent
                agent[t, c, x, y, z, action, x2, y2, z2] = model.addVar(
                    vtype=GRB.BINARY,
                    name=f'agent({t}, {c}, {x}, {y}, {z}, {action}, {x2}, {y2}, {z2})'
                )

    # Agent pickup variables
    action = 'pickup'
    for t in range(1, T-2):
        c = 0
        for x in range(X):
            for y in range(Y):
                for z in range(Z-1):
                    for (x2, y2) in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
                        if 0 <= x2 and x2 < X and 0 <= y2 and y2 < Y:
                            z2 = z
                            assert (t, c, x, y, z, action, x2, y2, z2) not in agent
                            agent[t, c, x, y, z, action, x2, y2, z2] = model.addVar(
                                vtype=GRB.BINARY,
                                name=f'agent({t}, {c}, {x}, {y}, {z}, {action}, {x2}, {y2}, {z2})'
                            )

    # Agent delivery variables
    action = 'delivery'
    for t in range(1, T-2):
        c = 1
        for x in range(X):
            for y in range(Y):
                for z in range(Z-1):
                    for (x2, y2) in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
                        if 0 <= x2 < X and 0 <= y2 < Y:
                            z2 = z
                            assert (t, c, x, y, z, action, x2, y2, z2) not in agent
                            agent[t, c, x, y, z, action, x2, y2, z2] = model.addVar(
                                vtype=GRB.BINARY,
                                name=f'agent({t}, {c}, {x}, {y}, {z}, {action}, {x2}, {y2}, {z2})'
                            )

    # Variables indicating the height of the structures
    height = tupledict()
    for t in range(T-1):
        for x in range(X):
            for y in range(Y):
                for z in range(Z):
                    for z2 in [z-1, z, z+1]:
                        if 0 <= z2 < Z:
                            height[t, x, y, z, z2] = model.addVar(
                                vtype=GRB.BINARY,
                                name=f'height({t}, {x}, {y}, {z}, {z2})'
                            )

    # Update
    model.update()

    # ------------------------------------------------------------------------------------

    # Set objective function
    if objective == 'Makespan':

        # Create variables indicating if a time step is used
        time_step_used = {t: model.addVar(vtype=GRB.BINARY, obj=1, name=f'time_step_used({t})') for t in range(1, T-1)}
        for t in range(1, T-2):
            model.addConstr(time_step_used[t+1] <= time_step_used[t])

        # Create linking constraints
        for ((t, c, x, y, z, action, x2, y2, z2), var) in agent.items():
            if t >= 1 and x != 'start':
                model.addConstr(var <= time_step_used[t])

        # Add a constant 1 to the objective function.
        model.ObjCon = 1

    elif objective == 'Sum-of-costs':

        # Add cost to every action
        for ((t, c, x, y, z, action, x2, y2, z2), var) in agent.items():
            if x != 'start':
                var.Obj = 1

    else:
        print('Invalid objective function')
        exit()

    # ------------------------------------------------------------------------------------

    # Height is 0 at the border
    for t in range(T-1):
        for (x, y, z) in border:
            model.addConstr(
                height[t, x, y, z, z] == 1,
                name=f'height_border({t}, {x}, {y})'
            )

    # Height is 0 at the first two time steps
    t = 0
    for x in range(X):
        for y in range(Y):
            model.addConstr(
                height[t, x, y, 0, 0] == 1,
                name=f'height_begin({x}, {y})'
            )

    # Height is equal to the building at the last two time steps
    t = T-2
    for x in range(X):
        for y in range(Y):
            z = structure[x, y] if (x, y) in structure else 0
            model.addConstr(
                height[t, x, y, z, z] == 1,
                name=f'height_end({x}, {y})'
            )

    # Exactly one height at each cell and time step
    for t in range(T-1):
        for x in range(X):
            for y in range(Y):
                model.addConstr(
                    quicksum(height.select(t, x, y, '*', '*')) == 1,
                    name=f'height_selection({t}, {x}, {y})'
                )

    # Change in height
    for t in range(T-2):
        for x in range(X):
            for y in range(Y):
                for z in range(Z):
                    model.addConstr(
                        quicksum(height.select(t, x, y, '*', z))
                        ==
                        quicksum(height.select(t+1, x, y, z, '*')),
                        name=f'height_flow({t}, {x}, {y}, {z})'
                    )

    # ------------------------------------------------------------------------------------

    # Maximum number of agents
    if A < float('inf'):
        A = int(A)
        for t in range(T):
            model.addConstr(
                quicksum(var for ((tt, c, x, y, z, action, x2, y2, z2), var) in agent.items() if tt == t) # and x != 'start'
                # Include start here because an agent takes one step to exit then come back into the map
                <= A,
                name=f'max_agents({t})'
            )

    # Flow - not carrying
    for t in range(T-2):
        for x in range(X):
            for y in range(Y):
                for z in range(Z):
                    model.addConstr(
                        quicksum(agent.select(t, 0, '*', '*', '*', 'move', x, y, z)) +
                        quicksum(agent.select(t, 1, x, y, z, 'delivery', '*', '*', '*'))
                        ==
                        quicksum(agent.select(t+1, 0, x, y, z, 'move', '*', '*', '*')) +
                        quicksum(agent.select(t+1, 0, x, y, z, 'pickup', '*', '*', '*')),
                        name=f'flow_not_carrying({t}, {x}, {y}, {z})'
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
                        quicksum(agent.select(t+1, 1, x, y, z, 'delivery', '*', '*', '*')),
                        name=f'flow_carrying({t}, {x}, {y}, {z})'
                    )

    # Vertex collision
    for t in range(1, T-1):
        for x in range(X):
            for y in range(Y):
                model.addConstr(
                    quicksum(agent.select(t, '*', x, y, '*', '*', '*', '*', '*')) +
                    quicksum(agent.select(t, '*', '*', '*', '*', 'pickup', x, y, '*')) +
                    quicksum(agent.select(t, '*', '*', '*', '*', 'delivery', x, y, '*'))
                    <= 1,
                    name=f'vertex_collision({t}, {x}, {y})'
                )

    # Edge collision
    for t in range(1, T-1):
        for x in range(X):
            for y in range(Y):
                for (x2, y2) in [(x+1, y), (x, y+1)]:
                    if 0 <= x2 < X and 0 <= y2 < Y:
                        model.addConstr(
                            quicksum(agent.select(t, '*', x, y, '*', 'move', x2, y2, '*')) +
                            quicksum(agent.select(t, '*', x2, y2, '*', 'move', x, y, '*'))
                            <= 1,
                            name=f'edge_collision({t}, {x}, {y}, {x2}, {y2})'
                        )

    # ------------------------------------------------------------------------------------

    # Height of flow
    for t in range(T-1):
        for x in range(X):
            for y in range(Y):
                for z in range(Z):
                    model.addConstr(
                        quicksum(height.select(t, x, y, z, '*'))
                        >=
                        quicksum(agent.select(t, '*', x, y, z, '*', '*', '*', '*')),
                        name=f'flow_height({t}, {x}, {y}, {z})'
                    )

    # Height decrease
    for t in range(T-1):
        for x in range(X):
            for y in range(Y):
                for z in range(Z-1):
                    model.addConstr(
                        height[t, x, y, z+1, z]
                        ==
                        quicksum(agent.select(t, 0, '*', '*', '*', 'pickup', x, y, z)),
                        name=f'height_decrease({t}, {x}, {y}, {z})'
                    )

    # Height increase
    for t in range(T-1):
        for x in range(X):
            for y in range(Y):
                for z in range(Z-1):
                    model.addConstr(
                        height[t, x, y, z, z+1]
                        ==
                        quicksum(agent.select(t, 1, '*', '*', '*', 'delivery', x, y, z)),
                        name=f'height_increase({t}, {x}, {y}, {z})'
                    )

    # ------------------------------------------------------------------------------------

    # Minimum number of agents
    model.addConstr(
        quicksum(agent.select('*', '*', 'start', 'start', 'start', 'move', '*', '*', '*'))
        >=
        sum(structure.values()),
        name=f'min_agents'
    )

    # Start at the first time step
    model.addConstr(
        quicksum(agent.select(0, '*', 'start', 'start', 'start', 'move', '*', '*', '*')) >= 1,
        name=f'first_time_step'
    )

    # ------------------------------------------------------------------------------------

    # Input warm start solution
    if len(sol_paths) > 0:

        # Set up warm start value for height variables
        for var in height.values():
            var.Start = 0
        for (t, x, y) in sol_height:
            if t < T-1:
                z = sol_height[t, x, y]
                z2 = sol_height[t+1, x, y]
                height[t, x, y, z, z2].Start = 1

        # Set up warm start value for agent variables
        for var in agent.values():
            var.Start = 0
        for (a, path) in enumerate(sol_paths):
            for (t, (c, x, y, z)) in enumerate(path):
                if x != '-':
                    if 0 <= t-1 < T-3 and path[t - 1][1] == '-' and path[t + 1][1] != '-':
                        if (a, t) in sol_pickup:
                            c = 0
                        elif (a, t) in sol_delivery:
                            c = 1
                        agent[t-1, c, 'start', 'start', 'start', 'move', x, y, z].Start = 1

                    if (a, t) in sol_pickup:
                        (x2, y2) = sol_pickup[a, t]
                        z2 = z
                        agent[t, 0, x, y, z, 'pickup', x2, y2, z2].Start = 1
                    elif (a, t) in sol_delivery:
                        (x2, y2) = sol_delivery[a, t]
                        z2 = z
                        agent[t, 1, x, y, z, 'delivery', x2, y2, z2].Start = 1
                    elif t < len(path)-1 and path[t+1][1] != '-':
                        (_, x2, y2, z2) = path[t+1]
                        agent[t, c, x, y, z, 'move', x2, y2, z2].Start = 1
                    elif 0 < t and t < len(path)-1 and path[t-1][1] != '-' and path[t+1][1] == '-':
                        agent[t, c, x, y, z, 'move', 'end', 'end', 'end'].Start = 1

    # ------------------------------------------------------------------------------------

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
        return 'Infeasible', run_time, -1

    elif sol_count == 0:
        # print('Did not find any feasible solution')
        lb = int(ceil(max(0.0, model.ObjBound)))
        return 'Unknown', run_time, lb

    else:
        # Get statistics
        lb = int(ceil(max(0.0, model.ObjBound)))
        ub = int(model.ObjVal)

        # Get heights
        sol_height = {
            (t+1, x, y): [z2 for z2 in range(Z) for var in height.select(t, x, y, '*', z2) if var.X > 1e-4]
            for t in range(T-1) for x in range(X) for y in range(Y)
        }
        for ((t, x, y), zs) in sol_height.items():
            assert len(zs) == 1
        sol_height = {key: val[0] for (key, val) in sol_height.items()}
        for x in range(X):
            for y in range(Y):
                sol_height[0, x, y] = 0

        # Get paths.
        N = int(sum(var.X for var in agent.select('*', '*', 'start', 'start', 'start', 'move', '*', '*', '*')))
        sol_paths = [[('-', '-', '-', '-')] * T for _ in range(N)]
        a = 0
        for ((t, c, x, y, z, action, x2, y2, z2), var) in agent.items():
            if var.X > 1e-4 and x == 'start':
                sol_paths[a][t+1] = (c, x2, y2, z2)
                a += 1
        for a in range(N):
            t = min(t for (t, (c, x, y, z)) in enumerate(sol_paths[a]) if x != '-')
            (c, x, y, z) = sol_paths[a][t]

            while True:
                next = []

                # Find next move.
                for (x2, y2) in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
                    for z2 in [z-1, z, z+1]:
                        if 0 <= x2 < X and 0 <= y2 < Y and 0 <= z2 < Z:
                            if (t, c, x, y, z, 'move', x2, y2, z2) in agent and agent[t, c, x, y, z, 'move', x2, y2, z2].X > 1e-4:
                                next.append((c, x2, y2, z2))
                (x2, y2, z2) = (x, y, z)
                if (t, c, x, y, z, 'move', x2, y2, z2) in agent and agent[t, c, x, y, z, 'move', x2, y2, z2].X > 1e-4:
                    next.append((c, x2, y2, z2))
                (x2, y2, z2) = ('end', 'end', 'end')
                if (t, c, x, y, z, 'move', x2, y2, z2) in agent and agent[t, c, x, y, z, 'move', x2, y2, z2].X > 1e-4:
                    next.append((c, x2, y2, z2))

                # Find next pickup or delivery.
                z2 = z
                if c == 0:
                    for (x2, y2) in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
                        if 0 <= x2 < X and 0 <= y2 < Y:
                            if (t, c, x, y, z, 'pickup', x2, y2, z2) in agent and agent[t, c, x, y, z, 'pickup', x2, y2, z2].X > 1e-4:
                                next.append((1, x, y, z))
                else:
                    for (x2, y2) in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
                        if 0 <= x2 < X and 0 <= y2 < Y:
                            if (t, c, x, y, z, 'delivery', x2, y2, z2) in agent and agent[t, c, x, y, z, 'delivery', x2, y2, z2].X > 1e-4:
                                next.append((0, x, y, z))

                # Stop if reached the end.
                assert len(next) == 1
                (c, x, y, z) = next[0]
                if x == 'end':
                    break

                # Store the action.
                t += 1
                sol_paths[a][t] = (c, x, y, z)

        # Get pickups and deliveries.
        sol_pickup = {}
        sol_delivery = {}
        for a in range(N):
            for (t, (c, x, y, z)) in enumerate(sol_paths[a]):
                if x != '-' and t < T-2:
                    if z < Z-1:
                        z2 = z
                        if c == 0:
                            for (x2, y2) in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
                                if 0 <= x2 < X and 0 <= y2 < Y:
                                    if agent[t, c, x, y, z, 'pickup', x2, y2, z2].X > 1e-4:
                                        assert (a, t) not in sol_pickup
                                        sol_pickup[a, t] = (x2, y2)
                        elif c == 1:
                            for (x2, y2) in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
                                if 0 <= x2 < X and 0 <= y2 < Y:
                                    if agent[t, c, x, y, z, 'delivery', x2, y2, z2].X > 1e-4:
                                        assert (a, t) not in sol_delivery
                                        sol_delivery[a, t] = (x2, y2)

        # Return
        if status == GRB.Status.TIME_LIMIT or status == GRB.Status.INTERRUPTED:
            return 'Feasible', run_time, lb, ub, sol_paths, sol_pickup, sol_delivery, sol_height
        elif status == GRB.Status.OPTIMAL:
            return 'Optimal', run_time, lb, ub, sol_paths, sol_pickup, sol_delivery, sol_height
        else:
            print(f'Invalid Gurobi status {status}')
            exit()


def print_instance(instance_name, A, X, Y, Z):
    print(f'Instance: {instance_name}')
    print(f'A: {A if A < float("inf") else "unlimited"}')
    print(f'X: {X}')
    print(f'Y: {Y}')
    print(f'Z: {Z}')
    print('')

# ----------------------------------------------------------------------------------------

# Print paths
def print_paths(T, sol_paths, sol_pickup, sol_delivery):
    print('Paths:')

    print(f'{"":9s}', end='')
    for t in range(T):
        print(f'{t:>18d}', end='')
    print('')

    for a in range(len(sol_paths)):
        print(f'Path {a:2d}: ', end='')
        for (t,(c,x,y,z)) in enumerate(sol_paths[a]):
            if x == '-':
                print(f'{"":>18s}', end='')
            elif (a,t) in sol_delivery:
                print(f"{f'({x},{y},{z})':>12s}", end='')
                (x,y) = sol_delivery[a,t]
                print(colored(f"{f'({x},{y})':>6s}", 'blue'), end='')
            elif (a,t) in sol_pickup:
                print(f"{f'({x},{y},{z})':>12s}", end='')
                (x,y) = sol_pickup[a,t]
                print(colored(f"{f'({x},{y})':>6s}", 'red'), end='')
            else:
                print(f"{f'({x},{y},{z})':>18s}", end='')
        print('')
    print('')

# ----------------------------------------------------------------------------------------

# Print status
def print_status(status, run_time, makespan, sum_of_costs, sol_paths):
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
            count = 0
            for path in sol_paths:
                count += (path[t][0] != '-')
            if t > 0:
                for path in sol_paths:
                    count += (path[t-1][0] != '-' and path[t][0] == '-')
            if count > agents:
                agents = count
        print(f'Agents: {agents}')


def read_instance(instance_path):
    instance_name = ''
    A = -1
    X = -1
    Y = -1
    structure = {}

    f = open(instance_path, 'r')
    lines = f.read().split('\n')
    for line in lines:

        match = re.match('\s*Name:\s*(\w+)\s*', line)
        if match:
            instance_name = match.groups()[0].strip()

        match = re.match('\s*A:\s*(\d+)\s*', line)
        if match:
            A = int(match.groups()[0])

        match = re.match('\s*X:\s*(\d+)\s*', line)
        if match:
            X = int(match.groups()[0])

        match = re.match('\s*Y:\s*(\d+)\s*', line)
        if match:
            Y = int(match.groups()[0])

    for (i,line) in enumerate(lines):
        match = re.match('\s*Structure:\s*', line)
        if match:
            for j in range(i+1,len(lines)):
                line = lines[j]
                match = re.match('\s*(\d+)\s*(\d+)\s*(\d+)\s*', line)
                if match:
                    (x,y,z) = map(int, match.groups())
                    structure[x,y] = z

    if instance_name == '':
        print(f'Missing "Name" parameter in instance file {instance_path}')
        exit(1)

    if A == -1:
        print(f'Missing "A" parameter in instance file {instance_path}')
        exit(1)

    if X == -1:
        print(f'Missing "X" parameter in instance file {instance_path}')
        exit(1)

    if Y == -1:
        print(f'Missing "Y" parameter in instance file {instance_path}')
        exit(1)

    if len(structure) == 0:
        print(f'Missing "Structure" parameter in instance file {instance_path}')
        exit(1)

    return (instance_name, A, X, Y, structure)


def render(sol_paths, sol_pickup, sol_delivery, X, Y, Z, T, relative_path):
    cubes = set()

    for t in range(T):
        agents = set()
        for p in range(len(sol_paths)):
            cur = sol_paths[p][t]
            removed = set()
            added = set()
            if cur[0] != '-':
                agents.add(cur[1:])
            prev = sol_paths[p][t - 1]

            if prev == cur:
                continue

            # if the agent has exited the grid
            if cur[0] == '-':
                if prev[0] != '-':
                    removed.add((prev[1], prev[2], prev[3]))
                    if prev[0] == 1:
                        removed.add((prev[1], prev[2], prev[3] + 1))
            elif cur[0] == 0:
                if prev[0] != '-':
                    if prev[0] == 0:
                        removed.add((prev[1], prev[2], prev[3]))
                    else:
                        removed.add((prev[1], prev[2], prev[3] + 1))
                        # find where the block is delivered
                        if (p, t - 1) in sol_delivery:
                            delivered = sol_delivery[(p, t - 1)]
                            added.add((delivered[0], delivered[1], prev[3]))
                added.add((cur[1], cur[2], cur[3]))
            else:  # cur[0] == 1
                if prev[0] != '-':
                    # find where the block is picked up
                    if prev[0] == 0:
                        if (p, t - 1) in sol_pickup:
                            picked = sol_pickup[(p, t - 1)]
                            removed.add((picked[0], picked[1], prev[3]))
                    else:
                        removed.add((prev[1], prev[2], prev[3]))
                        removed.add((prev[1], prev[2], prev[3] + 1))
                added.add((cur[1], cur[2], cur[3]))
                added.add((cur[1], cur[2], cur[3] + 1))

            for r in removed:
                cubes.discard(r)
            for a in added:
                cubes.add(a)

        print("Time step: ", t)
        print("Cubes' coordinates: ")
        print(cubes)
        print("Agents' coordinates: ")
        print(agents)
        print()

        # separate cubes into blocks and agents
        blocks = set()
        for cube in cubes:
            if cube not in agents:
                blocks.add(cube)

        blocks_array = np.zeros((X, Y, Z), dtype=bool)
        for block in blocks:
            blocks_array[block] = True
        colors = np.empty(blocks_array.shape, dtype=object)
        colors[blocks_array] = '#7A88CCC0'

        agents_array = np.zeros((X, Y, Z), dtype=bool)
        for agent in agents:
            agents_array[agent] = True
        colors[agents_array] = '#FFD65DC0'

        voxels = blocks_array | agents_array

        ax = plt.figure().add_subplot(projection='3d')
        ax.voxels(voxels, facecolors=colors, edgecolor='k')
        plt.savefig(f'{relative_path}/timestep_{t}.png')
        plt.show()

        # sleep for 1 second
        plt.pause(1)


def main():
    instance_path = "Input/instance_4.txt"
    (instance_name, A, X, Y, structure) = read_instance(instance_path)

    time_limit = 469469

    threads = multiprocessing.cpu_count()

    # Calculate maximum height
    Z = max(structure.values()) + 1

    # Print instance
    print_instance(instance_name, A, X, Y, Z)

    # Initialise
    status = 'Unknown'
    run_time = 0.0
    sum_of_costs = -1
    sol_paths = {}
    sol_pickup = {}
    sol_delivery = {}
    sol_height = {}

    # Minimise the makespan
    for makespan in range(3, 10000):

        # Solve
        iter_T = makespan + 1
        print('====================================================================================================')
        print(f'Attempting T = {iter_T}')
        print('')
        output = solve_mip(instance_name, A, iter_T, X, Y, Z, structure, 'Sum-of-costs', time_limit, threads,
                       sol_height, sol_paths, sol_pickup, sol_delivery)
        (status, iter_run_time) = output[:2]
        run_time += iter_run_time
        time_limit -= iter_run_time

        # Store the objective value
        if status == 'Feasible' or status == 'Optimal':

            # Get solution
            T = iter_T
            (status, run_time, lb, ub, sol_paths, sol_pickup, sol_delivery, sol_height) = output
            sum_of_costs = sum(1 for path in sol_paths for (_, x, _, _) in path if x != '-')
            if sum_of_costs != ub:
                print(f'Error: calculated sum-of-costs {sum_of_costs} and objective value {ub} mismatch')

            # Output
            print(
                '====================================================================================================')
            print_paths(T, sol_paths, sol_pickup, sol_delivery)

            # Done.
            break

        elif time_limit < 0:

            # Done
            break

    # Print status
    print('====================================================================================================')
    # log output to the Output folder, create one if it doesn't exist
    if not os.path.exists('Output'):
        os.makedirs('Output')
    # create a sub-folder inside Output with the instance name
    if not os.path.exists(f'Output/{instance_name}'):
        os.makedirs(f'Output/{instance_name}')

    print_status(status, run_time, makespan, sum_of_costs, sol_paths)
    render(sol_paths, sol_pickup, sol_delivery, X, Y, X, T, relative_path=f'Output/{instance_name}')

    print("CONSTRUCTION COMPLETED")


if __name__ == '__main__':
    main()

# TODO: Add comprehensive comments
