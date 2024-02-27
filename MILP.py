# -------------------------------------------------------------------------------------
# Implementation of the MILP method discussed in the paper:
# Exact Approaches to the Multi-agent Collective Construction Problem by E.Lam, et al.
# https://link.springer.com/chapter/10.1007/978-3-030-58475-7_43

# Adapted from the original code provided by E. Lam.
# -------------------------------------------------------------------------------------

import multiprocessing
from gurobipy import *
from math import ceil


def MILP(max_agents, T, X, Y, Z, structure, objective, time_limit, threads, sol_height, sol_paths, sol_pickup, sol_delivery):

    # initialize the model
    model = Model('MILP')

    # define model parameters
    model.Params.Method = 3     # concurrent optimization
    model.Params.Threads = threads
    model.Params.TimeLimit = time_limit

    # create borders
    borders = set()
    borders.add([x, 0, 0] for x in range(X))
    borders.add([x, Y-1, 0] for x in range(X))
    borders.add([0, y, 0] for y in range(1, Y-1))
    borders.add([X-1, y, 0] for y in range(1, Y-1))

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
        output = MILP(max_agents, iter_T, X, Y, Z, structure, 'Sum-of-costs', time_limit, threads,
                      sol_height, sol_paths, sol_pickup, sol_delivery)
        (status, iter_run_time) = output[:2]
        run_time += iter_run_time
        time_limit -= iter_run_time
