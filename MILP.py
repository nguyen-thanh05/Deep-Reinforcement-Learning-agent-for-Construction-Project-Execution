# -------------------------------------------------------------------------------------
# Implementation of the MILP method discussed in the paper:
# Exact Approaches to the Multi-agent Collective Construction Problem by E.Lam, et al.
# https://link.springer.com/chapter/10.1007/978-3-030-58475-7_43

# Adapted from the original code provided by E. Lam.
# -------------------------------------------------------------------------------------

import multiprocessing


def MILP():
    pass


if __name__ == "__main__":
    # leverage as many cores as possible for parallel processing
    threads = multiprocessing.cpu_count()
    print(f'Using {threads} threads')

    # create a toy instance first with all parameters hardcoded
    # TODO: interface with our 3D GridWorld environment to create test instance

    # set the upper bound for number of agents
    maxAgents = 2

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

    MILP()
