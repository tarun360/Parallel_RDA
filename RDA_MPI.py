"""
PARALLEL CODE USING MPI

CODE WRITTEN BY:
TARUN GUPTA
IIT INDORE
cse180001059

SOURCES USED:

Lawrence V. Snyder, Mark S. Daskin, A random-key genetic algorithm for the generalized traveling salesman problem,
European Journal of Operational Research, Volume 174, Issue 1, 2006, Pages 38-53, ISSN 0377-2217, https://doi.org/10.1016/j.ejor.2004.09.057.

R. Guha, B. Chatterjee, K. H. Sk, S. Ahmed, T. Bhattacharya, R. Sarkar, “Py_FS: A Python Package for Feature Selection using Meta-heuristic Optimization Algorithms”, accepted for publication in Springer AISC series of 3rd International Conference on Computational Intelligence in Pattern Recognition (CIPR-2021) to be held on 24-25 April, 2021, Kolkata, India.

Fathollahi-Fard, A.M., Hajiaghaei-Keshteli, M. & Tavakkoli-Moghaddam, R. Red deer algorithm (RDA): a new nature-inspired meta-heuristic. Soft Comput 24, 14637–14665 (2020). https://doi.org/10.1007/s00500-020-04812-z

https://mpi4py.readthedocs.io/en/stable/tutorial.html

"""

import numpy as np
np.random.seed(44)

import time
import matplotlib.pyplot as plt
import random, math
import sys
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import seaborn as sns

from utilities import Solution, initialize, sort_agents, cycle_cost, display
from mpi4py import MPI

sns.set()

def RDA(num_agents, max_iter, graph, N_vertices, obj_function, save_conv_graph, alpha, beta, gamma, num_males_frac, UB, LB, myrank, N_PROCS):

    if (myrank == 0):
        # initialize red deers and Leader (the agent with the max fitness)
        deer = initialize(num_agents, N_vertices)
        fitness = np.zeros(num_agents)
        cost = np.zeros(num_agents)
        Leader_agent = np.zeros((1, N_vertices))
        Leader_fitness = float("-inf")
        Leader_cost = float("-inf")

        # initialize convergence curves
        convergence_curve = {}
        convergence_curve['fitness'] = np.zeros(max_iter)
        convergence_curve['feature_count'] = np.zeros(max_iter)

        # create a solution object
        solution = Solution()
        solution.num_agents = num_agents
        solution.max_iter = max_iter
        solution.N_vertices = N_vertices
        solution.obj_function = obj_function

    # start timer
    if(myrank == 0):
        start_time = MPI.Wtime()

    # main loop
    for iter_no in range(max_iter):
        num_males = int(num_males_frac * num_agents)
        num_hinds = num_agents - num_males

        if (myrank == 0):
            deer, fitness = sort_agents(deer, obj_function, graph)
            males = deer[:num_males,:]
            hinds = deer[num_males:,:]
        else:
            males = None
            hinds = None

        assert num_males % N_PROCS == 0

        local_num_males = num_males // N_PROCS
        males_scattered = np.zeros((local_num_males, N_vertices))

        comm.Scatter(males, males_scattered, root=0)
        # roaring of male deer
        for i in range(local_num_males):
            new_male = males_scattered[i].copy()

            sign_arr = np.random.choice([1, -1], size=N_vertices, p=[.5, .5])
            random_arr1 = np.random.rand(N_vertices)
            random_arr2 = np.random.rand(N_vertices)
            new_male += sign_arr*random_arr1*((UB-LB)*random_arr2+LB)

            if obj_function(new_male, graph) < obj_function(males_scattered[i], graph):
                males_scattered[i] = new_male

        comm.Gather(males_scattered, males, root=0)

        # selection of male commanders and stags
        num_coms = int(num_males * gamma) # Eq. (4)
        num_stags = num_males - num_coms # Eq. (5)

        assert num_coms % N_PROCS == 0
        assert num_stags % N_PROCS == 0

        local_num_coms = num_coms // N_PROCS
        local_num_stags = num_stags // N_PROCS

        if (myrank == 0):
            coms = males[:num_coms,:]
            stags = males[num_coms:,:]
        else:
            coms = None
            stags = None

        coms_scattered = np.zeros((local_num_coms,N_vertices))
        stags_scattered = np.zeros((local_num_stags,N_vertices))

        comm.Scatter(coms, coms_scattered, root=0)
        comm.Scatter(stags, stags_scattered, root=0)

        # fight between male commanders and stags
        for i in range(local_num_coms):
            chosen_com = coms_scattered[i].copy()
            chosen_stag = random.choice(stags_scattered)

            random_arr1 = np.random.rand(N_vertices)
            random_arr2 = np.random.rand(N_vertices)

            new_male_1 = (chosen_com + chosen_stag)/2 + random_arr1*( (UB-LB)*random_arr2 + LB )
            new_male_2 = (chosen_com + chosen_stag)/2 - random_arr1*( (UB-LB)*random_arr2 + LB )

            fitness = np.zeros(4)
            fitness[0] = obj_function(chosen_com, graph)
            fitness[1] = obj_function(chosen_stag, graph)
            fitness[2] = obj_function(new_male_1, graph)
            fitness[3] = obj_function(new_male_2, graph)

            bestfit = np.max(fitness)
            if fitness[0] < fitness[1] and fitness[1] == bestfit:
                coms_scattered[i] = chosen_stag.copy()
            elif fitness[0] < fitness[2] and fitness[2] == bestfit:
                coms_scattered[i] = new_male_1.copy()
            elif fitness[0] < fitness[3] and fitness[3] == bestfit:
                coms_scattered[i] = new_male_2.copy()

        comm.Gather(coms_scattered, coms, root=0)
        comm.Gather(stags_scattered, stags, root=0)

        if (myrank == 0):
            # formation of harems
            coms, fitness = sort_agents(coms, obj_function, graph)
            norm = np.linalg.norm(fitness)
            normal_fit = fitness / norm
            total = np.sum(normal_fit)
            power = normal_fit / total # Eq. (9)
            num_harems = np.array([int(x * num_hinds) for x in power]) # Eq.(10)
            num_harems_len = len(num_harems)
            max_harem_size = np.max(num_harems)
            harem = np.empty(shape=(num_coms, max_harem_size, N_vertices))
            harem_shape = harem.shape
            random.shuffle(hinds)
            itr = 0
            for i in range(num_coms):
                harem_size = num_harems[i]
                for j in range(harem_size):
                    harem[i][j] = hinds[itr]
                    itr += 1
        else:
            harem = None
            num_harems = None

        num_harems = comm.bcast(num_harems, root=0)
        harem = comm.bcast(harem, root=0)

        if (myrank == 0):
            # mating of commander with hinds in his harem
            num_harem_mate = [int(x * alpha) for x in num_harems] # Eq. (11)
            population_pool = list(deer)
        else:
            num_harem_mate = None

        num_harem_mate = comm.bcast(num_harem_mate, root=0)
        coms = comm.bcast(coms, root=0)

        population_pool_addition_local = []

        lo = myrank*(num_coms//N_PROCS)
        hi = (myrank+1)*(num_coms//N_PROCS)

        comm.Barrier()
        for i in range(lo, hi):
            random.shuffle(harem[i])
            for j in range(num_harem_mate[i]):
                r = np.random.random() # r is a random number in [0, 1]
                random_arr = np.random.rand(N_vertices)
                offspring = (coms[i]+harem[i][j]) / 2 + (UB-LB)*random_arr

                population_pool_addition_local.append(list(offspring))

                # if number of commanders is greater than 1, inter-harem mating takes place
                if num_coms > 1:
                    # mating of commander with hinds in another harem
                    k = i
                    while k == i:
                        k = random.choice(range(num_coms))

                    num_mate = int(num_harems[k] * beta) # Eq. (13)

                    np.random.shuffle(harem[k])
                    for j in range(num_mate):
                        random_arr = np.random.rand(N_vertices)
                        offspring = (coms[i]+harem[k][j])/2 + (UB-LB)*random_arr

                        population_pool_addition_local.append(list(offspring))

        if(myrank != 0):
            comm.send(population_pool_addition_local, dest=0)
        else:
            population_pool_addition_local_arr = []
            population_pool_addition_local_arr.append(population_pool_addition_local)
            for i in range(1,N_PROCS):
                population_pool_addition_local_arr.append(comm.recv(source = i))

        if(myrank == 0):
            for i in range(len(population_pool_addition_local_arr)):
                for j in range(len(population_pool_addition_local_arr[i])):
                    population_pool.append(population_pool_addition_local_arr[i][j])

        comm.Barrier()
        # mating of stag with nearest hind
        assert num_hinds % N_PROCS == 0

        local_num_hinds = num_hinds // N_PROCS
        hinds_scattered = np.zeros((local_num_hinds, N_vertices))

        comm.Scatter(hinds, hinds_scattered, root=0)
        comm.Scatter(stags, stags_scattered, root=0)

        if( myrank == 0):
            # list(stags) is used only to define the size of population_pool_addition
            # converting list to numpy array as Gather works only with numpy array and not list
            population_pool_addition = np.array(list(stags))
        else:
            population_pool_addition = None

        population_pool_addition_local = []

        for stag in stags_scattered:
            dist = np.zeros(local_num_hinds)
            for i in range(local_num_hinds):
                dist[i] = np.sqrt(np.sum((stag-hinds_scattered[i])*(stag-hinds_scattered[i])))
            min_dist = np.min(dist)
            for i in range(local_num_hinds):
                distance = np.sqrt(np.sum((stag-hinds_scattered[i])*(stag-hinds_scattered[i]))) # Eq. (14)
                if(distance == min_dist):
                    random_arr = np.random.rand(N_vertices)
                    offspring = (stag + hinds_scattered[i])/2 + (UB - LB) * random_arr
                    population_pool_addition_local.append(list(offspring))
                    break

        comm.Gather(hinds_scattered, hinds, root=0)
        comm.Gather(stags_scattered, stags, root=0)
        comm.Gather(np.array(population_pool_addition_local), population_pool_addition, root=0)


        if (myrank == 0):
            # selection of the next generation
            for it in population_pool_addition:
                population_pool.append(it)

            population_pool = np.array(population_pool)
            population_pool, fitness = sort_agents(population_pool, obj_function, graph)
            maximum = sum([f for f in fitness])
            selection_probs = [f/maximum for f in fitness]
            indices = np.random.choice(len(population_pool), size=num_agents, replace=True, p=selection_probs)
            deer = population_pool[indices]

            # update final information
            deer, fitness = sort_agents(deer, obj_function, graph)
            if fitness[0] > Leader_fitness:
                Leader_agent = deer[0].copy()
                Leader_fitness = fitness[0].copy()
            convergence_curve['fitness'][iter_no] = Leader_fitness

    if (myrank == 0):
        # compute final cost
        Leader_agent, Leader_cost = sort_agents(Leader_agent, obj_function, graph)
        deer, cost = sort_agents(deer, obj_function, graph)

        # stop timer
        end_time = MPI.Wtime()
        exec_time = end_time - start_time

        # plot convergence curves
        iters = np.arange(max_iter)+1
        fig, axes = plt.subplots()
        fig.tight_layout(pad = 5)
        fig.suptitle(f'Multi-processing with {N_PROCS} processes')

        axes.set_title('Total Distance vs Iterations')
        axes.set_xlabel('Iteration')
        axes.set_ylabel('Total Distance')
        axes.plot(iters, 1/convergence_curve['fitness'])

        if(save_conv_graph):
            plt.savefig('convergence_graph_RDA.jpg')
        plt.show()

        # update attributes of solution
        solution.best_agent = Leader_agent
        solution.best_fitness = Leader_fitness
        solution.best_cost = Leader_cost
        solution.convergence_curve = convergence_curve
        solution.final_population = deer
        solution.final_fitness = fitness
        solution.final_cost = cost
        solution.execution_time = exec_time

        return solution
    else:
        return None

#ALL ERRORS RESOLVED, WORKING
if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()
    N_PROCS = comm.Get_size()
    N_vertices_sample_1 = 20
    graph_sample_1 = np.zeros((N_vertices_sample_1,N_vertices_sample_1))

    if(myrank == 0):
        for i in range(N_vertices_sample_1):
            for j in range(N_vertices_sample_1):
                if(i<=j):
                    break
                graph_sample_1[i][j] = np.random.randint(200)
                graph_sample_1[j][i] = graph_sample_1[i][j]

    comm.Bcast(graph_sample_1, root=0)
    solution = RDA(num_agents=300, max_iter=20, graph=graph_sample_1, N_vertices=N_vertices_sample_1, obj_function=cycle_cost, save_conv_graph=False, alpha=0.9, beta=0.4, gamma=0.5, num_males_frac=0.20, UB=1, LB=0, myrank=myrank, N_PROCS=N_PROCS)

    if(myrank == 0):
        print('\n================================================================================\n')
        print('RESULTS OBTAINED: ')
        print('TSP SOLUTION: Shortest possible route that visits each city exactly once and returns to the origin city:  : {}'.format(int(1/solution.best_cost)))
        print("EXECUTION TIME: ",solution.execution_time)
        print('\n================================================================================\n')

# #----------Small example to check correctness of code-----------------
# N_vertices = 4
# graph = np.array([[0, 10, 15, 20],
#                   [10, 0, 35, 25],
#                   [15, 35, 0, 30],
#                   [20, 25, 30, 0]])
# # lowest cost for this example is 80 => 1-2-4-3-1
#
# solution = RDA(num_agents=20, max_iter=3, graph=graph, N_vertices=N_vertices, obj_function=cycle_cost, save_conv_graph=True, alpha=0.2, beta=0.1, gamma=0.5, num_males_frac = 0.25, UB=5, LB=-5, myrank=myrank, N_PROCS=N_PROCS)
# #----------Small example-----------------
