# -*- coding: utf-8 -*-
"""RDA_ALGO.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13ZJGOKhtCdrwxten-W_EDiwLVQFrV9XZ
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import random, math
import sys
from tqdm import tqdm
np.random.seed(44)

from utilities import Solution, initialize, sort_agents, cycle_cost, display

from mpi4py import MPI


def RDA(num_agents, max_iter, graph, N_vertices, obj_function, save_conv_graph, alpha, beta, gamma, num_males_frac, UB, LB, myrank, N_PROCS):

    # Red Deer Algorithm
    ############################### Parameters ####################################
    #                                                                             #
    #   num_agents: number of red deers                                           #
    #   max_iter: maximum number of generations                                   #
    #   obj_function: the function to maximize while doing feature selection      #
    #   trans_function_shape: shape of the transfer function used                 #
    #   save_conv_graph: boolean value for saving convergence graph               #
    #                                                                             #
    ###############################################################################

    # Number of agents must be at least 8
    if num_agents < 8:
        print("[Error!] The value of the parameter num_agents must be at least 8", file=sys.stderr)
        sys.exit(1)

    short_name = 'RDA'
    agent_name = 'RedDeer'

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
    start_time = time.time()

    # main loop
    for iter_no in tqdm(range(max_iter)):
        # print('\n================================================================================')
        # print('                          Iteration - {}'.format(iter_no+1))
        # print('================================================================================\n')
        num_males = int(num_males_frac * num_agents)
        if (myrank == 0):
            deer, fitness = sort_agents(deer, obj_function, graph)
            num_hinds = num_agents - num_males
            males = deer[:num_males,:]
            hinds = deer[num_males:,:]
        else:
            males = None

        # print("NUM MALES=", num_males)
        assert num_males % N_PROCS == 0
        local_num_males = num_males // N_PROCS
        males_scattered = np.zeros((local_num_males, N_vertices))

        comm.Scatter(males, males_scattered, root=0)
        # roaring of male deer
        for i in range(local_num_males):
            r1 = np.random.random() # r1 is a random number in [0, 1]
            r2 = np.random.random() # r2 is a random number in [0, 1]
            r3 = np.random.random() # r3 is a random number in [0, 1]
            new_male = males_scattered[i].copy()
            if r3 >= 0.5:                                    # Eq. (3)
                new_male += r1 * (((UB - LB) * r2) + LB)
            else:
                new_male -= r1 * (((UB - LB) * r2) + LB)

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
            r1 = np.random.random()
            r2 = np.random.random()
            new_male_1 = (chosen_com + chosen_stag) / 2 + r1 * (((UB - LB) * r2) + LB) # Eq. (6)
            new_male_2 = (chosen_com + chosen_stag) / 2 - r1 * (((UB - LB) * r2) + LB) # Eq. (7)

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

        if (myrank == 0):
            # formation of harems
            coms, fitness = sort_agents(coms, obj_function, graph)
            norm = np.linalg.norm(fitness)
            normal_fit = fitness / norm
            total = np.sum(normal_fit)
            power = normal_fit / total # Eq. (9)
            num_harems = [int(x * num_hinds) for x in power] # Eq.(10)
            max_harem_size = np.max(num_harems)
            harem = np.empty(shape=(num_coms, max_harem_size, N_vertices))
            random.shuffle(hinds)
            itr = 0
            for i in range(num_coms):
                harem_size = num_harems[i]
                for j in range(harem_size):
                    harem[i][j] = hinds[itr]
                    itr += 1


            # mating of commander with hinds in his harem
            num_harem_mate = [int(x * alpha) for x in num_harems] # Eq. (11)
            population_pool = list(deer)
            for i in range(num_coms):
                random.shuffle(harem[i])
                for j in range(num_harem_mate[i]):
                    r = np.random.random() # r is a random number in [0, 1]
                    offspring = (coms[i] + harem[i][j]) / 2 + (UB - LB) * r # Eq. (12)

                    population_pool.append(list(offspring))

                    # if number of commanders is greater than 1, inter-harem mating takes place
                    if num_coms > 1:
                        # mating of commander with hinds in another harem
                        k = i
                        while k == i:
                            k = random.choice(range(num_coms))

                        num_mate = int(num_harems[k] * beta) # Eq. (13)

                        np.random.shuffle(harem[k])
                        for j in range(num_mate):
                            r = np.random.random() # r is a random number in [0, 1]
                            offspring = (coms[i] + harem[k][j]) / 2 + (UB - LB) * r
                            population_pool.append(list(offspring))

            # mating of stag with nearest hind
            for stag in stags:
                dist = np.zeros(num_hinds)
                for i in range(num_hinds):
                    dist[i] = math.sqrt(np.sum((stag-hinds[i])*(stag-hinds[i])))
                min_dist = np.min(dist)
                for i in range(num_hinds):
                    distance = math.sqrt(np.sum((stag-hinds[i])*(stag-hinds[i]))) # Eq. (14)
                    if(distance == min_dist):
                        r = np.random.random() # r is a random number in [0, 1]
                        offspring = (stag + hinds[i])/2 + (UB - LB) * r
                        population_pool.append(list(offspring))

                        break

            # selection of the next generation
            population_pool = np.array(population_pool)
            population_pool, fitness = sort_agents(population_pool, obj_function, graph)
            maximum = sum([f for f in fitness])
            selection_probs = [f/maximum for f in fitness]
            indices = np.random.choice(len(population_pool), size=num_agents, replace=True, p=selection_probs)
            deer = population_pool[indices]

            # update final information
            deer, fitness = sort_agents(deer, obj_function, graph)
            #display(deer, fitness, agent_name)
            if fitness[0] > Leader_fitness:
                Leader_agent = deer[0].copy()
                Leader_fitness = fitness[0].copy()
            convergence_curve['fitness'][iter_no] = Leader_fitness
            convergence_curve['feature_count'][iter_no] = int(np.sum(Leader_agent))

    if (myrank == 0):
        # compute final cost
        Leader_agent, Leader_cost = sort_agents(Leader_agent, obj_function, graph)
        deer, cost = sort_agents(deer, obj_function, graph)

        print('\n================================================================================')
        print('                                    Final Result                                  ')
        print('================================================================================\n')
        print('Leader ' + agent_name + ' Fitness : {}'.format(Leader_fitness))
        print('Leader ' + agent_name + ' Lowest cost : {}'.format(-Leader_cost))
        print('\n================================================================================\n')

        # stop timer
        end_time = time.time()
        exec_time = end_time - start_time

        # # plot convergence curves
        # iters = np.arange(max_iter)+1
        # fig, axes = plt.subplots()
        # fig.tight_layout(pad = 5)
        # # fig.suptitle('Convergence Curves')
        #
        # axes.set_title('Total Distance vs Iterations')
        # axes.set_xlabel('Iteration')
        # axes.set_ylabel('Total Distance')
        # axes.plot(iters, -convergence_curve['fitness'])
        #
        # if(save_conv_graph):
        #     plt.savefig('convergence_graph_'+ short_name + '.jpg')
        # plt.show()

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


if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()
    N_PROCS = comm.Get_size()
    N_vertices_sample_1 = 120
    graph_sample_1 = np.zeros((120,120))

    if(myrank == 0):
        for i in range(N_vertices_sample_1):
            for j in range(N_vertices_sample_1):
                if(i<=j):
                    break
                graph_sample_1[i][j] = np.random.randint(1000)
                graph_sample_1[j][i] = graph_sample_1[i][j]

    comm.Bcast(graph_sample_1, root=0)
    solution = RDA(num_agents=1000, max_iter=20, graph=graph_sample_1, N_vertices=N_vertices_sample_1, obj_function=cycle_cost, save_conv_graph=True, alpha=0.9, beta=0.4, gamma=0.5, num_males_frac=0.20, UB=5, LB=-5, myrank=myrank, N_PROCS=N_PROCS)
    if(myrank == 0):
        print(solution.execution_time)
