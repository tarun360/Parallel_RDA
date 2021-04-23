"""

SERIAL CODE

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
import time
import matplotlib.pyplot as plt
import random, math
import sys
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

from utilities import Solution, initialize, sort_agents, cycle_cost, display

def RDA(num_agents, max_iter, graph, N_vertices, obj_function, save_conv_graph, alpha, beta, gamma, num_males_frac, UB, LB):

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

    # initializing parameters
    # UB = 5 # Upper bound
    # LB = -5 # Lower bound
    # gamma = 0.7 # Fraction of total number of males who are chosen as commanders
    # alpha = 0.9 # Fraction of total number of hinds in a harem who mate with the commander of their harem
    # beta = 0.4 # Fraction of total number of hinds in a harem who mate with the commander of a different harem

    # start timer
    start_time = time.time()

    # main loop
    for iter_no in tqdm(range(max_iter)):
        # print('\n================================================================================')
        # print('                          Iteration - {}'.format(iter_no+1))
        # print('================================================================================\n')
        deer, fitness = sort_agents(deer, obj_function, graph)
        num_males = int(num_males_frac * num_agents)
        num_hinds = num_agents - num_males
        males = deer[:num_males,:]
        hinds = deer[num_males:,:]

        # roaring of male deer
        for i in range(num_males):
            # r1 = np.random.random() # r1 is a random number in [0, 1]
            # r2 = np.random.random() # r2 is a random number in [0, 1]
            # r3 = np.random.random() # r3 is a random number in [0, 1]
            # new_male = males[i].copy()

            sign_arr = np.random.choice([1, -1], size=N_vertices, p=[.5, .5])
            random_arr1 = np.random.rand(N_vertices)
            random_arr2 = np.random.rand(N_vertices)
            new_male += sign_arr*random_arr1*((UB-LB)*random_arr2+LB)
            # new_male = (new_male-np.min(new_male))/(np.max(new_male)-np.min(new_male))
            # if r3 >= 0.5:                                    # Eq. (3)
            #     new_male += r1 * (((UB - LB) * r2) + LB)
            # else:
            #     new_male -= r1 * (((UB - LB) * r2) + LB)

            if obj_function(new_male, graph) < obj_function(males[i], graph):
                males[i] = new_male


        # selection of male commanders and stags
        num_coms = int(num_males * gamma) # Eq. (4)
        num_stags = num_males - num_coms # Eq. (5)

        coms = males[:num_coms,:]
        stags = males[num_coms:,:]
        # fight between male commanders and stags
        for i in range(num_coms):
            chosen_com = coms[i].copy()
            chosen_stag = random.choice(stags)

            random_arr1 = np.random.rand(N_vertices)
            random_arr2 = np.random.rand(N_vertices)

            new_male_1 = (chosen_com + chosen_stag)/2 + random_arr1*( (UB-LB)*random_arr2 + LB )
            new_male_2 = (chosen_com + chosen_stag)/2 - random_arr1*( (UB-LB)*random_arr2 + LB )

            # new_male_1 = (new_male_1-np.min(new_male_1))/(np.max(new_male_1)-np.min(new_male_1))
            # new_male_2 = (new_male_2-np.min(new_male_2))/(np.max(new_male_2)-np.min(new_male_2))

            # r1 = np.random.random()
            # r2 = np.random.random()
            # new_male_1 = (chosen_com + chosen_stag) / 2 + r1 * (((UB - LB) * r2) + LB) # Eq. (6)
            # new_male_2 = (chosen_com + chosen_stag) / 2 - r1 * (((UB - LB) * r2) + LB) # Eq. (7)

            fitness = np.zeros(4)
            fitness[0] = obj_function(chosen_com, graph)
            fitness[1] = obj_function(chosen_stag, graph)
            fitness[2] = obj_function(new_male_1, graph)
            fitness[3] = obj_function(new_male_2, graph)

            bestfit = np.max(fitness)
            if fitness[0] < fitness[1] and fitness[1] == bestfit:
                coms[i] = chosen_stag.copy()
            elif fitness[0] < fitness[2] and fitness[2] == bestfit:
                coms[i] = new_male_1.copy()
            elif fitness[0] < fitness[3] and fitness[3] == bestfit:
                coms[i] = new_male_2.copy()

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
        # print(len(coms),len(harem),len(num_harem_mate))
        for i in range(num_coms):
            random.shuffle(harem[i])
            for j in range(num_harem_mate[i]):

                random_arr = np.random.rand(N_vertices)
                offspring = (coms[i]+harem[i][j]) / 2 + (UB-LB)*random_arr
                # offspring = (offspring-np.min(offspring))/(np.max(offspring)-np.min(offspring))

                # r = np.random.random() # r is a random number in [0, 1]
                # offspring = (coms[i] + harem[i][j]) / 2 + (UB - LB) * r # Eq. (12)

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
                        random_arr = np.random.rand(N_vertices)
                        offspring = (coms[i]+harem[k][j])/2 + (UB-LB)*random_arr
                        # offspring = (offspring-np.min(offspring))/(np.max(offspring)-np.min(offspring))
                        # r = np.random.random() # r is a random number in [0, 1]
                        # offspring = (coms[i] + harem[k][j]) / 2 + (UB - LB) * r
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
                    random_arr = np.random.rand(N_vertices)
                    offspring = (stag + hinds[i])/2 + (UB - LB) * random_arr
                    # offspring = (offspring-np.min(offspring))/(np.max(offspring)-np.min(offspring))
                    # r = np.random.random() # r is a random number in [0, 1]
                    # offspring = (stag + hinds[i])/2 + (UB - LB) * r
                    population_pool.append(list(offspring))

                    break

        # selection of the next generation
        population_pool = np.array(population_pool)
        population_pool, fitness = sort_agents(population_pool, obj_function, graph)
        maximum = sum([f for f in fitness])
        selection_probs = [f/maximum for f in fitness]
        indices = np.random.choice(len(population_pool), size=num_agents, replace=True, p=selection_probs)
        deer = population_pool[indices]
        # deer = population_pool[:num_agents]
        # update final information
        deer, fitness = sort_agents(deer, obj_function, graph)
        #display(deer, fitness, Red Deer)
        if fitness[0] > Leader_fitness:
            Leader_agent = deer[0].copy()
            Leader_fitness = fitness[0].copy()
        convergence_curve['fitness'][iter_no] = Leader_fitness
        # print(Leader_agent)
    # compute final cost
    Leader_agent, Leader_cost = sort_agents(Leader_agent, obj_function, graph)
    deer, cost = sort_agents(deer, obj_function, graph)

    # stop timer
    end_time = time.time()
    exec_time = end_time - start_time

    # plot convergence curves
    iters = np.arange(max_iter)+1
    fig, axes = plt.subplots()
    fig.tight_layout(pad = 5)
    # fig.suptitle('Convergence Curves')

    axes.set_title('Total Distance vs Iterations')
    axes.set_xlabel('Iteration')
    axes.set_ylabel('Fitness')
    axes.plot(iters, 1/convergence_curve['fitness'], marker='o')

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


# #----------Small example-----------------
# N_vertices = 4
# graph = np.array([[0, 10, 15, 20],
#                   [10, 0, 35, 25],
#                   [15, 35, 0, 30],
#                   [20, 25, 30, 0]])
# # lowest cost for this example is 80 => 1-2-4-3-1
#
# solution = RDA(num_agents=8, max_iter=3, graph=graph, N_vertices=N_vertices, obj_function=cycle_cost, save_conv_graph=True, alpha=0.2, beta=0.1, gamma=0.5, num_males_frac = 0.25, UB=5, LB=-5)
# #----------Small example-----------------

np.random.seed(44)

if __name__ == "__main__":

    N_vertices_sample_1 = 100
    graph_sample_1 = np.zeros((N_vertices_sample_1,N_vertices_sample_1))

    for i in range(N_vertices_sample_1):
      for j in range(N_vertices_sample_1):
        if(i<=j):
          break
        graph_sample_1[i][j] = np.random.randint(1000)
        graph_sample_1[j][i] = graph_sample_1[i][j]

    solution = RDA(num_agents=600, max_iter=20, graph=graph_sample_1, N_vertices=N_vertices_sample_1, obj_function=cycle_cost, save_conv_graph=False, alpha=0.8, beta=0.9, gamma=0.4, num_males_frac=0.20, UB=1, LB=0)

    print('\n================================================================================\n')
    print('RESULTS OBTAINED: ')
    # print('Leader Red Deer Fitness : {}'.format(Leader_fitness))
    print('Leader Red Deer Lowest cost : {}'.format(1/solution.best_cost))
    print("EXECUTION TIME: ",solution.execution_time)
    print('\n================================================================================\n')
