import rda_algo_cython
import numpy as np

N_vertices_sample_1 = 120
graph_sample_1 = np.zeros((120,120))

for i in range(N_vertices_sample_1):
    for j in range(N_vertices_sample_1):
        if(i<=j):
            break
        graph_sample_1[i][j] = np.random.randint(1000)
        graph_sample_1[j][i] = graph_sample_1[i][j]

solution = rda_algo_cython.RDA(num_agents=1000, max_iter=20, graph=graph_sample_1, N_vertices=N_vertices_sample_1, obj_function=rda_algo_cython.cycle_cost, save_conv_graph=True, alpha=0.9, beta=0.4, gamma=0.5, num_males_frac=0.15, UB=5, LB=-5)
