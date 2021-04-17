import numpy as np
np.random.seed(44)

class Solution():
    #structure of the solution
    def __init__(self):
        self.N_vetices = None
        self.num_agents = None
        self.max_iter = None
        self.obj_function = None
        self.execution_time = None
        self.convergence_curve = {}
        self.best_agent = None
        self.best_fitness = None
        self.best_cost = None
        self.final_population = None
        self.final_fitness = None
        self.final_cost = None

def initialize(num_agents, N_vertices):

    agents = np.zeros((num_agents,N_vertices))

    for agent_no in range(num_agents):
      agents[agent_no] = np.random.choice([0, 1], size=N_vertices, p=[.5, .5])

    return agents


def sort_agents(agents, obj_function, graph):

    # if there is only one agent
    if len(agents.shape) == 1:
        num_agents = 1
        fitness = obj_function(agents, graph)
        return agents, fitness

    # for multiple agents
    else:
        num_agents = agents.shape[0]
        fitness = np.zeros(num_agents)
        for id, agent in enumerate(agents):
            fitness[id] = obj_function(agent, graph)
        idx = np.argsort(-fitness)
        sorted_agents = agents[idx].copy()
        sorted_fitness = fitness[idx].copy()

    return sorted_agents, sorted_fitness

def cycle_cost(agent, graph):
  encoding = np.argsort(np.argsort(agent)) #gives sorted rank of each element, eg: [0.2, 0.5, 0.1, 0.9] -> [1 2 0 3]
  cycle = np.append(encoding,encoding[0])
  cost = 0
  for i in range(0, len(cycle)-1):
    cost = cost + graph[cycle[i]][cycle[i+1]]
  return -cost

def display(agents, fitness, agent_name='Agent'):
    # display the population
    print('\nNumber of agents: {}'.format(agents.shape[0]))
    print('\n------------- Best Agent ---------------')
    print('Fitness: {}'.format(fitness[0]))
    print('----------------------------------------\n')

    for id, agent in enumerate(agents):
        print('{} {} - Fitness: {}'.format(agent_name, id+1, fitness[id]))

    print('================================================================================\n')
