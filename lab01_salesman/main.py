import utils

from numpy.random import seed
from brute_force import brute_force
from greedy_search import greedy_search
from heuristics import heuristic_search


# seed(6)ssss
dimmension = 9

cities = utils.generate_cities(dimmension)
graph = utils.get_symmetric_graph(cities)

print(brute_force(graph))
print(greedy_search(graph))
print(heuristic_search(graph))