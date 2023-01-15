import utils

from numpy.random import seed
from brute_force import brute_force
from greedy_search import greedy_search
from heuristics import heuristic_search


# seed(6)
max_cities = 9
for dimmension in range(5, max_cities + 1):
    cities = utils.generate_cities(dimmension)

    fc_sym_graph = utils.get_symmetric_graph(cities)
    fc_asym_graph = utils.get_asymmetric_graph(fc_asym_graph, cities)
    nfc_sym_graph = utils.remove_connections(fc_sym_graph)
    nfc_asym_graph = utils.remove_connections(fc_asym_graph)

    print(brute_force(graph))
    print(greedy_search(graph))
    print(heuristic_search(graph))