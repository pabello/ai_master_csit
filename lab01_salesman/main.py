import utils

from numpy.random import seed
from brute_force import brute_force
from greedy_search import greedy_search
from heuristics import heuristic_search


with open("results.md", "w") as file:
    file.write(f"| Dimmension | Smmetric | Fully Connected | Algorithm | Solution | Time |\n")
    file.write(f"|------------|----------|-----------------|-----------|----------|------|\n")


max_cities = 8
for dimmension in range(5, max_cities + 1):
    cities = utils.generate_cities(dimmension)

    fc_sym_graph = utils.get_symmetric_graph(cities)
    fc_asym_graph = utils.get_asymmetric_graph(fc_sym_graph, cities)
    nfc_sym_graph = utils.remove_connections(fc_sym_graph)
    nfc_asym_graph = utils.remove_connections(fc_asym_graph)

    graphs = ((fc_sym_graph, "SYM", "FC"),
              (fc_asym_graph, "ASYM", "FC"),
              (nfc_sym_graph, "SYM", "NFC"),
              (nfc_asym_graph, "ASYM", "NFC"),)

    for graph in graphs:
        print(graph[1:])
        bf = brute_force(graph[0])
        gs = greedy_search(graph[0])
        hs = heuristic_search(graph[0])

        with open("results.md", "a") as file:
            file.write(f"| {dimmension} | {graph[1]} | {graph[2]} | Brute force DFS | {bf.get('dfs')[0]} | {bf.get('dfs')[1] * 1000:0.4f} ms |\n")
            file.write(f"| {dimmension} | {graph[1]} | {graph[2]} | Brute force BFS | {bf.get('bfs')[0]} | {bf.get('bfs')[1] * 1000:0.4f} ms |\n")
            file.write(f"| {dimmension} | {graph[1]} | {graph[2]} | Greedy search | {gs[0]} | {gs[1] * 1000:0.4f} ms |\n")
            file.write(f"| {dimmension} | {graph[1]} | {graph[2]} | A* AVG | {hs.get('avg')[0]} | {hs.get('avg')[1] * 1000:0.4f} ms |\n")
            file.write(f"| {dimmension} | {graph[1]} | {graph[2]} | A* MIN | {hs.get('min')[0]} | {hs.get('min')[1] * 1000:0.4f} ms |\n")
    
    with open("results.md", "a") as file:
        file.write("|||||||\n")