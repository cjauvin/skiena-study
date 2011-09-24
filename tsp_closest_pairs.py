from numpy import *
import networkx as nx

def dist(a, b):
    return linalg.norm(b - a)

e = 0
x = 1 + e
y = 1 - e
pts = [array((0,0)), array((x,0)), array((2*x, 0)),
       array((0,y)), array((x,y)), array((2*x, y))]
g = nx.Graph()
for i in range(len(pts)): g.add_node(i)

for _ in pts:
    d = float('inf')
    chains = nx.connected_component_subgraphs(g)
    for chain_s in chains:
        for chain_t in chains:
            if chain_s == chain_t: continue
            st_endnodes = [[], []]
            for i, chain in enumerate([chain_s, chain_t]):
                for node in chain:
                    if g.degree(node) <= 1:
                        st_endnodes[i].append(node)
            for idx in [[0,0], [0,-1], [-1,0], [-1,-1]]:
                s, t = st_endnodes[0][idx[0]], st_endnodes[1][idx[1]]
                dist_st = dist(pts[s], pts[t])
                if dist_st <= d:
                    sm = s
                    tm = t
                    d = dist_st
    g.add_edge(sm, tm)
assert(len(nx.connected_component_subgraphs(g)) == 1)
last_pair = []
for node in g.nodes():
    if g.degree(node) == 1:
        last_pair.append(node)
g.add_edge(last_pair[0], last_pair[1])
print g.edges()
