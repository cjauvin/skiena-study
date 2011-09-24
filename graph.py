# Python implementation of concepts 
# covered in http://amzn.com/0387948600
#
# Christian Jauvin <cjauvin@gmail.com> 2011

import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

class SkienaGraph:

    def floyd(self):
        n = len(self)
        idx_to_node = dict(zip(range(n), self.nodes()))
        node_to_idx = dict(zip(self.nodes(), range(n)))
        D = defaultdict(lambda: defaultdict(lambda: float('inf')))
        for v in self.nodes():
            for v, w in self.edges(v):
                D[node_to_idx[v]][node_to_idx[w]] = self[v][w]['weight']
        for i in range(n): 
            D[i][i] = 0
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    D[i][j] = min(D[i][j], D[i][k] + D[k][j])
        for i in range(n):
            for j in range(n):
                print D[i][j],
            print        

class SkienaUndirectedGraph(SkienaGraph, nx.Graph):

    def __init__(self, data):
        if len(data[0]) == 2: # unweighted
            nx.Graph.__init__(self, data=data)
        elif len(data[0]) == 3: # weighted
            nx.Graph.__init__(self)
            self.add_weighted_edges_from(data)
        self.discovered = dict([(n, False) for n in self.nodes()])
        self.processed = dict([(n, False) for n in self.nodes()])
        self.parent = dict([(n, -1) for n in self.nodes()])
        self.entry_time = dict([(n, -1) for n in self.nodes()])
        self.exit_time = dict([(n, -1) for n in self.nodes()])
        self.reachable_ancestors = dict([(n, n) for n in self.nodes()])
        self.tree_out_degree = dict([(n, 0) for n in self.nodes()])
        self.time = 0
        self.articulation_vertices = set()
        
    def process_vertex_early(self, v):

        #print 'vertex %s' % v
        pass

    def process_edge(self, x, y):

        edge_class = self.edge_classification(x, y)
        print 'processing edge %s->%s (%s)' % (x, y, edge_class)
        if edge_class == 'TREE':
            self.tree_out_degree[x] += 1
        if edge_class == 'BACK' and self.parent[x] != y:
            if self.entry_time[y] < self.entry_time[self.reachable_ancestors[x]]:
                self.reachable_ancestors[x] = y

    def process_vertex_late(self, v):

#        print 'reachable_ancestors:', self.reachable_ancestors
        print 'late processing of vertex %s:' % v, 

        if self.parent[v] < 0: # is v the root?

            if self.tree_out_degree[v] > 1:
                print 'root articulation vertex %s' % v
                self.articulation_vertices.add(v)
            print
            return

        if self.parent[self.parent[v]] >= 0: # only if parent[v] is not root

            if self.reachable_ancestors[v] == self.parent[v]:
                print 'parent articulation vertex %s;' % self.parent[v],
                self.articulation_vertices.add(self.parent[v])

            if self.reachable_ancestors[v] == v: 
                print 'bridge articulation vertex %s;' % self.parent[v],
                self.articulation_vertices.add(self.parent[v])

                if self.tree_out_degree[v] > 0: # v is not a leaf
                    print 'bridge articulation vertex %s' % v,
                    self.articulation_vertices.add(v)

        print

        time_v = self.entry_time[self.reachable_ancestors[v]]
        time_parent = self.entry_time[self.reachable_ancestors[self.parent[v]]]
    
        if time_v < time_parent:
            self.reachable_ancestors[self.parent[v]] = self.reachable_ancestors[v]

        #print 'reachable_ancestors:', self.reachable_ancestors
        #print

    def dfs(self, v):

        self.discovered[v] = True
        self.time += 1
        self.entry_time[v] = self.time
        self.process_vertex_early(v)
        for p in self.edges(v):
            y = p[1]
            if not self.discovered[y]:
                self.parent[y] = v
                self.process_edge(v, y)
                self.dfs(y)
            elif not self.processed[y] and self.parent[v] != y:
                self.process_edge(v, y)
        self.process_vertex_late(v)
        self.exit_time[v] = self.time
        self.time += 1
        self.processed[v] = True

    def edge_classification(self, x, y):

        if self.parent[y] == x: return 'TREE'
        if self.discovered[y] and not self.processed[y]: return 'BACK'
        assert False

    def prim(self, start=None):

        intree = defaultdict(bool)
        distance = defaultdict(lambda: float('inf'))
        parent = defaultdict()

        if start is None:
            v = self.nodes()[1]
            start = v
        else:
            v = start
        distance[v] = 0
            
        while not intree[v]:

            intree[v] = True
            for _, w in self.edges(v):
                if intree[w]: continue
                if distance[w] > self[v][w]['weight']:
                    distance[w] = self[v][w]['weight']
                    parent[w] = v

            v = start

            min_dist = float('inf')
            for i in self.nodes():
                if not intree[i] and distance[i] < min_dist:
                    min_dist = distance[i]
                    v = i

        print 'has_parent:', dict(parent)
        print 'distance:', dict(distance)

    def dijkstra(self, start, target):

        intree = defaultdict(bool)
        distance = defaultdict(lambda: float('inf'))
        parent = defaultdict()

        v = start
        distance[v] = 0
            
        while True:

            intree[v] = True
            for _, w in self.edges(v):
                if distance[v] + self[v][w]['weight'] < distance[w] and not intree[w]:
                    distance[w] = distance[v] + self[v][w]['weight']
                    parent[w] = v

            v = None
            min_dist = float('inf')
            for u in self.nodes():
                if distance[u] < min_dist and not intree[u]:
                    min_dist = distance[u]
                    v = u

            if v is None or v == target: break

#        print 'has_parent:', dict(parent)
#        print 'distance:', dict(distance)
        path = [target]
        while path[-1] != start:
            path.append(parent[path[-1]])
        print 'path: %s (dist=%s)' % (path[::-1], distance[target])

    def kruskal(self):
        parent = defaultdict()
        uf = UnionFind(self)
        weighted_edges = sorted([(self[v][w]['weight'], v, w) for v, w in self.edges()])
        for weight_vw, v, w in weighted_edges:
            if not uf.sameComponent(v, w):
                uf.union(v, w)
                print v, w

        #print 'has_parent:', dict(parent)

    # using the UnionFind data structure
    def connectedComponents(self):
        uf = UnionFind(self)
        for v, w in self.edges():
            uf.union(v, w)
        connected_components = {} # v -> root
        for v in self.nodes():
            connected_components[v] = uf.find(v)
        print connected_components
        print set(connected_components.values())

class SkienaDirectedGraph(SkienaGraph, nx.DiGraph):

    def __init__(self, data):
        if len(data[0]) == 2: # unweighted
            nx.DiGraph.__init__(self, data=data)
        elif len(data[0]) == 3: # weighted
            nx.DiGraph.__init__(self)
            self.add_weighted_edges_from(data)
        #nx.DiGraph.__init__(self, data=data)
        self.discovered = dict([(n, False) for n in self.nodes()])
        self.processed = dict([(n, False) for n in self.nodes()])
        self.parent = dict([(n, -1) for n in self.nodes()])
        self.entry_time = dict([(n, -1) for n in self.nodes()])
        self.exit_time = dict([(n, -1) for n in self.nodes()])
        self.time = 0
        self.low = dict([(n, n) for n in self.nodes()])
        self.scc = dict([(n, -1) for n in self.nodes()])
        self.components_found = 0
        self.active = [] # used as a stack
        
    def process_vertex_early(self, v):

        #print 'early processing of vertex %s' % v
        self.active.append(v)

    def process_edge(self, x, y):

        edge_class = self.edge_classification(x, y)
        print 'processing edge %s->%s (%s)' % (x, y, edge_class)

        if edge_class == 'BACK':
            if self.entry_time[y] < self.entry_time[self.low[x]]:
                self.low[x] = y

        elif edge_class == 'CROSS':
            if self.scc[y] == -1:
                if self.entry_time[y] < self.entry_time[self.low[x]]:
                    self.low[x] = y

        else: pass

    def process_vertex_late(self, v):

#        print 'late processing of vertex %s:' % v
        if self.low[v] == v:
            print 'popping component at v=%s' % v
            self.pop_component(v)

        if self.parent[v] < 0: return

        #v_debug = 2
        #if v == v_debug: print 'late v=%s before:' % v_debug, self.low
        
        if self.entry_time[self.low[v]] < self.entry_time[self.low[self.parent[v]]]:
            self.low[self.parent[v]] = self.low[v]

        #if v == v_debug: print 'late v=%s after:' % v_debug, self.low

    def pop_component(self, v):

        self.components_found += 1
        self.scc[v] = self.components_found
        while True:
            t = self.active.pop()
            if t != v:
                self.scc[t] = self.components_found
            else:
                break

    def dfs(self, v):

        self.discovered[v] = True
        self.time += 1
        self.entry_time[v] = self.time
        self.process_vertex_early(v)
        for p in self.edges(v):
            y = p[1]
            if not self.discovered[y]:
                self.parent[y] = v
                self.process_edge(v, y)
                self.dfs(y)
            elif not self.processed[y]: 
                self.process_edge(v, y)
        self.process_vertex_late(v)
        self.exit_time[v] = self.time
        self.time += 1
        self.processed[v] = True

    def edge_classification(self, x, y):

        if self.parent[y] == x: return 'TREE'
        elif self.discovered[y] and not self.processed[y]: return 'BACK'
        elif self.processed[y] and self.entry_time[y] > self.entry_time[x]: return 'FORWARD'
        elif self.processed[y] and self.entry_time[y] < self.entry_time[x]: return 'CROSS'
        assert False                    

# from the book: http://amzn.com/0133350681
class BBGraph(nx.Graph):

    def __init__(self, data):
        if len(data[0]) == 2: # unweighted
            nx.Graph.__init__(self, data=data)
        elif len(data[0]) == 3: # weighted
            nx.Graph.__init__(self)
            self.add_weighted_edges_from(data)
        self.prenum = defaultdict(int)
        self.postnum = defaultdict(int)
        self.visited = defaultdict(bool)
        self.highest = defaultdict(int)
        self.prenum_counter = 0
        self.postnum_counter = 0
        self.parent = defaultdict(int)
        self.children = defaultdict(list)

    def dfs(self, v):
        assert not self.visited[v]
        self.visited[v] = True
        self.prenum_counter += 1
        self.prenum[v] = self.prenum_counter
        for _, w in self.edges(v):
            if not self.visited[w]:
                self.parent[w] = v
                self.children[v].append(w)
                self.dfs(w)
        self.postnum_counter += 1
        self.postnum[v] = self.postnum_counter
        self.highest[v] = self.compute_highest(v)

    def compute_highest(self, v):
        possible_values = [self.prenum[v]]
        for _, w in self.edges(v):
            if self.parent[v] == w: continue
            possible_values.append(self.prenum[w])
        for x in self.children[v]:
            possible_values.append(self.highest[x])
        return min(possible_values)
        
    def find_articulation_points(self):
        self.articulation_points = set()
        for v in self.nodes():
            if self.parent[v] == 0:
                if len(self.children[v]) >= 2:
                    print 'found av %s (root)' % v
                    self.articulation_points.add(v)
            else:
                for x in self.children[v]:
                    if self.highest[x] >= self.prenum[v]:
                        print 'found av %s (non-root)' % v
                        self.articulation_points.add(v)
                        break
    def prim(self):

        T = set()
        L = nx.to_numpy_matrix(self)
        n = len(self)
        for i in range(n):
            for j in range(n):
                if L[i,j] == 0:
                    L[i,j] = float('inf')
        idx_to_node = dict(zip(range(n), self.nodes()))
        nearest = [0] * n
        mindist = [0] * n
        for i in range(1, n):
            nearest[i] = 0
            mindist[i] = L[i,0]
        print mindist
        for _ in range(n - 1):
            curr_mindist = float('inf')
            for j in range(1, n):
                if 0 <= mindist[j] < curr_mindist:
                    curr_mindist = mindist[j]
                    k = j
            T.add((idx_to_node[nearest[k]], idx_to_node[k]))
            mindist[k] = -1
            for j in range(1, n):
                if L[j,k] > 0 and L[j,k] < mindist[j]:
                    mindist[j] = L[j,k]
                    nearest[j] = k
        return T

class UnionFind:

    def __init__(self, g):
#        self.g = g
        self.parent = dict([(n, n) for n in g.nodes()])
        self.size = defaultdict(lambda: 1) # size of subtree i
                
    def find(self, i):
        if self.parent[i] == i:
            return i
        else:
            return self.find(self.parent[i])

    def union(self, i, j):
        ri = self.find(i)
        rj = self.find(j)
        if ri == rj: return
        if self.size[ri] >= self.size[rj]: # attach rj as a ri child
            self.parent[rj] = ri
            self.size[ri] += self.size[rj]
        else:
            pass

    def sameComponent(self, i, j):
        return self.find(i) == self.find(j)
            

#g = SkienaGraph([(1,2), (1,6), (2,3), (3,4), (4,5), (5,2)])
#g = SkienaGraph([(1,2), (2,3), (3,4), (4,5), (4,2)])
#g = SkienaGraph([(1,3), (3,5), (5,6), (2,4), (4,5), (5,6), (1,6), (5,2)])

# weighted 
#g = SkienaGraph([(1,2,1), (1,4,4), (2,3,2), (2,4,6), (2,5,4), (3,5,5), 
#                 (3,6,6), (4,5,3), (4,7,4), (5,6,8), (5,7,7), (6,7,3), (8,9,1)])

#g = SkienaDirectedGraph([(1,2,5), (2,1,50), (2,3,15), (2,4,5), (3,1,30), (3,4,15), (4,1,15), (4,3,5)])
#g = SkienaUndirectedGraph([(1,2,5), (2,3,15), (2,4,5), (3,1,30), (3,4,15), (4,1,15)])

#g.floyd()

# g = BBGraph([(1,2,1), (1,4,4), (2,3,2), (2,4,6), (2,5,4), (3,5,5), 
#              (3,6,6), (4,5,3), (4,7,4), (5,6,8), (5,7,7), (6,7,3)])

#g = SkienaGraph([('a','b',5), ('a','d',20), ('b','c',10), ('b','d',15)])

#g.prim()
#g.dijkstra(1, 6)
#g.dijkstra(6, 7)

#g.kruskal()
#g.connectedComponents()

#g = BBGraph([(1,3), (3,5), (5,6), (2,4), (4,5), (5,6), (1,6), (5,2)])
#g = BBGraph([(1,2), (1,6), (2,3), (3,4), (4,5), (5,2)])

#g = SkienaDiGraph([(1,2), (2,3), (3,4), (4,2)])
#g = SkienaDiGraph([(1,2), (2,3), (3,1), (3,4), (4,2)])
#g = SkienaDiGraph([(1,2), (2,3), (3,1), (3,4), (4,5), (5,4)])
#g = SkienaDiGraph([(1,2), (2,3), (3,1), (2,10), (10,4), (4,5), (5,6), (6,4)])
#g = SkienaDiGraph([(1,2), (2,1)])

#g.dfs(1)

#print g.low
#print g.scc
#assert len(set(g.scc.values())) == nx.number_strongly_connected_components(g)

#g.find_articulation_points()

#print g.prenum
#print g.highest
#print g.articulation_points

#print 'articulation vertices:', g.articulation_vertices

#print g.entry_time
#print g.exit_time

#nx.draw_circular(g)
#plt.show()
