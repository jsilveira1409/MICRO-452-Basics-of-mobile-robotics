import numpy as np
from collections import namedtuple 

INF = 80000
THYMIO_RADIUS = 200

#----------------------------------------- AUGMENTING OBSTACLES ---------------------------------------------------------------------------
def augment(input_points):
    modif = []
    for o in input_points :
        o_n = np.array(o)
        center = np.mean(o_n,axis=0)
        o_n = o_n + THYMIO_RADIUS*(o_n-center)/np.transpose([np.linalg.norm(o_n-center, axis=1),np.linalg.norm(o_n-center, axis=1)])
        modif.append(o_n.tolist())
    return modif
    

#----------------------------------------- GRAPH CONSTRUCTION ---------------------------------------------------------------------------

def euclid_dist(point1, point2):
    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def closest_pair(list1, list2):
    idx1 = 0
    idx2 = 0
    min_dist = INF
    for i in range(len(list1)):
        for j in range(len(list2)):
            dist = euclid_dist(list1[i],list2[j]) #abs(list1[i,0]-list2[j,0])

            if(dist<min_dist):
                min_dist = dist
                idx1 = i
                idx2 = j
    
    return idx1, idx2, min_dist

def get_farthest(point,list_p):
    idx = 0
    max_dist = 0
    for i in range(len(list_p)):
        dist = euclid_dist(point,list_p[i])

        if(dist>max_dist):
            max_dist = dist
            idx = list_p[i][2]
    
    return idx

def number_vertices(points:list):
    i = 0
    for obs in points:
        for p in obs:
            p.append(i)
            i+=1


def construct_graph(vertices:list,start,end):
    graph = list()

    #memorization to look for closest points to start and end
    idx_s = 0
    min_d_s = INF
    idx_e = 0
    min_d_e = INF

    #connecting every vertex to the vertices in the same obstacle, except the opposite one
    for o in vertices:
        for i in range(len(o)):
            p = o[i]
            i_next = (i+1 if i<len(o)-1 else 0)
            graph.append([[o[i-1][2],euclid_dist(o[i-1],p)],[o[i_next][2],euclid_dist(o[i_next],p)]])
            #graph.append([[op[2],euclid_dist(op,p)] for op in o if(op[2]!=p[2] and (op[2] not in opposite))])

            #update closest vertex to start or end if p is closer than previous closest
            dist_s = euclid_dist(start,p)
            if(dist_s<min_d_s):
                min_d_s = dist_s
                idx_s = p[2]

            dist_e = euclid_dist(end,p)
            if(dist_e<min_d_e):
                min_d_e = dist_e
                idx_e = p[2]
    
    #adding start and end vertices, connected to the closest other vertex
    nb_v = len(graph)
    graph.append([[idx_s,dist_s],[idx_s,dist_s]])
    graph[idx_s].append([nb_v,dist_s])
    graph.append([[idx_e,dist_e],[idx_e,dist_e]])
    graph[idx_e].append([nb_v+1,dist_e])

    #connecting obstacles with other obstacles (closest to closest)
    nb_obs = len(vertices)
    for i in range(nb_obs):
        for j in range(i+1,nb_obs):
            other_points = [p for obs in vertices[j:] for p in obs] #flatten the list of all points of obstacles we're looking at
            idx1, idx2, dist = closest_pair(vertices[i],other_points)

            graph[vertices[i][idx1][2]].append([other_points[idx2][2],dist])
            graph[other_points[idx2][2]].append([vertices[i][idx1][2],dist])

    return graph

def print_list(l):
    for element in l:
        print(element)

#---------------------------------------------------- DIJKSTRA ----------------------------------------------------------------------

def get_weight(a,b,graph):
    #selecting vertex with the smallest number of connections
    if(len(graph[a])<len(graph[b])):
        ensemble = graph[a]
        idx = b
    else:
        ensemble = graph[b]
        idx = a

    for point in ensemble:
        if point[0] == idx:
            return point[1]
    return -1

def remove(a, list_r):
    idx = list_r.index(a)
    list_r.pop(idx)


def dijkstra(graph):
    nb_v = len(graph)
    unconnected = list(range(nb_v))
    distances = INF*np.ones((nb_v,2))
    distances[:,1] = range(nb_v)
    distances[-2,0] = 0
    predecessor = np.zeros(nb_v, dtype=int)

    while(len(unconnected)!=0):
        dp = distances.copy()[unconnected]
        idx_a = np.argmin(dp,axis=0)[0]
        a = int(dp[idx_a,1])
        remove(a,unconnected)

        for v_b in graph[a]:
            b = v_b[0]

            w_ab = get_weight(a,b,graph)
            if((w_ab > 0) and (distances[b,0]> distances[a,0] + w_ab)):
                distances[b,0] = distances[a,0] + w_ab
                predecessor[b] = a

    
    return distances,predecessor


def construct_path(start,end,coordlist,pred):
    i = len(pred)-1
    path_num = []
    coord = []
    coord.append(end)

    while(pred[i] != len(pred)-2):
        i = pred[i]
        path_num.append(i)
        coord.append(coordlist[i][0:2])

    #coord.append(start)
    coord.reverse()
    return coord

#----------------------------------------------- MAIN FUNCTION ------------------------------------------------------------------------
def compute_shortest_path(input_ar,start,end):
    input_ar = augment(input_ar)
    flat = [p for obs in input_ar for p in obs]
    number_vertices(input_ar)
    g = construct_graph(input_ar,start,end)
    d,pred = dijkstra(g)
    path = construct_path(start,end,flat,pred)
    return path

#----------------------------------------------- INPUTS ------------------------------------------------------------------------
#input_ar = [[[5.0,3.0],[8,3],[8,6],[5,4]],
#            [[8,13],[7,12],[8,10],[9,11]],
#            [[4,9],[2,10],[4,12]]]
#
#A = [1,1]
#B = [9,14]    

#----------------------------------------------- TEST STUFF ------------------------------------------------------------------------

#compute_shortest_path(input_ar,A,B)
'''flat = [p for obs in input_ar for p in obs]
number_vertices(input_ar)
g = construct_graph(input_ar,A,B)
d,pred = dijkstra(g)
print("PATH")
path = construct_path(A,B,flat,pred)
print(path)'''