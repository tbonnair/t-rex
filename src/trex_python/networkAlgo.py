#Tony BONNAIRE
#created on October, 2018

from sklearn.neighbors import NearestNeighbors, KDTree
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
import time
import trex_python.utility as utility
from scipy.spatial import Delaunay, cKDTree
import scipy.sparse as sp
from tqdm import tqdm

axSize = [0.1,0.1,0.8,0.8]
cbarPos = [0.9, 0.1, 0.025, 0.8]
figSize = (10, 8)

#==========================
#  Construct network
#==========================
def construct_network_KNN(data, K_NN):
    '''Compute the K-NN algorithm using kd_tree algorithm

    [+] data: (N,D) numpy array
            N vectors of D elements (D dimensions, N datapoints)
    [+] K_NN: float
            Number of neighbors

    [-] indices:
            Vector of N elements (number of datapoints) from which element i is a vector containing
            the list of indices that are connected to i
    '''
    nbrs = NearestNeighbors(n_neighbors=K_NN, algorithm='kd_tree').fit(data)
    distances, indices = nbrs.kneighbors(data)
    return indices



def construct_network_SISG(data, k):
    '''Build a graph with SISG criteria

    [+] data: (N,D) numpy array
            N vectors of D elements (D dimensions, N datapoints)
    [+] k: float
            Density parameter

    [-] indices:
            Vector of N elements (number of datapoints) from which element i is a vector containing
            the list of indices that are connected to i
    '''
    indices = []

    if len(data.T) == 3:
        x, y, z = data.T
    else:
        x, y = data.T
        z = []

    nbrs = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(data)
    distances, _ = nbrs.kneighbors(data)

    distances = distances.T[1]  #Distance to the closest neighbor

    for t in range(len(x)):
        if len(z) == 0:
            distToCurrentPoint = np.sqrt((x-x[t])**2 + (y-y[t])**2)
        else:
            distToCurrentPoint = np.sqrt((x-x[t])**2 + (y-y[t])**2 + (z-z[t])**2)

        tmp = [t]
        tmp2 = np.where((distToCurrentPoint < k*distances[t]) & (distToCurrentPoint > 0))[0]
        tmp = np.hstack((tmp, tmp2))
        indices.append(tmp)

    return indices



def construct_network_fixedLength(data, R):
    '''Link vertices using a fixed length R given as a parameter

    [+] data: (N,D) numpy array
            N vectors of D elements (D dimensions, N datapoints)
    [+] R: float
            fixed radius to link datapoints (same unit as data)

    [-] indices:
        vector of N elements (number of datapoints) from which element i is a vector containing
        the list of indices that are connected to i
    '''
    if len(data.T) < 2 or len(data.T) > 3:
        raise Exception("Don't know how to work with " + str(len(data.T)) + " dimensions")

    kdtree = KDTree(data)
    indices = kdtree.query_radius(data, R)
    indices = indices.tolist()

    return indices



def construct_network_ellipsoid(data, R):
    '''Link vertices using a fixed length R given as a parameter

    [+] data: (N,D) numpy array
            N vectors of D elements (D dimensions, N datapoints)
    [+] R: float
            fixed radius to link datapoints (same unit as data)
    [-] indices:
        vector of N elements (number of datapoints) from which element i is a vector containing
        the list of indices that are connected to i
    '''
    border = 75000
    margin = 500

    if len(data.T) < 2 or len(data.T) > 3:
        raise Exception("Don't know how to work with " + str(len(data.T)) + " dimensions")

    if len(data.T) == 3:
        x, y, z = data.T
    else:
        x, y = data.T
        z = []

    indicesTmp = []
    indices = []
    for t in range(len(x)):
        # 1) Get all points in the radius R
        if len(z) == 0:
            distToCurrentPoint = np.sqrt((x-x[t])**2 + (y-y[t])**2)
        else:
            distToCurrentPoint = np.sqrt((x-x[t])**2 + (y-y[t])**2 + (z-z[t])**2)

        tmp = [t]
        tmp2 = np.where((distToCurrentPoint < R) & (distToCurrentPoint > 0))[0]
        indicesTmp = np.hstack((tmp, tmp2))     #Indices of points in the sphere of radius R

        #if we have enough points to compute a reasonable covariance ellipse
        #and if we are not too close to the border in any direction
        computablePoint = len(indicesTmp) > 15 and x[t] > margin and y[t] > margin and x[t] < border - margin and y[t] < border - margin
        if len(z) != 0:
            computablePoint = computablePoint and z[t] > margin and z[t] < border - margin

        if computablePoint:
            # 2) Perform a eig decomposition to get the shape of the covariance ellipse around the current point
            if len(z) == 0:
                mu = [x[t], y[t]]
                sigma = np.cov(x[indicesTmp], y[indicesTmp])
            else:
                mu = [x[t], y[t], z[t]]
                sigma = np.cov((x[indicesTmp], y[indicesTmp], z[indicesTmp]))

            eigvals, eigvects = np.linalg.eigh(sigma)
            radii = 2*np.sqrt(eigvals)

            # 3) Create an ellipse with volume V and link to all points in the ellipse
            if len(z) == 0:
                r = max(radii) / min(radii)
                u, v = eigvects[:, 0]
                a = R * r
                b = R / r
                if b < a:
                    exc = b
                    b = a
                    a = exc
                th = np.arctan2(v, u)
                isInEllipse = ( (x-mu[0])*np.cos(th) + (y-mu[1])*np.sin(th) )**2 / a**2 + ( (x-mu[0])*np.sin(th) - (y-mu[1])*np.cos(th) )**2 / b**2 <= 1
                tmp2 = np.where(isInEllipse == True)[0]
                indices.append(np.hstack((t, tmp2)))

            else:   #3D case
                indices.append(indicesTmp)
        else:
            indices.append(indicesTmp)

    return indices



def construct_network_ellipsoid_optimR(data, Rit, flagFig=0, verbose=1):
    '''Link vertices using a fixed length R found by optimising the diamater of the graph

    [+] data: (N,D) numpy array
            N vectors of D elements (D dimensions, N datapoints)
    [+] R: vector
            Range to look for the best fit
    [+] flagFig: int
            1 => plot diameter = f(R)    0 => no plot
    [+] verbose: int
            1 => display info of the computation      0 => nothing displayed
    '''
    diam = []
    it = 0

    if verbose:
        print("Finding optimal value of R according to the diamater of the graph")

    for R in Rit:
        it += 1
        indices = construct_network_ellipsoid(data, R)
        G, Edges = compute_network(indices)
        diam.append(G.diameter(directed=False))
        if verbose:
            print("Iteration " + str(it) + "/" + str(len(Rit)) + " -- R = " + str(utility.nice_looking_str(R)) +
              " and D = " + str(utility.nice_looking_str(diam[len(diam) - 1])))

    bestIndex = diam.index(max(diam))
    Ropt = Rit[bestIndex]
    indices = construct_network_ellipsoid(data, Ropt)

    if(flagFig):
        plt.figure()
        plt.plot(Rit, diam, 'o-')
        plt.title('Diameter of the network')
        plt.xlabel('Length')
        plt.ylabel('Diamater')

    return Ropt, max(diam), indices



def construct_network_fixedLength_optimR(data, Rit, weighted=0, flagFig=0, verbose=1):
    '''Link vertices using a fixed length R found by optimising the diamater of the graph

    [+] data: (N,D) numpy array
            N vectors of D elements (D dimensions, N datapoints)
    [+] Rit: vector
            Range to look for the best fit
    [+] flagFig: int
            1 => plot diameter = f(R)    0 => no plot
    [+] verbose: int
            1 => display info of the computation      0 => nothing displayed
    '''
    diam = []
    it = 0

    if verbose:
        print("Finding optimal value of R according to the diamater of the graph")

    for R in Rit:
        it += 1
        indices = construct_network_fixedLength(data, R)
        G, Edges = compute_network(indices)
        weight_edges = [np.linalg.norm(data[e[0]] - data[e[1]]) for e in Edges]

        if weighted == 1:
            diam.append(G.diameter(directed=False, weights=weight_edges))
        else:
            diam.append(G.diameter(directed=False))

        if verbose:
            print("Iteration " + str(it) + "/" + str(len(Rit)) + " -- R = " + str(utility.nice_looking_str(R)) +
              " and D = " + str(utility.nice_looking_str(diam[len(diam) - 1])))

    bestIndex = diam.index(max(diam))
    Ropt = Rit[bestIndex]
    indices = construct_network_fixedLength(data, Ropt)

    if(flagFig):
        plt.figure()
        plt.plot(Rit, diam, 'o-')
        plt.title('Diameter of the network')
        plt.xlabel('Length')
        plt.ylabel('Diamater')

    return Ropt, max(diam), indices



def construct_network_fixedLength_maxGC(data, Rit, flagFig=0):
    '''Link vertices using a fixed length R found by optimising the diamater of the graph

    [+] data: (N,D) numpy array
            N vectors of D elements (D dimensions, N datapoints)
    [+] Rit: vector
            Range to look for the best fit
    [+] flagFig: int
            1 => plot diameter = f(R)    0 => no plot
    '''
    it = 0
    vecN = []

    print("Finding optimal value of R according to the GC of the graph")

    for R in Rit:
        it += 1
        indices = construct_network_fixedLength(data, R)
        G, Edges = compute_network(indices)
        idx = in_giant(G)
        if idx == -1:
            n = 0
        else:
            n = np.sum(idx)
        vecN.append(n)
        print("Iteration " + str(it) + "/" + str(len(Rit)) + " -- R = " + str(utility.nice_looking_str(R)) +
              " and per_in_gc = " + str(utility.nice_looking_str(n/len(data)*100)))

    bestIndex = vecN.index(max(vecN))
    Ropt = Rit[bestIndex]
    indices = construct_network_fixedLength(data, Ropt)

    vecN = np.asarray(vecN)

    if(flagFig):
        plt.figure()
        plt.plot(Rit, vecN/len(data), 'o-')
        plt.title('Percentage of points in GC')
        plt.xlabel('Linking length')
        plt.ylabel('Percentage of points in GC')

    return Ropt, max(vecN), indices


def construct_network_gradient(data, info, K_NN1, K_NN2):
    '''First perform a nearest-neighbors algorithm with K_NN1 (> K_NN2) and then keep the K_NN2 among them with the highest info
[+] data: N vectors of D elements (D dimensions, N datapoints)
[+] info: vector of size N containing the value of interest (to maximize)
[+] K_NN1: number of intermediate neighbors in which we check the value of the info
[+] K_NN2: number of final neighbors
[-] indices: vector of N elements (number of datapoints) from which element i is a vector containing the list of indices that are connected to i'''

    nbrs = NearestNeighbors(n_neighbors=K_NN1, algorithm='kd_tree').fit(data)
    distances, indices = nbrs.kneighbors(data)

    newIndices = []
    for t in range(len(indices)):
        subset = info[indices[t][1:K_NN1]]
        indices_sorted = np.argsort(-subset)
        tmp = [t]
        for j in range(K_NN2-1):
            tmp.append(indices[t][indices_sorted[j]+1])
        newIndices.append(tmp)

    indices = newIndices

    return indices



from sklearn.utils import resample
def construct_network_min_length(data, B, perc, k):
    G_span_bs = []
    boot = []
    idxForBS = range(len(data))

    G_span, edges_span = buildMST(data)
    newG, _ = buildMST(data)
    weight_edges_span = compute_weights(data, edges_span)   #Weights of the MST

    edges_span_new = edges_span.copy()   #Copy of edges
    weight_edges_new = np.copy(weight_edges_span)  #Copy of weigts
    p = np.zeros(len(edges_span))   #Probability of edges
    edgeAlreadyCounted = np.zeros(len(edges_span_new))

    for b in range(B):
        print('Bootstrap sample ' + str(b))
        bs = np.array(resample(idxForBS, replace=False, n_samples=int(len(data)*perc)))
        boot.append(bs)
        data_b = data[bs]

#        t1 = time.time()

        #Compute the DT
#        G_DT_tmp, Edges_DT_tmp = compute_network_delaunay(data_b)

#        #Choose a graph to work with
#        G_tmp = G_DT_tmp
#        Edges_tmp = Edges_DT_tmp
#        weight_edges_tmp = compute_weights(data_b, Edges_tmp)
#
#        #Compute spanning tree
#        G_span_temp = G_tmp.spanning_tree(weights = weight_edges_tmp)
#        edges_span_tmp = G_span_temp.get_edgelist()
#
#        #Save MST and edges
#        G_span_bs.append(G_span_temp)

        #TB 200317
        G_span_temp, edges_span_tmp = buildMST_fast(data_b)
        G_span_bs.append(G_span_temp)

        #Initialized edges counted in bootstrap realization
        edgeAlreadyCounted = np.zeros(len(edges_span_new))

#        t2 = time.time()
#        print('MST took {} seconds to compute'.format(t2-t1))

        #Update the initial MST
        for (i,e) in enumerate(edges_span_tmp):
            n1 = bs[e[0]]
            n2 = bs[e[1]]       #(n1, n2) is the edge of the bs MST in the real dataset

#            tin1 = time.time()

            #if it exists, thats ok
            if (n1, n2) in edges_span_new:
                ed = edges_span_new.index((n1,n2))
                if edgeAlreadyCounted[ed] == 0:
                    p[ed] += 1/B  #We count the road that link the two nodes
                    edgeAlreadyCounted[ed] = 1

#                tin2 = time.time()
#                print('First if took {}'.format(tin2 - tin1))

            else:       #If not, we see if a close path exists or not in the tree
                sh_path = newG.get_shortest_paths(n1, n2, output='epath')
                sh_path = np.array(sh_path)
                costPath = np.sum(weight_edges_new[sh_path[0]])   #On the initial MST
                newCost = np.linalg.norm(data[n1] - data[n2])     #New cost if we add the edge

                #if the cost is not that much a gain
                if newCost > costPath/k:    #costPath < k
                    edgesOk = np.where(edgeAlreadyCounted[sh_path[0]] == 0)[0]
                    ed = sh_path[0][edgesOk]
                    p[ed] += 1/B  #We count the road that link the two nodes
                    edgeAlreadyCounted[ed] = 1

#                    tin3 = time.time()
#                    print('Not known path but ok took {}'.format(tin3 - tin1))
                    continue

                else:
                    edges_span_new.append((n1,n2))
                    p = np.append(p, 0)
                    weight_edges_new = np.append(weight_edges_new, np.linalg.norm(data[n1]-data[n2]))
                    newG = ig.Graph(edges_span_new)
                    edgeAlreadyCounted = np.append(edgeAlreadyCounted, 0)


#                    tin4 = time.time()
#                    print('Not known path to add took {}'.format(tin4 - tin1))

#        t3 = time.time()
#        print('Post-processing {} edges took {} seconds'.format(len(edges_span_tmp), t3-t2))

    return boot, G_span_bs, newG, edges_span_new, p



def getAdjacency_MST(data):
    '''
    Compute the sparse adjacency matrix for the MST of the data set

    [+] data: (N, D) numpy array
            Data set
    [-] MST: (N,N) scipy sparse csr matrix
            Adjacency matrix for the MST of the data
    '''
#    t_ini = time.time()
    tri = Delaunay(data, qhull_options="QJ")   #QJ to force all data points to be a vertex of the DT (or use Qc then tri.coplanar simplicies)
#    t2 = time.time()
#    print('----> Scipy DT took {}' .format(t2 - t_ini))

    Edges = []
    if len(data.T) == 2:
        vec = np.array([[0, 1], [0, 2], [1, 2]])
    else:
        vec = np.array([[0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [2, 3]])
    for i in vec:
        stri = np.sort(tri.simplices[:, i], axis=1)
        Edges.append(stri)

    if len(tri.coplanar) > 0:
        stri = np.sort(tri.coplanar[:,[0,2]], axis=1) #np.sort(tri.coplanar[:, i], axis=1)
        Edges.append(stri)

    Edges = np.vstack(Edges)
    weight_edges = compute_weights(data, Edges)

#    A = getAdjacency_sparse(Edges, len(data), w=weight_edges)

    A = sp.csr_matrix((weight_edges, (Edges.T[0], Edges.T[1])), shape=(len(data), len(data)))
    count = sp.csr_matrix((np.ones(len(Edges.T[0])), (Edges.T[0], Edges.T[1])), shape=(len(data), len(data)))   #To know the number of times we see each edge
    A.data /= count.data  #Get the real weight matrix

    MST = sp.csgraph.minimum_spanning_tree(A)
    return MST


def getAdjacency_bootstrapMST(data, perc=0.75, B=500, display_progress_bar=False):
    '''
    Compute the sparse adjacency matrix for the bootstrap MST of the data set

    [+] data: (N, D) numpy array
            Data set of N points with dimension D
    [+] perc: float
            Percentage of the data set used for the subsamplig (N_b = int(N*perc))
    [+] B: float
            Number of bootstrap resampling
    [-] A: (N,N) scipy sparse csr matrix
            Adjacency matrix for the average MST of the data
    '''
    N = len(data)
    Nb = int(len((data))*perc)+1

    if Nb > N:
        Nb = N

    if display_progress_bar:
        pbar = tqdm(total=B, initial=0, position=0, leave=True)

    rows = []
    cols = []
    values = []
    for i in range(B):
        idx_bs = resample(np.arange(0, len(data)), replace=False, n_samples=Nb)
        dat = data[idx_bs]

        MST = getAdjacency_MST(dat)

        row, col = MST.nonzero()

        rows.append(idx_bs[row])
        cols.append(idx_bs[col])

        rows.append(idx_bs[col])   #Symmetry
        cols.append(idx_bs[row])   #Symmetry
        values.append(np.ones(len(row)*2))

        if display_progress_bar == 1:
            pbar.update(1)

    if display_progress_bar == 1:
        pbar.close()

    rows = np.hstack(rows)
    cols = np.hstack(cols)
    values = np.hstack(values)
    AA = sp.csr_matrix((values, (rows, cols)))

    AA.data = AA.data / B
    return AA


def build_bootstrap_MST(data, B=100, perc=0.75, threshold=0.35, verbose=True):
    if verbose:
        t1 = time.time()

    #Real MST
    MST = getAdjacency_MST(data)
    Edges_mst = np.array(MST.nonzero()).T
    idx = np.where(Edges_mst.T[1] < Edges_mst.T[0])[0]
    Edges_mst = np.delete(Edges_mst, idx, axis=0)

    #BS MST
    AA = getAdjacency_bootstrapMST(data, perc, B, display_progress_bar=verbose)
#    w = AA.copy()    #TB: to get weights
    AA.data[AA.data>threshold] = 1
    AA.data[AA.data<1] = 0

    allEdges = np.array(AA.nonzero()).T
#    weights = np.array(w[AA.nonzero()])[0]   #TB: to get weights
    idx = np.where(allEdges.T[1] < allEdges.T[0])[0]
    allEdges = np.delete(allEdges, idx, axis=0)
#    weights = np.delete(weights, idx)

    #Stack and remove doublons
    allEdges = np.vstack((allEdges, Edges_mst))
    allEdges = np.unique(allEdges, axis=0)

    #Back to a list
    Edges = allEdges
    src = Edges[:,0].tolist()
    trgt = Edges[:,1].tolist()
    Edges = list(zip(src, trgt))

    #Build the graph
    G = ig.Graph(Edges)

    if verbose:
        t2 = time.time()
        print('Time to compute bootstrap MST: {} secs'.format(t2-t1))

    return G, Edges




def buildMST(data, verbose=0):
    '''Compute the MST from the Delaunay Tessellation using igraph

    [+] data: (N,D) numpy array
            N vectors of D elements (D dimensions, N datapoints)

    [-] G_span: igraph object
            Describing the resulting graph
    [-] edges_span: list
            Edges in the graph
    '''

    if verbose:
        print('buildMST -- Starting...')

    ti = time.time()
    G_DT, Edges_DT = compute_network_delaunay(data)                 #Construct network with iGraph
    t2 = time.time()
    if verbose:
        print('buildMST -- DT took {}'.format(t2-ti))

#    t1 = time.time()
    weight_edges = compute_weights(data, Edges_DT)
#    t2 = time.time()
#    print('weights took {}'.format(t2-t1))

#    t1 = time.time()
    G_span = G_DT.spanning_tree(weights=weight_edges)               #Weighted MST
#    t2 = time.time()
#    print('MST took {}'.format(t2-t1))

#    t1 = time.time()
    edges_span = G_span.get_edgelist()
#    t2 = time.time()
#    print('edgelist took {}'.format(t2-t1))

    ttot = time.time()
    if verbose:
        print('buildMST -- Time to build MST = {}'.format(ttot - ti))

    return G_span, edges_span



def buildMST_fast(data, verbose=0):
    '''Compute the MST from the Delaunay Tessellation using igraph (faster than buildMST)

    [+] data: (N,D) numpy array
            N vectors of D elements (D dimensions, N datapoints)
    [+] verbose: boolean
            Displays information

    [-] G_span: igraph object
            Describing the resulting graph
    [-] edges_span: list
            Edges in the graph
    '''

    t_ini = time.time()
    if verbose:
        print('buildMST_fast -- Starting...')

    MST = getAdjacency_MST(data)

    #Build MST from adjacency matrix
    edges_span = np.array(MST.nonzero()).T

    src = edges_span[:,0].tolist()
    trgt = edges_span[:,1].tolist()
    edges_span = list(zip(src, trgt))

    G_span=ig.Graph(edges_span, directed=False)

    t4 = time.time()
    if verbose:
        print('buildMST_fast -- Total time = {}'.format(t4 - t_ini))

    return G_span, edges_span



def build_fullGraph(data):
    '''Compute the MST from the Delaunay Tessellation using igraph (faster than buildMST)

    [+] data: (N,D) numpy array
            N vectors of D elements (D dimensions, N datapoints)

    [-] G_span: igraph object
            Describing the resulting graph
    [-] edges_span: list
            Edges in the graph
    '''

    Edges = []
    idx = np.arange(0, len(data))
    for l in idx:
        for l2 in idx:
            if l2 >= l:
                break
            Edges.append((l,l2))

    G = ig.Graph(Edges, directed=False)

    return G, Edges


def build_KNN(data, k=10):
    '''Compute a KNN graph

    [+] data: (N,D) numpy array
            N vectors of D elements (D dimensions, N datapoints)
    [+] k: int
            Number of neighbors to use

    [-] G_span: igraph object
            Describing the resulting graph
    [-] edges_span: list
            Edges in the graph
    '''
    indices_KNN = construct_network_KNN(data, k)
    G_KNN, Edges_KNN = compute_network(indices_KNN)       #Construct network with iGraph

    Edges_KNN = np.vstack(Edges_KNN)
    Edges_KNN = np.unique(Edges_KNN, axis=0)   #Remove double edges

    #Back to a list
    Edges = Edges_KNN
    src = Edges[:,0].tolist()
    trgt = Edges[:,1].tolist()
    Edges = list(zip(src, trgt))

    #Build the graph
    G_KNN = ig.Graph(Edges)

    return G_KNN, Edges_KNN


def build_KrNN(data, k=15):
    '''Compute a K reciprocal NN graph

    [+] data: (N,D) numpy array
            N vectors of D elements (D dimensions, N datapoints)
    [+] k: int
            Number of neighbors to use

    [-] G_span: igraph object
            Describing the resulting graph
    [-] edges_span: list
            Edges in the graph
    '''
    indices_KNN = construct_network_KNN(data, k)
    G_KNN, Edges_KNN = compute_network(indices_KNN)       #Construct network with iGraph

    Edges_KNN = np.vstack(Edges_KNN)
    Edges_KNN = np.unique(Edges_KNN, axis=0)   #Remove double edges

    A = getAdjacency_sparse(Edges_KNN, len(data))
    A_sym = A.T + A
    A_sym[A_sym==2] = 0
    A_sym[A_sym==4] = 1

    Edges_KNN = np.array(A_sym.nonzero()).T

    #Back to a list
    Edges = Edges_KNN
    src = Edges[:,0].tolist()
    trgt = Edges[:,1].tolist()
    Edges = list(zip(src, trgt))

    #Build the graph
    G_KNN = ig.Graph(Edges)

    return G_KNN, Edges_KNN


def build_RL(data):
    '''Compute the relative neighboroud graph

    [+] data: (N,D) numpy array
            N vectors of D elements (D dimensions, N datapoints)

    [-] G_span: igraph object
            Describing the resulting graph
    [-] edges_span: list
            Edges in the graph
    '''
    G_DT, Edges_DT = compute_network_delaunay(data)                 #Construct network with iGraph
    weight_edges = compute_weights(data, Edges_DT)

    kdt = cKDTree(data)
    ed = []

    for (i,e) in enumerate(Edges_DT):
        cl0, i0 = kdt.query([data[e[0]]], len(data)-1, distance_upper_bound=weight_edges[i])
        cl1, i1 = kdt.query([data[e[1]]], len(data)-1, distance_upper_bound=weight_edges[i])
        cl0 = cl0[0][1::]; i0 = i0[0][1::]
        cl1 = cl1[0][1::]; i1 = i1[0][1::]

        idx0 = np.where(np.isinf(cl0)==False)
        cl0 = cl0[idx0]; i0 = i0[idx0]

        idx1 = np.where(np.isinf(cl1)==False)
        cl1 = cl1[idx1]; i1 = i1[idx1]

        intersec = np.where(np.in1d(i0, i1))[0]
        if len(intersec) == 0:
            ed.append(e)

        #Remove edge if there is a point closer to e[0] or e[1] than dist(e[0], e[1])
#        if cl0 >= weight_edges[i] and cl1 >= weight_edges[i]:
#            ed.append(e)

    G = ig.Graph(ed)
#    indices = G.get_adjlist()

    return G, ed


def build_FL(data, R=150):
    indices_FL = construct_network_fixedLength(data, R)
    G_FL, Edges_FL = compute_network(indices_FL)       #Construct network with iGraph

    return G_FL, Edges_FL


#import trash.homology as homo
#def buildHOPES(data, verbose=1):
#    '''Compute the MST from the Delaunay Tessellation using igraph
#[+] data: N vectors of D elements (D dimensions, N datapoints)
#[-] G_span: igraph object describing the resulting graph
#[-] edges_span: list of all edges in the graph '''
#
#    if verbose > 0:
#        t1 = time.time()
#
#    st = homo.build_alpha_complex(data)
#    edges_mst, per0d = homo.build_minimum_spanning_tree(st)
#    critical_edges, per1d = homo.critical_edges(st)
#    positive_edges, spanGap, _ = homo.positive_edges(critical_edges, per1d, method='Bootstrap', thresh=10,
#                                                  data = data, B=30, eps=1e-2, n_processes=0)   #only used if bootstrap is chosen
#
#    #Adding a condition on the length of the edge
##    weights_positive = compute_weights(data, positive_edges)
##    _, w = homo.bootstrap_MST(data, B=10)
##    w = np.hstack(w)
##    quantile = np.percentile(w, np.linspace(0,100,1000+1))[999] #it is less than 0.1% chance that a length is above this one
##    positive_edges[weights_positive<quantile]
#
#    if verbose > 2:
#        t2 = time.time()
#        print('Total time to compute HoPeS-1D: {} secs'.format(t2-t1))
#
#    edges = []
#    for e in np.vstack((edges_mst, positive_edges)):
#        edges.append((e[0], e[1]))
#
#    G = ig.Graph(edges)
#
#    return G, edges



def removeShortEdges(F, edges_span, length):
    w = compute_weights(F, edges_span)
    edgesToRemove = np.where(w < length)[0]
    while len(edgesToRemove) > 0:
        arr_edges = np.array(edges_span)
        arr_edges = np.delete(arr_edges, edgesToRemove, axis=0)
        indexF_used = np.unique(arr_edges)
        F_new = F[indexF_used, :]

        G_span, edges_span = buildMST(F_new)
        w = compute_weights(F_new, edges_span)
        edgesToRemove = np.where(w < length)[0]
        print(edgesToRemove)
        print(str(len(F) - len(F_new)) + ' nodes removed')
        F = np.copy(F_new)
        if len(F) - len(F_new) == 0:
            break

    return F, G_span, edges_span



def compute_edges(indices):
    '''Compute the list of all edges of the network from indices vectors

    [+] indices: list
            Vector of N elements (number of datapoints) from which element i is a vector containing the list of indices that are connected to i
    [-] Edges: list
            Set of all edges in the graph given as (X,Y) corresponding to a link between points indices X and Y of the dataset
    '''
    edges = []
    for i in range(len(indices)):
        edges += [(i, indices[i][k+1]) for k in range(len(indices[i])-1)]

    return edges



def compute_weights(data, edges):
    '''Compute the weights of edges from data positions

    [+] Edges: list
            Set of all edges in the graph given as (X,Y) corresponding to a link between points indices X and Y of the dataset
    [+] data: (N,D) numpy array
        N vectors of D elements (D dimensions, N datapoints)
    '''

    edges_ar = np.array(edges)
    return np.linalg.norm(data[edges_ar[:,0]] - data[edges_ar[:,1]], axis=1)



def compute_XYZ_of_edges(edges, data):
    '''Compute the XYZ positions of edges with "none" between edges anchor points

    [+] Edges: list
            Set of all edges in the graph given as (X,Y) corresponding to a link between points indices X and Y of the dataset
    [+] data: (N,D) numpy array
        N vectors of D elements (D dimensions, N datapoints)

    [-] Xe, Ye, Ze: list
            3 lists of X,Y,Z positions of beginning and ending of edges, each separated by 'None'
    '''
    wantZ = 0
    if len(data[0]) == 3:
        wantZ = 1

    Xe=[]
    Ye=[]
    Ze=[]
    for e in edges:
        Xe+=[data[e[0]][0],data[e[1]][0], None]     #x-coordinates of edge ends
        Ye+=[data[e[0]][1],data[e[1]][1], None]
        if wantZ:
            Ze+=[data[e[0]][2],data[e[1]][2], None]

    return Xe, Ye, Ze



def compute_XYZ_of_edges_noNone(edges, data):
    '''Compute the XYZ positions of edges with "none" between edges anchor points

    [+] Edges: list
            Set of all edges in the graph given as (X,Y) corresponding to a link between points indices X and Y of the dataset
    [+] data: (N,D) numpy array
        N vectors of D elements (D dimensions, N datapoints)

    [-] Xe, Ye, Ze: list
            3 lists of X,Y,Z positions of beginning and ending of edges
    '''

    wantZ = 0
    if len(data[0]) == 3:
        wantZ = 1

    Xe=[]
    Ye=[]
    Ze=[]
    for e in edges:
        Xe+=[data[e[0]][0],data[e[1]][0]]     #x-coordinates of edge ends
        Ye+=[data[e[0]][1],data[e[1]][1]]
        if wantZ:
            Ze+=[data[e[0]][2],data[e[1]][2]]

    return Xe, Ye, Ze


def compute_network(indices):
    '''Build the graph object

    [+] indices: list
            Vector of N elements (number of datapoints) from which element i is a vector containing the list of indices that are connected to i

    [-] G: igraph object
            Contains the graph structure as an igraph object
    [-] Edges: list
            Set of all edges in the graph given as (X,Y) corresponding to a link between points indices X and Y of the dataset
    '''
    Edges = []
    for i in range(len(indices)):
        Edges += [(i, indices[i][k]) for k in range(len(indices[i])) if i != indices[i][k]]

    G = ig.Graph(Edges, directed=False)

    return G, Edges


def compute_network_delaunay(data):
    '''Build the graph of the Delaunay Tesselation

    [+] indices: list
            Vector of N elements (number of datapoints) from which element i is a vector containing the list of indices that are connected to i

    [-] G: igraph object
            Contains the graph structure as an igraph object
    [-] Edges: list
            Set of all edges in the graph given as (X,Y) corresponding to a link between points indices X and Y of the dataset
    '''
    tri = Delaunay(data)

    Edges = []
    if len(data.T) == 2:
        vec = np.array([[0, 1], [0, 2], [1, 2]])
    else:
        vec = np.array([[0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [2, 3]])
    for i in vec:
        stri = np.sort(tri.simplices[:, i], axis=1)
        Edges.append(stri)

    Edges = np.vstack(Edges)
    Edges = np.unique(Edges, axis=0)

    src = Edges[:,0].tolist()
    trgt = Edges[:,1].tolist()
    Edges = list(zip(src, trgt))

    G=ig.Graph(Edges, directed=False)

    return G, Edges


#==========================
#  Tools to study network
#==========================
def in_giant(G):
    '''Find the nodes in the giant component

    [+] G: igraph object
            Graph information as an igraph object

    [-] indices: list
            Indices of nodes in the giant component
    '''
    cl = G.components()
    cl_sizes = cl.sizes()
    if len(cl_sizes) != 0:
        giant_component_index = cl_sizes.index(max(cl_sizes))
    else:
        return -1
    return [x == giant_component_index for x in cl.membership]


#==========================
#  Multiscale analysis
#==========================
def iterPruning(indices, d, l_cut, edges_span):
    dIter = np.copy(d)
    n1 = len(d)
    n2 = 0
    i = 0
    init = []
    while n2 != n1:
        n1 = len(init)
        idxIter, init, _ = newPruningOnion(indices, dIter, l_cut, init)

        edges_pruned = [e for (i,e) in enumerate(edges_span) if (e[0] in idxIter and e[1] in idxIter)]
        G_pruned = ig.Graph(edges_pruned, directed=False)
        dIter = np.asarray(G_pruned.degree())
        dIter = np.append(dIter, np.zeros(len(d) - len(dIter)))
        n2 = len(init)
        i += 1
    return idxIter


def newPruningOnion(indices, d, l_cut, init=[]):
    deg = np.ones(len(d)) * d   #So that it doesn't modify the parameter d
    indExtr = np.ones(len(d), dtype='int32') * (-1)
    indBif = np.ones(len(d), dtype='int32') * (-1)

    # if we have an init we use it to set those points at -2
    if len(init) != 0:
        indExtr[init] = -2

    bifurcations = np.where(deg > 2)[0]
    extremities = np.where(deg==1)[0]
    indExtr[extremities] = extremities
    for idx in extremities:
#        print(idx)
        node = idx
        adjBif = np.array([])
        while 1:
            deg[node] -= 1
            #Check wether there are adjacent degree with deg > 0 (bifurcations)
            for (i, idNode) in enumerate(indices[node]):
                if idNode in bifurcations:
                    adjBif = [i]  #np.where(deg[indices[node]] > 2)[0]

            if len(adjBif) > 0:
                indBif[np.where(indExtr==idx)[0]] = indices[node][adjBif[0]]
                deg[indices[node][adjBif[0]]] -= 1
                break

            adjNoBif = np.where((deg[indices[node]] == 2))[0]
            if len(adjNoBif) > 0:
                pertinentId = np.where((indExtr[indices[node]] == -1) & (deg[indices[node]] == 2))[0]  #index not crossed yet
                if len(pertinentId) > 0:
                    pertinentId = pertinentId[0]
                    idAdjNoBif = indices[node][pertinentId]
                    indExtr[idAdjNoBif] = idx
            else:
                break
            deg[idAdjNoBif] -= 1    #Decrease the degree of all adjacent nodes
            node = idAdjNoBif

    idxBranches = np.unique(indExtr[indExtr>-1])
    lengthBranches = []
    for idBranch in idxBranches:
        lengthBranches.append(len(np.where(indExtr == idBranch)[0]))

    lengthBranches = np.array(lengthBranches)
    idxBranchesToKeep = np.where(lengthBranches>l_cut)[0]

    if len(idxBranchesToKeep) > 0:
        idxToKeep = np.hstack((np.where(indExtr==-1)[0], np.hstack([np.where(indExtr == i)[0] for i in idxBranches[idxBranchesToKeep]])))
    else:
        idxToKeep = np.where(indExtr==-1)[0]

    idxToThrow = np.linspace(0, len(d)-1, len(d)).astype(int)
    idxToThrow = np.delete(idxToThrow, idxToKeep)

    return idxToKeep, idxToThrow, deg


def onionDecomposition(indices, d, flagFig=1, verbose=1, edges_span=[]):
    core = 1
    currentLayer = 1
    globalLayer = 1
    deg = np.ones(len(d)) * d   #So that it doesn't modify the parameter d

    V = np.ones(len(d))
    coreness = np.zeros(len(d))
    layer = np.zeros(len(d))
    V[np.where(deg == 0)[0]] = 0

    layersCurrentCore = []
    kcores = []
    globalLayers = np.zeros(len(d))
#    edgesInLayer = []

    while np.sum(V) != 0.0 and np.sum(deg>0) != 0:
        nodesInLayer = np.asarray(np.where((deg <= core) & (deg > 0))[0])
        coreness[nodesInLayer] = core
        layer[nodesInLayer] = currentLayer

        layersCurrentCore.append(nodesInLayer)
        globalLayers[nodesInLayer] = globalLayer

        for idx in nodesInLayer:
            deg[indices[idx]] -= 1    #Decrease the degree of all adjacent nodes

        V[nodesInLayer] = 0        #Remove nodes of this layer from
        deg[nodesInLayer] = 0
        currentLayer += 1
        globalLayer += 1

        if np.sum(deg>0) != 0:
            #New core
            if min(deg[deg>0]) >= core + 1:
                core += 1 #min(deg[deg>0])
                currentLayer = 1

                kcores.append(layersCurrentCore)
                layersCurrentCore = []
        if verbose == 1:
            print('Layer ' + str(currentLayer) + ' with ' + str(len(nodesInLayer)) + ' nodes   ' + 'V = ', str(int(np.sum(V))))

    #Add the last core and layer
    if len(layersCurrentCore) != 0:
        kcores.append(layersCurrentCore)
        #Remaining points has to be put in the last layer
        idxLast = np.where(layer == 0)[0]
        layer[idxLast] = currentLayer
        globalLayers[idxLast] = currentLayer
        coreness[idxLast] = core


    fraction = []
    for k in kcores:      #For each cores
            for l in k:     #For each layer of the core
                fraction.append(len(l) / len(d))
    optLayer = -1
    for i in range(len(fraction)-5):
        if fraction[i] == fraction[i+1] and fraction[i] == fraction[i+2] and fraction[i] == fraction[i+3] and fraction[i] == fraction[i+4]:
            optLayer = i+1
            break;

    return coreness, layer, kcores, globalLayers


#==========================
#  Utility computations
#==========================
def getAdjacency(G):
    ''' Compute the adjacency matrix from a igraph object G. This is faster than the igraph routine.

    [+] G: igraph object
            Graph structure

    [-] A: numpy array
            Adjacency matrix
    '''
    n = G.vcount()
    A = np.zeros((n, n))
    edges = G.get_edgelist()
    if not G.is_directed():
        edges.extend([(v, u) for u, v in edges])

    rows, cols = zip(*edges)
    A[rows, cols] = 1
    return A



def getAdjacency_sparse(edges, N, w=[]):
    '''
    Compute the scipy sparse (csc_matrix) adjacency matrix from edge list

    [+] edges: (E,2) numpy array
        Contains indices of nodes to link
    [+] N: float
        Number of nodes in the graph
    [+] w: numpy array
        Contains the weights to put in the adjacency matrix (1 by default)

    [-] A: scipy sparse csc_matrix
        Adjacency (if w = []) or weights matrix

    Note: The graph is assumed undirected with no self loops and repetitions
          Hence, edges list reflects only half of the adjacency matrix.
    '''

    if w == []:
        halfA = sp.csc_matrix((np.ones(len(edges.T[0])), (edges.T[0], edges.T[1])), shape=(N, N))
    else:
        halfA = sp.csc_matrix((w, (edges.T[0], edges.T[1])), shape=(N,N))
#    halfA.data /= halfA.data  #Remove in case of doublons
    A = halfA + halfA.T

    return A



def getDegree(edges, N):
    '''
    Compute degrees of all N nodes in the graph

    [+] edges: (E,2) numpy array
        Contains indices of nodes to link
    [+] N: float
        Number of nodes in the graph

    [-] d: (N,) numpy array
        Contains the degree associated to each node of the graph
    '''

    A = getAdjacency_sparse(edges, N)
    d = np.array(A.sum(axis=0))[0]

    return d
