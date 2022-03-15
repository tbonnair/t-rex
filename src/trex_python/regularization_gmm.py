#For computation
import numpy as np
from scipy.spatial import cKDTree
from scipy import sparse as sp
import pandas as pd

#To handle graphs
import trex_python.networkAlgo as na
import igraph as ig

#Gestion of functions
from tqdm import tqdm
import time

#Some utility functions
import trex_python.utility as utility
import trex_python.GMM_utils as gutil
import h5py

#For preprocessing of data
from sklearn.utils import resample
from sklearn.neighbors import KDTree, NearestNeighbors

#Hiding warnings
import warnings
warnings.filterwarnings("ignore")

# ===========================================================================
# Classes
# ===========================================================================
#class mixture_:
#    def __init__(self):
#        #Parameters for gaussian clusters
#        self.F = 0          #Mean position of gaussian clusters
#        self.sig = 0        #Spatial extension of clusters (defined trough A0)
#        self.pi_k = 0       #Amplitude of Gaussian clusters
#
#        #Parameters for uniform component
#        self.alpha = 0         #Amplitude of the background noise

class regGraph_():
    def __init__(self):
        self.F = 0          #Mean position of gaussian clusters
        self.sig = 0        #Spatial extension of clusters (defined trough A0)
        self.pi_k = 0       #Amplitude of Gaussian clusters
        self.alpha = 0      #Amplitude of the background noise

        self.edges = []     #List of edges of the regularised graph
        self.responsibilities = 0   #Responsibility matrix    #DEBUG

class param_:
    #Class for set of parameters of the algorithm
    def __init__(self):
        #Parameters for parallel computation
#        self.parallel = 0       #To run in parallel
#        self.n_processes = 0    #Number of processes (used if parallel=1)

        #Parameters for denoising operation
        self.denoising = ''     #Redirect to the function to use prior to compute the topology, to remove outliers/noise in the data
        self.denoisingParameter = 0              #Assume only one parameter to denoise

        #Parameters for the graph topology
        self.topology = ''      #Redirect to the function to use to compute the topological constraint of centroids
        self.niter_update_graph = 1  #Number of iterations before updating the graph topology

        #Parameters for regularisation
        self.lam = 0            #Topological constraint
        self.lam_sig = 0        #Shape parameter of the inverse-Gamma prior distribution on variances
        self.lam_pi = 0         #Precision parameter of the Gaussian prior on weights

        #Parameters for regularization
        self.A0 = 0             #Constant used in Silverman's rule
        self.maxIter = 0        #Number max of iteration
        self.eps = 0            #For convergence analysis
        self.verbose_level = 0  #0, 1 or 2 (Minimal traces, mid or all)

        #Parameters for the bootstrap
        self.B = 0
        self.perc = 0

        #Robustness to noise and adaptative local scale
        self.covariance_update = 'fixed'      #'adaptative' or 'fixed'
        self.covariance_type = 'spherical'       #'spherical', 'diag' or 'full'
        self.minSigma = 0             #Minimum variance allowed (regularization on the covariance matrix to avoid collapse)

        #Robustness to noise
        self.background_noise = False #Wether we account for a uniform component or not
        self.domain_area = 0          #Used for the probability of the background noise

        self.ksig = 7              #Maximum sigma for udpate
        
        self.sig = 0            #Initial spatial extension of clusters (defined trough A0)
        self.pi_k = 0           #Initial amplitude of Gaussian clusters
        self.alpha = 0          #Initial amplitude of the background noise

        #Logger for traces
        self.log = ''

    def set(self, data, topology, A0=0.1, lam=1, lam_sig=5, lam_pi=1, denoisingParameter=4, B=100, perc=0.75, maxIter=10,
                              eps=1e-6, verbose=0, ksig=7, covariance_type='spherical',
                              covariance_update='fixed', minSigma=0, background_noise=False, alpha=0,
                              domain_area=0, log='', niter_update_graph=1):
#        self.parallel = parallel
        self.topology = topology
        self.niter_update_graph = niter_update_graph
        self.A0 = A0
        self.lam = lam
        self.denoisingParameter = denoisingParameter
        self.B = B
        self.perc = perc
        self.maxIter = maxIter
        self.eps = eps
#        self.n_processes = n_processes
        self.verbose_level = verbose
        self.ksig = ksig
        self.covariance_type = covariance_type
        self.minSigma = minSigma
        self.log = log
        self.domain_area = domain_area
        self.background_noise = background_noise
        self.alpha = alpha
        self.covariance_update = covariance_update
        self.lam_sig = lam_sig
        self.lam_pi = lam_pi

        self.computeBandwidth(data)  #Update variance of clusters

    def show(self):
        print('\t ========= General =========')
        print('\t logger file: {}'.format(self.log))
        print('\t verbose level: {}'.format(self.verbose_level))
#        print('\t parralel = {}'.format(self.parallel))
#        print('\t n_processes = {}'.format(self.n_processes))
        print('\n \t ========= Topology =========')
        print('\t denoising method = {}' .format(self.denoising))
        print('\t pruning level = {}'.format(self.denoisingParameter))
        print('\t topology function: {}'.format(self.topology))
        print('\t update graph: {}'.format(self.niter_update_graph))
        print('\n \t ========= Regularization =========')
        print('\t A0 = {}'.format(self.A0))
        #print('\t variance = {} (can be updated in functions)'.format(self.sig))
        print('\t lambda = {}'.format(self.lam))
        print('\t lambda_sig = {}'.format(self.lam_sig))
        print('\t lambda_pi = {}'.format(self.lam_pi))
        print('\t maximum number of iterations: {}'.format(self.maxIter))
        print('\t maximum sigma for update: {}'.format(self.ksig))
        print('\t variance type: {}'.format(self.covariance_type))
        print('\t variance update: {}'.format(self.covariance_update))
        print('\t epsilon: {}'.format(self.eps))
        print('\n \t ========= Bootstrap =========')
        print('\t Number of samples B: {}'.format(self.B))
        print('\t Sizes of samples: N*{}'.format(self.perc))
        print('\n')

    def computeBandwidth(self, data):
        '''
        Compute the variance to use for gaussian kernels using the modified Silverman's rule
        [+] data: NxD numpy array
                Matrix containing D-dimensional datapoints
        '''
        sig = np.std(data.T, axis=1)
        minsig = np.min(sig[np.nonzero(sig)])       #Minimum of std in all directions
        h = self.A0 * (1 / (len(data.T)+2)) ** (1 / (len(data.T)+4)) * len(data) ** (-1/(len(data.T)+4)) * minsig

        self.sig = h**2   #Update sigma with h**2


    def setCovariance(self, cov):
        '''
        Set the variance of gaussian clusters manually
        '''
        self.sig = cov


class filaments_:
    ''' Filament class containing information on:
        - nodes: list of nodes indices forming the ridge
        - edges: list of doublets forming edges of the filaments
        - datapoints: list of datapoints indices associated to the filament
        - type: {0:extremity to extremity, 2:extremity to bifurcation , 3:bifurcation to bifurcation}
        - length: the sum of edges weights
        - curvature: 1 - (euclidean length / geodisic length)
        - spatial_uncertainty: associated to each node
        - residuals: residual value of the projection of each associated datapoint to the ridge
        '''
    def __init__(self):
        self.nodes = []             # Set of nodes forming the ridge of the filament
        self.edges = []             # Set of edges forming the ridge of the filament
        self.datapoints = []        # Datapoints assigned with the filament
        self.type = -1              # Type de filament (bif to bif, bif to extr or extr to extr)
        self.length = 0             # Length of the filament
        self.curvature = 0          # Curvature of the filament
        self.spatial_uncertainty = [] # Spatial uncertainty associated to nodes
        self.residuals = []         # Residuals of the projection on filament of the assigned datapoints

    def summary(self):
        print(' Branch of type {} made of {} nodes and {} edges'.format(self.type, len(self.nodes), len(self.edges)))
        print(' Containing {} datapoints'.format(len(self.datapoints)))
        print(' Geodesic length = {}'.format(self.length))
        print(' Curvature = {}'.format(self.curvature))
        if len(self.datapoints) > 0:
            print(' Mean residual value = {}'.format(np.mean(self.residuals)))
    pass

# ===========================================================================
# Internal Functions
# ===========================================================================
def _residualsFilaments(X, mu, edges, fils):
    '''
    -- INTERNAL FUNCTION --
    Update the input 'fils' object with the residuals of the associated galaxies in each filament

    [+] X: NxD numpy array
            Datapoints positions
    [+] F: KxD numpy array
            Gaussian clusters means
    [+] edges: list
            Indices representing connections of gaussian clusters
    [+] sigs: Kx1 numpy array
            Scalar covariances associated to nodes
    [+] fils: list
            Regularization_GMM.filaments_ objects of detected filaments
    '''

    D = X.shape[1]

    #for each position x, project it on the filamnetary structure
    #and find the probability that it has been drawn from that
    pbar = tqdm(total=len(fils), initial=0, position=0, leave=True)
#    residuals = []
    residuals = np.ones(len(X))*(-1)

    arr_edges = np.array(edges)  #From list to numpy array to allow indexing

    for (indexFil, f) in enumerate(fils):
        f.residuals = []
        ed = arr_edges[np.array(f.edges)]
        edg = ed.T

        for (i,p) in enumerate(X[f.datapoints]):
            if D == 2:
                    dist, posNearest = utility.vectorized_pnt2line(np.hstack((p, 0)),
                                                                   np.hstack((mu[edg[0]], np.array([np.zeros(len(edg[1]))]).T)),
                                                    np.hstack((mu[edg[1]], np.array([np.zeros(len(edg[1]))]).T)))
            elif D == 3:
                    dist, posNearest = utility.vectorized_pnt2line(p, mu[edg[0]], mu[edg[1]])

            dist = np.array(dist)
            posNearest = np.array(posNearest)

            argmin = np.argmin(dist)
            d = dist[argmin]

            f.residuals.append(d)
#            residuals.append(d)
            residuals[f.datapoints] = d

        pbar.update(1)

    pbar.close()

    return np.array(residuals)



def sparse_distance_matrix(kdtree_X, X, F, max_distance):
    '''
    Compute a sparse distance matrix from a sklearn.neighbours.KDTree object between positions in X and in F

    [+] kdtree_X: sklearn.neighbours.KDTree object
            KDTree over datapoints positions
    [+] X: NxD numpy array
            Datapoint positions
    [+] F: KxD numpy array
            Gaussian clusters means
    [+] max_distance: (K,) numpy array
            Maximum distance radius around the point i in F

    [-] R: (N,K) scipy.sparse.coo_matrix
            Sparse distance matrix between X and F
    '''

    N = X.shape[0]
    K = F.shape[0]

#    t0 = time.time()
    nbrs, dist = kdtree_X.query_radius(F, max_distance, return_distance=True)
#    t1 = time.time()

    dst = np.hstack(dist)
    cols = np.repeat(np.arange(len(F)), np.array(list(map(len, nbrs))))
    rows = np.hstack(nbrs)

    R = sp.coo_matrix((dst, (rows, cols)), shape=(N, K))

#    t2 = time.time()

#    print('Time for query = {:.3f}'.format(t1-t0))
#    print('Time for post process = {:.3f}'.format(t2-t1))

    return R



#import cython_test as ct
#from scipy.special import gamma  #For laplace distribution
def _computeResponsibilityMatrix(data, F, sig, pi_k, kdtree_X, covariance_type, covariance_update, k_sig=7,
                                 background_noise=False, domain_area=0, alpha=0, Kf=0):
    '''
    -- INTERNAL FUNCTION --
    Compute the responsibility matrix using pairwise distances between elements of X
    and gaussian clusters of means F and variances sig
    Note that all distances above param.ksig*max(np.sqrt(eig(sig))) are set to 0.

    [+] data: NxD numpy array
            Datapoints positions
    [+] F: KxD numpy array
            Positions of graph nodes
    [+] sig: type and size depends on covariance_type and covariance_update
                float               if covariance_type = 'spherical' and covariance_update = 'fixed'
                (K,) numpy array    if covariance_type = 'spherical' and covariance_update = 'adaptative'
                (D,) numpy array    if covariance_type = 'diag' and covariance_update = 'fixed'
                (K,D) numpy array   if covariance_type = 'diag' and covariance_update = 'adaptative'
                (D,D) numpy array   if covariance_type = 'full' and covariance_update = 'fixed'
                (K,D,D) numpy array if covariance_type = 'full' and covariance_update = 'adpatative'
            Spatial extension of clusters, assumes a fixed and scalar variance
    [+] pi_k: scalar or (K,) numpy array
            Amplitudes of gaussian clusters
            float                   if equally_distributed = True
            (K,)                    if equally_distributed = False
    [+] kdtree_X: Scipy cKDTree object
            Input KDTree computed over the data array

    [-] R: NxK sparse csc_matrix
            The normalized responsability matrix
    '''

    D = len(data.T)
    K = len(F)
    N = len(data)

    #=================
    #Spherical covariances
    #=================
    if covariance_type == 'spherical':
#        t0 = time.time()

        if isinstance(sig, np.ndarray):    #Only if covariance_update = True
            max_dist = k_sig*np.sqrt(sig)
#            t1 = time.time()
            R = sparse_distance_matrix(kdtree_X, data, F, max_distance=max_dist)
#            t2 = time.time()
#            print('Time to compute sparse distance matrix = {:.3f}'.format(t2-t1))
        else:
            max_dist = k_sig*np.sqrt(sig)
            kdtree_F = cKDTree(F)
            R = kdtree_X.sparse_distance_matrix(kdtree_F, max_distance=max_dist, output_type='coo_matrix')

#        t1 = time.time()

        if isinstance(sig, np.ndarray):     #Only if covariance_update = True
            R.data = np.exp(-R.data**2 / sig[R.col] / 2)
            R.data = 1/(2*np.pi*sig[R.col])**(D/2)*R.data    #True Gaussian (required if background_noise=True or if sigmas are different)
        else:
#            R.data = np.exp(-R.data / sig)
#            R.data = 1/(2**D)/(np.pi)**((D-1)/2)/sig**(D/2)*gamma((D+1)/2)*R.data    #Laplace distribution
            R.data = np.exp(-R.data**2 / sig / 2)
            R.data = 1/(2*np.pi*sig)**(D/2)*R.data


        if isinstance(pi_k, np.ndarray):    #Only if covariance_update = True
            R.data = pi_k[R.col]*R.data
        else:
            R.data = pi_k * R.data

#        t2 = time.time()

        resp_background = 0
        if background_noise:
            #If background noise, then add a column for another component in the responsibility matrix
            resp_background = np.ones(len(data))*alpha/domain_area
            Rtot = sp.hstack((R, np.array([resp_background]).T))
        else:
            Rtot = R

        #Normalize responsibilities (GMM)
        sc = np.array(Rtot.sum(1).T)[0]
        R.data /= sc[R.row]

        #Normalize reponsabilites (MS)
#        sc = np.array(Rtot.sum(0).T).T[0]
#        R.data /= sc[R.col]

        if background_noise:
            resp_background /= sc

#        R.data = R.data / dist_mat.data * np.sqrt(sig)  #Laplace distribution

#        t3 = time.time()

#        print()
#        print('Time for distance: {:.3f}'.format(t1-t0))
#        print('Time for updating values: {:.3f}'.format(t2-t1))
#        print('Time for bkg and normalize: {:.3f}'.format(t3-t2))

    if covariance_type == 'full':
        resp_background = []
#        R = sp.csc_matrix((N, K))
        rows = []
        cols = []
        vals = []
        for k in range(K):
            invSig = np.linalg.inv(sig[k])
            chol_prec = np.linalg.cholesky(invSig)
            log_probs = np.sum((data @ chol_prec - F[k] @ chol_prec)**2, axis=1)
            probs = np.exp(-0.5*log_probs)
            #probs[probs < 1e-6] = 0
            probs = 1/K*1/(2*np.pi)**(D/2)/(np.linalg.det(sig[k])**0.5)*probs
            r = np.where(probs>0)[0]
            rows.append(r)
            cols.append(np.repeat(k, len(r)))
            vals.append(probs[probs>0])
#            R[:,k] = np.array([probs]).T
        rows = np.hstack(rows)
        vals = np.hstack(vals)
        cols = np.hstack(cols)
        R = sp.coo_matrix((vals, (rows, cols)), shape=(N, K))

        #Normalize responsibilities (GMM)
        sc = np.array(R.sum(1).T)[0]
        R.data /= sc[R.row]

    return R, resp_background #, R_non_norm


def _findIndexWithEntry(tab, v1, v2):
    '''
    Find all indices of element (v1, v2) in table tab:

        [+] tab: Nx2 numpy array
                Array of elements
        [+] v1: float
                Value in the first dimension
        [+] v2: float
                Value in the second dimension

        [-] idx: int
                The indices of the element [v1, v2] in the tab array
    '''

    idx = np.where((tab[:, 0] == v1) & (tab[:,1] == v2))

    return idx


def _findIndexWithEntry_list(tab, v1, v2):
    '''
    Find all indices of element (v1, v2) in table tab:

        [+] tab: list
                list of elements
        [+] v1: float
                Value in the first dimension
        [+] v2: float
                Value in the second dimension

        [-] idx: int
                The indices of the element [v1, v2] in the tab array
    '''

    idx = tab.index((v1, v2))

    return idx

# ===========================================================================
# Public Functions
# ===========================================================================

def regGMM(data, F, param, computeObj=0, display_progress_bar=0, step_update_graph=1):
    '''
    Implements the regularized GMM algorithm (Bonnaire et al., 2019) using Expectation-Maximization solver:
        handle N-D computation and uses sparse matrices to optimize RAM usage

    [+] data: NxD numpy array
            Array of D-dimensional datapoints
    [+] F: KxD numpy array
            Initial positions of clusters centers
    [+] param: Instance of 'param_' class
            The set of parameters to use
    [+] display_progress_bar=0: float
            If set to 1, a progress bar is displayed
    [+] step_update_graph: int
            Number of EM iterations before updating the graph prior
    [+] computeObj: boolean (0 or 1)
            Wether or not compute the full log-posterior (computationally costly)

    [-] F: KxD numpy array
            Updated set of clusters centers
    [-] edges_span: list of doublets
            Set of edges of the optimal regularized MST
    [-] objectives: numpy array
            Set of objective values at each iteration
            Note: costly operation
    '''

#    #=====================TEST=========================
#    if param.covariance_type == 'diag':
#        D = len(data.T)
#        if param.covariance_update == 'adaptative':
#            sig = np.repeat(sig, len(F))
#            sig = np.repeat(sig, D).reshape(len(F), D)
#        else:
#            sig = np.repeat(sig, D)
#    #==================================================

    #Get value of the parameters
    lam = param.lam
    maxT = param.maxIter
    eps = param.eps
    verbose = param.verbose_level
    covariance_type = param.covariance_type
    covariance_update = param.covariance_update

    time_start = time.time()

    COMPUTE_OBJ = computeObj

    K, D = F.shape
    min_var = param.minSigma
    alpha = param.alpha

    if isinstance(param.pi_k, np.ndarray):
        pi_k = param.pi_k
    else:
        if param.pi_k == 0:
            pi_k = (1-alpha)/K
        else:
            pi_k = param.pi_k

    if isinstance(param.sig, np.ndarray):
        lam2 = param.lam
        sig = param.sig
    else:
        lam2 = lam/param.sig
        if covariance_update == 'adaptative':
            sig = np.ones(K)*param.sig
        else:
            sig = param.sig

    #Compute graph topology from current positions of centroids
#    G_span, edges_span = na.buildMST_fast(F) #param.topology(F)
    G_span, edges_span = param.topology(F)
#    lam_saved = lam
    #lam = 0

#    w = na.compute_weights(F, edges_span)
    #Laplacian of the graph
    adj = na.getAdjacency_sparse(np.array(edges_span), K)#, w=w)
    L = sp.csgraph.laplacian(adj, normed=False)

    #Degrees
    dk = na.getDegree(np.array(edges_span), len(F))

    #Used to boost the computation of the responsability matrix R
    if covariance_type == 'spherical':
        if covariance_update == 'fixed':
            kdtree = cKDTree(data)
        elif covariance_update == 'adaptative':
            kdtree = KDTree(data)

    elif covariance_type == 'diag' and covariance_update == 'fixed':
        kdtree = cKDTree(data/np.sqrt(sig))

    #Objectives values
    mov = 1e30
    vecMov = []
    objectives = []

    t1 = 0
    t2 = 0

    #Initialise progress bar
    if display_progress_bar == 1:
        pbar = tqdm(total=maxT, initial=0, position=0, leave=True)

    t = 0
    oldF = F.copy()
    old_obj = 0

    while t < maxT: #np.linalg.norm(mov) > eps:
        if t > 0 and np.max(np.linalg.norm(F-oldF, axis=1)) < eps:
            break

        if verbose >= 1:
            print('iteration ' + str(t) + '/' + str(maxT-1))

        oldF = np.copy(F)       #Save previous result for convergence
        time_it = time.time()

        #Evaluate responsibility matrix
        t11 = time.time()
        R, R_bn = _computeResponsibilityMatrix(data, F, sig, pi_k, kdtree, covariance_type, covariance_update,
                                               background_noise=param.background_noise, domain_area=param.domain_area,
                                               alpha=alpha, k_sig=param.ksig)

        t2 += time.time() -  t11

        if verbose >= 1:
            time_R = time.time()
            print('R computed in {} seconds' .format(time_R - time_it))

        tmp = np.array(R.sum(0))
        if len(tmp.shape) > 1:
            tmp = tmp[0]

        tmp[tmp<eps] = eps
        Lambda = sp.diags(tmp)
        tmp /= sig
        Lambda2 = sp.diags(tmp)

        if verbose >= 1:
            time_Lam = time.time()
            print('Lambda computed in {} seconds' .format(time_Lam - time_R))

        t0 = time.time()
        #Update positions of projected points
        if covariance_update == 'adaptative':
            A2 = L*lam2 + Lambda2
            V = R.copy()
            V.data = V.data / sig[V.col]

            F, sig0, pi_k, alpha = gutil.M_step(data, V, R, A2, Lambda, update_pos_only=False,
                                                    background_noise=param.background_noise, resp_background=R_bn)

            lam4 = param.lam_pi
            pi_k = (pi_k + lam4 * (1-alpha) / len(F)) / (1+lam4)

            #Inverse-Gamma prior on sigma
            lam3 = param.lam_sig #/param.sig
            nk = Lambda.data[0]
            aa = adj.copy()
            aa.data[aa.data>0] = 1
            dk = np.sum(aa, axis=0).A[0]
            sig_nbrs = (aa @ sig) / dk[np.newaxis, :]
            sig = np.array((sig0 + 4 * lam3 * sig_nbrs / D / nk) / (1 + 4*lam3/D/nk))[0]

            sig[sig < min_var] = min_var            #Regularisation on covariances to avoid collapse to 0 locally

        elif covariance_update == 'fixed':
            A2 = L*lam2 + Lambda2
            V = R.copy()
            V.data = V.data / sig
            F = gutil.updatePositionsGMM(data, V, A2)

            #Equivalent scalar solve of the M-step for F (slower)
#            deg = na.getDegree(np.array(edges_span), len(F))
#            ff = []
#            Rcsc = R.tocsc()
#            for k in range(len(F)):
#                aa = np.array(adj[k,:].todense())[0]
#                reg = aa.T@F
#                num = Rcsc[:,k].T@data/sig + 2*lam2/2*reg
#                den = Lambda.data[0][k]/sig + 2*lam2/2*deg[k] #- 2*lam2/2*reg#- 2*lam/2*deg[k]
#                ff.append(num[0]/den)
#
#            ff = np.array(ff)
#            F = ff.copy()
#            print('')
#            print(F)
#            print('---------------')
#            print(ff)
#            print(np.allclose(F, ff, 1e-30))

            if param.background_noise:
                alpha = 1/len(data) * np.sum(R_bn)
                pi_k = (1-alpha)/K
            
        if verbose >= 1:
            time_update = time.time()
            print('New positions computed in {} seconds'.format(time_update - time_Lam))

        if verbose >= 1:
            print('New F computed in {} seconds'.format(time.time() - time_it))

        t1 += time.time() - t0
        
#        if t < maxT:
        if t%step_update_graph == 0 and t > 0 or t<=3:
            #Put lambda to its value
            #lam = lam_saved
            
            #Compute the new graph
            G_span, edges_span = param.topology(F)

#            w = na.compute_weights(F, edges_span)
            #Laplacian of the graph
            adj = na.getAdjacency_sparse(np.array(edges_span), K)#, w=w)
            L = sp.csgraph.laplacian(adj, normed=False)

            #Degrees
            dk = na.getDegree(np.array(edges_span), len(F))

            if verbose >= 1:
                time_topo = time.time()
                print('Topology computed in {} seconds'.format(time_topo - time_update))

        t+=1

        if display_progress_bar == 1:
            #Update progress bar
            pbar.update(1)

        if COMPUTE_OBJ:
            #Compute the objective function
            obj = gutil.evaluateObjective(data, F, V, Lambda2, L, lam2, sig, pi_k,
                                          background_noise=param.background_noise, alpha=alpha, domain_area=param.domain_area)
            objectives.append(obj)

        mov = np.linalg.norm(F-oldF)
        vecMov.append(mov)
        if verbose >= 1:
            print('Total displacement = ' + str(mov))

        if mov < eps:
            #If convergence is reached, break the loop
            print('Converged: Displacement')
            break

        if t > 0 and COMPUTE_OBJ:
            if np.abs(obj - old_obj) < eps:
                #If convergence is reached, break the loop
                print('Converged: Likelihood is not moving')
                break

            old_obj = obj

    if verbose >= 1:
        print('T-ReX done! Time elapsed: {} seconds'.format(time.time()-time_start))

    vecMov = np.asarray(vecMov)
    if COMPUTE_OBJ==0:
        objectives = vecMov

    if display_progress_bar == 1:
        pbar.close()

    #Write final estimations into regGraph object
    regGraph = regGraph_()
    regGraph.sig = sig
    regGraph.alpha = alpha
    regGraph.pi_k = pi_k
    regGraph.F = F
    regGraph.edges = edges_span
    regGraph.responsibilities = (R, R_bn)

    print('Time in E-step: {:.3f}'.format(t2))
    print('Time in M-step: {:.3f}'.format(t1))

    return regGraph, objectives, vecMov, adj


def bootstrap(data, param, display_progress_bar=0):
    '''
    Computes param.B realisations of regularised GMM with param.perc*N randomly chosen datapoints

    [+] data: NxD numpy array
            Array of D-dimensional datapoints
    [+] param: Instance of 'param_' class
            The set of parameters to use
    [+] display_progress_bar=0: float
            If set to 1, a progress bar is displayed

    [-] allF: list
            Contains all numpy arrays of centroids positions of the param.B realizations
    [-] allEdges: list
            Contains all lists of edges of the param.B realizations
    '''

    #Get parameters
    B = param.B
    perc = param.perc
    verbose = param.verbose_level

    if verbose >= 1:
        print('\n Computing bootstrap with current parameters:')
        param.show()

    #Initialization
    idxForBS = range(len(data))
    allF = []
    allEdges = []

    if len(data.T) == 2:
        data_temp = np.vstack((data.T, np.zeros(len(data))))
        data_temp = data_temp.T
    else:
        data_temp = np.copy(data)

    Nb = int(len(data)*perc)

    #Initialise progress bar
    if display_progress_bar == 1:
        pbar = tqdm(total=B, initial=0, position=0, leave=True)

    for b in range(B):
        boot = resample(idxForBS, replace=False, n_samples=Nb)
        boot = np.asarray(boot)

        F, opt_edges = getRegularisedGraph(data, boot, param)

        allF.append(F)
        allEdges.append(opt_edges)

        #Update progress bar
        if display_progress_bar == 1:
            pbar.update(1)

#        if(int(b/B*100)%10 == 0 and int(b/B*100) != lastperc):
#            print(str(int(b/B*100)) + '% completed ({} seconds passed)' .format(time.time()-t_ini))
#            lastperc = int(b/B*100)

    if display_progress_bar == 1:
        pbar.close()

#    if verbose >= 0:
#        print('Bootstrap done! Time elapsed: {} seconds'.format(time.time()-t_ini))

    return allF, allEdges



def pruning(G, param):
    '''
    Compute the pruning of the graph G by iteratively removing nodes standing at branches extremities.
    The graph is pruned at a level param.denoisingParameter.

    [+] G: igraph object
            The graph built with nodes F
    [+] param: Instance of 'param_' class
            The set of parameters to use

    [-] idx: numpy array
            Indices of the nodes to keep to get the pruned graph
    '''
    l_cut = param.denoisingParameter

    if l_cut >= 0:
        indices_span = G.get_adjlist()
        d_span = np.asarray(G.degree())

#        t1 = time.time()
        coreness, layers, kcores, globalLayers = na.onionDecomposition(indices_span, d_span, 0, 0)    #no figures and verbose
#        print('Onion in {}'.format(time.time() - t1))

        idx = np.where(globalLayers > l_cut)[0]   #layers

    return idx



def getRegularisedGraph(data, idxBoot, param):
    '''
    Compute the regularised graph through regGMM function using only data[idxBoot] datapoints:
        1) Compute graph topology
        2) Prune the graph
        3) Update the bandwidth
        4) Call regGMM to get a smooth version of the pruned graph

    [+] data: NxD numpy array
            Array of D-dimensional datapoints
    [+] idxBoot: Mx1 numpy array
            Array of M (<=N) indices from data to use for the computation
    [+] param: Instance of 'param_' class
            The set of parameters to use

    [-] F: KxD numpy array
            Updated set of clusters centers
    [-] opt_edges: list of doublets
            Set of edges of the optimal regularized MST
    '''
    t_ini = time.time()

    #Get parameters for the computation
    data_b = data[idxBoot]
    verbose = param.verbose_level

    #Compute the graph topology
    G_temp, _ = param.topology(data_b)

    if verbose >= 2:
        t_mst = time.time()
        print('Time for computing graph: {}' .format(t_mst-t_ini))

    #Pruning MST
    if param.denoisingParameter > 0:
        idx_temp = pruning(G_temp, param)
    else:
        idx_temp = np.arange(0, len(data_b))

    if verbose >= 2:
        t_pruning = time.time()
        print('Time for pruning: {}' .format(t_pruning-t_mst))

    #Find smoothed skeleton of the BT sample
    X_skel = np.copy(data_b[idx_temp])
#    F_skel = np.copy(data_b[idx_temp])
    F_skel = np.copy(X_skel)
#    F_skel = X_skel[np.sort(np.random.choice(len(X_skel), 5000, replace=False))] #[idx]

    #Compute the variance sigma for clusters' extension
#    param.computeBandwidth(data=data)

    #Regularized graph
    F = []
    opt_edges = []
    try:
        regGraph, _, _, _ = regGMM(X_skel, F_skel, param, display_progress_bar=0)
        F = regGraph.F
        opt_edges = regGraph.edges
    except Exception as e:
        print(e)

    if verbose >= 2:
        t_reg = time.time()
        print('Time for regularization: {}' .format(t_reg-t_pruning))

    return F, opt_edges



def probabilityMap(data, param, centroids, edges, Nx, bins=[]):
    '''
    Build the probability map with a Nx resolution from datapoints data and computed given centroids and edges
    Edges are part of the map and must be sampled before stacking histograms

    [+] data: NxD numpy array
            Array of D-dimensional datapoints
    [+] param: Instance of 'param_' class
            The set of parameters to use
    [+] centroids: NbxD numpy array
            Set of B centroids positions computed during the bootstrap
    [+] edges: list
            List of B lists of edges obtained during each realization
    [+] Nx: dx1 numpy array
             Vector containing resolution of the resulting probability map in each dimension
    [+] bins: list of numpy arrays (OPTIONAL)
            Predefined set of bins on which to compute the map

    [-] probMap: list
            Element 0 of the list is the Nx[0]xNx[1]x...xNx[D] numpy array of the D dimensional histogram
            Element 1 of the list is the corresponding bins
    '''

    N, D = data.shape   #Get shape of data
    boundingBox, sizeBox = utility.get_boundingBox(data)
    sampling = (sizeBox/max(Nx))

    #Build the grid for histograms if not already given
    if len(bins) == 0:
        bins = []
        for d in range(D):
            bins.append(np.mgrid[min(data.T[d]):max(data.T[d]):(max(data.T[d])-min(data.T[d]))/Nx[d]])

    if param.verbose_level >= 0:
        print('Computing binary histograms')
        t1 = time.time()

    i = 0
    temp = np.histogramdd(centroids[0], bins=bins)[0]  #To get the shape
    hh = np.zeros(temp.shape)       #Initialize the hh
    for (i, ed) in enumerate(edges):
        interpolateF = centroids[i]
        test = np.array(ed)
        spaces = np.linalg.norm(interpolateF[test[:,0]]-interpolateF[test[:,1]], axis=1) / sampling * 5
        interpolateF = utility.create_ranges(interpolateF[test[:,0]], interpolateF[test[:,1]], spaces)

        temp = np.histogramdd(interpolateF, bins=bins)[0]
        temp[temp>0]=1      #binary histogram
        hh += temp/len(centroids)

        if param.verbose_level >= 0:
            if int(i/len(edges)*100)%10 == 0:
                t2 = time.time()
                print(str(int(i/len(edges)*100)) + '% done (' + str(t2-t1) + 'secs passed)')

    probMap = (hh, bins)

    return probMap



def filamentFinding(data, regGraph, param, key='weight', type='all'):
    ''' Filament identification for a graph G with a tree structure assumed. Three types of filaments exists:
               type 1 : extremity to extremity
               type 2 : extremity to bifurcation
               type 3 : bifurcation to bifurcation

    [+] data: NxD numpy array
            Array of D-dimensional datapoints
    [+] regGraph: regGraph_ object
            Contains regularized graph information (model parameters)
    [+] param: Instance of 'param_' class
            The set of parameters to use
    [+] key: String
            key for weights to use in the igraph object
    [+] type: String 'all' or anything else
            Type of filaments  o extract ('all' => all; anything else => 1 and 2 only)

    [-] filaments: list
            Set of filaments_ class instances
    '''

    t0 = time.time()

    F = regGraph.F
    edges = regGraph.edges

#    N = len(data)
    K = len(F)

    #Build an igraph object from edges and nodes
    G = ig.Graph(edges, directed=False)
    d = np.asarray(G.degree())
    d = np.append(d, np.zeros(len(F.T[0]) - len(d)))
    w = na.compute_weights(F, edges)
    G.es.set_attribute_values('weight', w)

    verbose = param.verbose_level
    filaments = []
    indices = G.get_adjlist()
    edges_array = np.array(edges)
    d = np.asarray(G.degree())
    affil_clusters = np.zeros(len(F))
    alreadyUsed = np.zeros(len(F))

    dok = sp.csc_matrix((np.arange(0, len(edges_array)), (edges_array[:,0], edges_array[:,1])), shape=(K,K)).todok()

    try:
        w = np.asarray(G.es.get_attribute_values(key))
    except:
        w = np.ones(len(edges))

    #Step 1: find the topological class of each nodes (Junction, bifurcation or extremity)
    node_topo = np.zeros(len(d))
    node_topo[d==1] = 1  #Extremity
    node_topo[d==2] = 2  #Junction
    node_topo[d>2] = 3  #Bifurcation

    extremities = np.where(node_topo == 1)[0]
    bifurcations = np.where(node_topo == 3)[0]

    # 1 single loop, exceptional case
    if len(extremities) == 0 and len(bifurcations) == 0:
        print('\n Exceptional case')
        junctions = np.where(node_topo == 2)[0]
        filament = filaments_()
        length = 0
        prev_node = -1

        j = junctions[0]     #Take a node
        filament.nodes.append(j)
        jIni = j
        jNext = indices[j][0]
        try:
            idx_edge = edges.index((jNext, j))
        except:
            idx_edge = edges.index((j, jNext))

        length = w[idx_edge]
        filament.edges.append(idx_edge) #edges[idx_edge])
        prev_node = j
        j = jNext

        while j != jIni:
            filament.nodes.append(j)
            idx = np.where(np.asarray(indices[j]) != prev_node)[0][0]
            jNext = indices[j][idx]
            try:
                idx_edge = edges.index((jNext, j))
            except:
                idx_edge = edges.index((j, jNext))
            length += w[idx_edge]
            filament.edges.append(idx_edge) #edges[idx_edge])

            prev_node = j
            j = jNext

        filament.nodes = np.array(filament.nodes)
        filament.type = 0
        filament.length = length
        filament.curvature = 1 - np.linalg.norm(F[filament.nodes[0]] - F[filament.nodes[len(filament.nodes)-1]])/length
        filament.spatial_uncertainty = np.zeros(len(filament.nodes))
        filaments.append(filament)
        return filaments

    if verbose > 1:
        t1 = time.time()
        print('\nfilamentFinding -- step 1 executed in {:.2f} secs'.format(t1 - t0))

    #Step 2: find filaments defines as segments of different types:
    #           type 1 : extremity to extremity
    #           type 2 : extremity to bifurcation
    #           type 3 : bifurcation to bifurcation

    #First looking for type 1 and 2
    for e in extremities:
        filament = filaments_()
        filament.nodes.append(e)
        i = indices[e][0]     #there should be only 1 element for extremities
        length = 0
        prev_node = e

        #TEST
        alreadyUsed[e] = 1

        while node_topo[i] != 1 and node_topo[i] != 3: #(not(i in bifurcations) and not(i in extremities)):
            filament.nodes.append(i)
#            idx_edge = _findIndexWithEntry(edges_array, min(prev_node, i), max(prev_node, i))[0][0]
            idx_edge = dok[min(prev_node, i), max(prev_node, i)]   #TB 200316: faster

            #TEST
            alreadyUsed[i] = 1

            length += w[idx_edge]
            idx = np.where(np.asarray(indices[i]) != prev_node)[0][0]
            prev_node = i       #Previous node
            i = indices[i][idx]  #There should be 2 elements for junction so we take the node we don't come from
            filament.edges.append(idx_edge) #edges[idx_edge])


        #Adding the last node (bifurcation or extremity)
#        idx_edge = _findIndexWithEntry(edges_array, min(prev_node, i), max(prev_node, i))[0][0]
        idx_edge = dok[min(prev_node, i), max(prev_node, i)]   #TB 200316: faster
        length += w[idx_edge]
        filament.nodes.append(i)
        filament.edges.append(idx_edge) #edges[idx_edge])
        prev_node = i

        typeOfFil = -1
        if node_topo[prev_node] == 1: #prev_node in extremities:
            typeOfFil = 1
        elif node_topo[prev_node] == 3: #prev_node in bifurcations:
            typeOfFil = 2

        #Save the filament of type 1 or 2 (respectively ext to ext and ext to bif)
        filament.nodes = np.array(filament.nodes)
        filament.type = typeOfFil
        filament.length = length
        filament.curvature = 1 - np.linalg.norm(F[filament.nodes[0]] - F[filament.nodes[len(filament.nodes)-1]])/length
        filament.spatial_uncertainty = np.zeros(len(filament.nodes))

        affil_clusters[filament.nodes] = len(filaments)
        filaments.append(filament)

    if verbose>1:
        t2 = time.time()
        print('filamentFinding -- step 2 executed in {:.2f} secs'.format(t2 - t1))
        print('\t' + str(len(filaments)) + ' filaments of type 1 or 2 found')

    if type == 'all':
        #Then looking for type 3 (bif to bif)
        cnt = 0
        for b in bifurcations:
            for p in indices[b]:        #For all possible path at the bifurcation

                if node_topo[p] != 3 and alreadyUsed[p]:
                    continue;   #If the point is not a bifurcation but already used, then this path is already encoded

                filament = filaments_()
                filament.nodes.append(b)
                length = 0
                prev_node = b
                alreadyUsed[p] = 1

                i = p
                while node_topo[i] != 1 and node_topo[i] != 3: #(not(i in bifurcations) and not(i in extremities)):
                    filament.nodes.append(i)
#                    idx_edge = _findIndexWithEntry(edges_array, min(prev_node, i), max(prev_node, i))[0][0]
                    idx_edge = dok[min(prev_node, i), max(prev_node, i)]   #TB 200316: faster

                    length += w[idx_edge]
                    idx = np.where(np.asarray(indices[i]) != prev_node)[0][0]
                    prev_node = i       #Previous node
                    i = indices[i][idx]  #There should be 2 elements for junction so we take the node we don't come from
                    filament.edges.append(idx_edge) #edges[idx_edge])
                    alreadyUsed[i] = 1

                #Adding the last node (bifurcations or extremities)
#                idx_edge = _findIndexWithEntry(edges_array, min(prev_node, i), max(prev_node, i))[0][0]
                idx_edge = dok[min(prev_node, i), max(prev_node, i)]   #TB 200316: faster
                length += w[idx_edge]
                filament.nodes.append(i)
                filament.edges.append(idx_edge) #edges[idx_edge])
                prev_node = i

                typeOfFil = -1
                if node_topo[prev_node] == 3: #prev_node in bifurcations:       #Save it as a filament only if it ends on a bifurcation
                    filament.type = 3
                    filament.length = length
                    filament.curvature = 1 - np.linalg.norm(F[filament.nodes[0]] - F[filament.nodes[len(filament.nodes)-1]])/length
                    filament.spatial_uncertainty = np.zeros(len(filament.nodes))
                    affil_clusters[filament.nodes] = len(filaments)
                    filaments.append(filament)
                    cnt += 1
        if verbose>1:
            print('\t' + str(cnt)  + ' filaments of type 3 found')
            t3 = time.time()
            print('filamentFinding -- Step 3 executed in {:.2f} secs'.format(t3 - t2))

    dataSeg = np.ones(len(data))*(-1)

    if verbose > 1:
        print('\nfilamentFinding -- Starting data segmentation')

    #Get the segmentation of datapoints in the different branches
    if param.covariance_type == 'spherical':
        if param.covariance_update == 'adaptative':
            kdtree = KDTree(data)
        elif param.covariance_update == 'fixed':
            kdtree = cKDTree(data)
    elif param.covariance_type == 'diag':
        kdtree = cKDTree(data/np.sqrt(param.sig))

    R, R_bn = _computeResponsibilityMatrix(data, F, regGraph.sig, regGraph.pi_k, kdtree, param.covariance_type, param.covariance_update, k_sig=7,
                                     background_noise=param.background_noise, domain_area=param.domain_area, alpha=regGraph.alpha)
#    if param.background_noise:
#        R = sp.hstack((R, np.array([R_bn]).T))

    zi = R.argmax(axis=1)
    if len(zi.shape)>1:
#        zi[zi == len(F)+1] = -1                    #Remove datapoints associated to background noise
        zi[np.where(R.sum(axis=1) == 0)[0]] = -1    #Datapoints that are above k sigma for all clusters are removed
        zi = np.array(zi.T)[0]
    else:
#        zi[zi == len(F)+1] = -1                    #Remove datapoints associated to background noise
        zi[R.sum(axis=1) == 0] = -1                 #Datapoints that are above k sigma for all clusters are removed

    #If background component dominates, reject it
#    zi[R_bn > 0.5] = -1
    zi[R_bn > np.array(R.sum(axis=1)).T[0]] = -1

    if verbose > 1:
        t4 = time.time()
        print('filamentFinding -- responsibilities computed in {:.2f} secs'.format(t4-t3))

    #zi is the affiliation of datapoints to Gaussian clusters
    #Since we have the affiliation of clusters to filament, we can link both information
    #to get the affiliation of datapoints to filaments
    for (indexFil, f) in enumerate(filaments):
        #Return indices of zi that have a non-null intersection with nodes of the considered filaments
        idx = np.argwhere(np.in1d(zi, f.nodes)).T[0]
        dataSeg[idx] = indexFil
        f.datapoints = idx

    dataSeg.astype(int)

    if verbose > 1:
        t5 = time.time()
        print('\nfilamentFinding -- Tagged assignations in {:.2f} secs'.format(t5 - t4))
        print('filamentFinding -- Computing residuals...')

    res = _residualsFilaments(data, F, edges, filaments)

    if verbose > 1:
        t6 = time.time()
#        print('\n FilamentFinding -- Residuals computed in {:.2f} secs'.format(t6 - t5))

    if verbose > 1:
        print('\nfilamentFinding -- Exit after {:.2f} secs'.format(t6 - t0))

    return filaments, dataSeg, affil_clusters, res



def assign_labels(data, regGraph, filaments, param):
    '''
    Assign a label to a set of input data according a graph and a set of extracted branches

    [+] data: NxD numpy array
            Array of D-dimensional datapoints
    [+] regGraph: regGraph_ object
            Contains regularized graph information (model parameters)
    [+] filaments: list
        Set of filaments_ class instances
    [+] param: Instance of 'param_' class
            The set of parameters to use

    [-] labels: (N,) numpy array
            Contains labels (index of filament) assigned to the ith input data
    '''

    F = regGraph.F

    if param.covariance_type == 'spherical':
        if param.covariance_update == 'adaptative':
            kdtree = KDTree(data)
        elif param.covariance_update == 'fixed':
            kdtree = cKDTree(data)

    R, R_bn = _computeResponsibilityMatrix(data, F, regGraph.sig, regGraph.pi_k, kdtree,
                                           param.covariance_type, param.covariance_update, k_sig=3,
                                           background_noise=param.background_noise,
                                           domain_area=param.domain_area, alpha=regGraph.alpha)
#    if param.background_noise:
#        R = sp.hstack((R, np.array([R_bn]).T))

    zi = R.argmax(axis=1)
    if len(zi.shape)>1:
#        zi[zi == len(F)+1] = -1                    #Remove datapoints associated to background noise
        zi[np.where(R.sum(axis=1) == 0)[0]] = -1    #Datapoints that are above k sigma for all clusters are removed
        zi = np.array(zi.T)[0]
    else:
#        zi[zi == len(F)+1] = -1                    #Remove datapoints associated to background noise
        zi[R.sum(axis=1) == 0] = -1                 #Datapoints that are above k sigma for all clusters are removed

    #If background component dominates, reject it
#    zi[R_bn > 0.5] = -1
    zi[R_bn > np.array(R.sum(axis=1)).T[0]] = -1

    #Create label vector
    labels = np.ones(len(data))*(-1)

    for (indexFil, f) in enumerate(filaments):
        #Return indices of zi that have a non-null intersection with nodes of the considered filaments
        idx = np.argwhere(np.in1d(zi, f.nodes)).T[0]
        labels[idx] = indexFil

    return labels.astype(int)


def in_out(data, regGraph, param):
    '''
    '''

    F = regGraph.F

    if param.covariance_type == 'spherical':
        if param.covariance_update == 'adaptative':
            kdtree = KDTree(data)
        elif param.covariance_update == 'fixed':
            kdtree = cKDTree(data)

    R, R_bn = _computeResponsibilityMatrix(data, F, regGraph.sig, regGraph.pi_k, kdtree,
                                           param.covariance_type, param.covariance_update, k_sig=7,
                                           background_noise=param.background_noise, domain_area=param.domain_area,
                                           alpha=regGraph.alpha)
#    if param.background_noise:
#        R = sp.hstack((R, np.array([R_bn]).T))

    zi = R.argmax(axis=1)
    if len(zi.shape)>1:
#        zi[zi == len(F)+1] = -1                    #Remove datapoints associated to background noise
        zi[np.where(R.sum(axis=1) == 0)[0]] = -1    #Datapoints that are above k sigma for all clusters are removed
        zi = np.array(zi.T)[0]
    else:
#        zi[zi == len(F)+1] = -1                    #Remove datapoints associated to background noise
        zi[R.sum(axis=1) == 0] = -1                 #Datapoints that are above k sigma for all clusters are removed

    #If background component dominates, reject it
#    zi[R_bn > 0.5] = -1
    zi[R_bn > np.array(R.sum(axis=1)).T[0]] = -1

    labels = np.ones(len(zi))
    labels[zi==-1] = 0

    return labels.astype(int)


def graphRefinement(data, regGraph, b=0, verbose=1):
    '''
    Perform a refinement of the graph based on 2 assumptions:
        - Branches of the graph structure should not have a length lower than b
        ### NOT DONE ### - Bifurcations should be merged if the path linking the 2 is straight (no branches)
                and with a length lower than b

    [+] data: (N, D) numpy array
            Array of D-dimensional datapoints
    [+] regGraph: regGraph_ object
            Contains regularized graph information (model parameters)
    [+] b: float
            Minimum value to keep branches of the graph


    [-] F_refined: (K', D) numpy array with K' <= K
            Positions of refined graph nodes
    [-] idx_refined: (K',) numpy array
            Indices in F such that F[idx_refined] = F_refined
    '''

    t0 = time.time()
    D = data.shape[1]

    F = regGraph.F
    edges = regGraph.edges

    key = 'weight'

    #Build an igraph object from edges and nodes
    G = ig.Graph(edges, directed=False)
    d = np.asarray(G.degree())
    d = np.append(d, np.zeros(len(F.T[0]) - len(d)))
    w = na.compute_weights(F, edges)
    G.es.set_attribute_values(key, w)

    if verbose >= 1:
        print()
        print('graphRefinement -- Branches shorter than {:.3f} will be removed iteratively'.format(b))

    newF = np.copy(F)

    cnt = 0
    while 'There are still branches with length < b':
        cnt += 1
        #Extract branches
        nodes_to_throw = []
        #filaments = []
        indices = G.get_adjlist()    #Could be done without igraph
        edges = G.get_edgelist()     #Could be given as an argument
        edges_array = np.array(edges)
        d = np.asarray(G.degree())   #Could be computed without igraph

        K = G.vcount()
        dok = sp.csc_matrix((np.arange(0, len(edges_array)), (edges_array[:,0], edges_array[:,1])), shape=(K, K)).todok()

        new_edges = np.vstack(edges)

        try:
            w = np.asarray(G.es.get_attribute_values(key))
        except:
            w = np.ones(len(edges))

        #Step 1: find the topological class of each nodes (Junction, bifurcation or extremity)
        node_topo = np.zeros(len(d))
        node_topo[d==1] = 1  #Extremity
        node_topo[d==2] = 2  #Junction
        node_topo[d>2] = 3   #Bifurcation

        extremities = np.where(node_topo == 1)[0]

        for e in extremities:
            i = indices[e][0]     #there should be only 1 element for extremities
            length = 0
            prev_node = e
            nodes_tmp = []
            nodes_tmp.append(e)
            while node_topo[i] != 1 and node_topo[i] != 3: #(not(i in bifurcations) and not(i in extremities)):
                nodes_tmp.append(i)
#                idx_edge = _findIndexWithEntry(edges_array, min(prev_node, i), max(prev_node, i))[0][0]
                idx_edge = dok[min(prev_node, i), max(prev_node, i)]   #TB 200316: faster
                length += w[idx_edge]
                idx = np.where(np.asarray(indices[i]) != prev_node)[0][0]
                prev_node = i        #Previous node
                i = indices[i][idx]  #There should be 2 elements for junction so we take the node we don't come from

            #Adding the last node (bifurcation or extremity)
#            idx_edge = _findIndexWithEntry(edges_array, min(prev_node, i), max(prev_node, i))[0][0]
            idx_edge = dok[min(prev_node, i), max(prev_node, i)]   #TB 200316: faster
            length += w[idx_edge]
            nodes_tmp.append(i)
            prev_node = i

            typeOfFil = -1
            if node_topo[prev_node] == 1: #prev_node in extremities:
                typeOfFil = 1
            elif node_topo[prev_node] == 3: #prev_node in bifurcations:
                typeOfFil = 2

            #If length too small, we remove this branch
            if (length < b) and (typeOfFil == 2):
                node_to_throw = nodes_tmp[::-1][1::]
                iid = np.mod(np.argwhere(np.in1d(new_edges.ravel('F'), node_to_throw)).T[0], len(new_edges))
                new_edges = np.delete(new_edges, iid, axis=0)
                newF[node_to_throw] = 0
                nodes_to_throw.append(nodes_tmp[0:len(nodes_tmp)-1])

        G = ig.Graph(new_edges.tolist())
        w = na.compute_weights(F, new_edges)
        G.es.set_attribute_values('weight', w)

        if len(nodes_to_throw) == 0:
            break

    #The new set of nodes and edges
    F_refined = newF
    idx_toThrow = np.unique(np.where(F_refined == [0] * D)[0])
    idx_refined = np.arange(len(F))
    idx_refined = np.delete(idx_refined, idx_toThrow, axis=0)
    F_refined = np.delete(F_refined, idx_toThrow, axis=0)

    #Go along oll thrown nodes and decrease the indices consequently to get correct remaining edges
    for (inc, i) in enumerate(idx_toThrow[::-1]):
        new_edges[new_edges > i] -= 1           #TB 200317: a bit slow..

    ed_ref = []
    for e in new_edges:
        ed_ref.append((e[0], e[1]))

    #Regularized graph structure for output
    regGraph_refined = regGraph_()
    regGraph_refined.F = F_refined
    regGraph_refined.alpha = regGraph.alpha
    regGraph_refined.edges = ed_ref

    if isinstance(regGraph.pi_k, np.ndarray):
        regGraph_refined.pi_k = regGraph.pi_k[idx_refined]
    else:
        regGraph_refined.pi_k = regGraph.pi_k

    if isinstance(regGraph.sig, np.ndarray):
        regGraph_refined.sig = regGraph.sig[idx_refined]
    else:
        regGraph_refined.sig = regGraph.sig

    if verbose >= 1:
        print('graphRefinement -- {:d} iterations done in {:.3f} seconds'.format(cnt, time.time()-t0))
        print('graphRefinement -- ' + str(len(F) - len(F_refined)) + ' nodes removed during the refinement')

    return regGraph_refined



def connectivity(pos, Rmax, F, edges, display_progress_bar=True):
    '''
    Computes the connectivity of a point at a given position and considering the graph structure
    [+] pos: (N,D) numpy array
            Positions to study
    [+] Rmax: (N,) numpy array
            Maximum euclidean distance to look around the positions pos
    [+] F: (K, D) numpy array
            Nodes positions of the graph
    [+] edges: list
            Set of all doublets of indices linked in the graph

    [-] (iclos, dclos): list of 2 floats
            Index and distance of closest node to the "pos" position
    [-] life: (Nc, 3) numpy array
            Encodes all changes in the connectivity around the location with variation of the distance from it
            col1: Connectivity events (+/-)
            col2: Geodesic distance of the event from the initial node
            col3: Euclidean distance of the event from the initial node

    Note: the function returns only events that intervienes at a euclidean distance < Rmax from the initial position on the graph

    Example of plot:
    ----------------
    ii = 2   # 1: Geodesic distance   2: Euclidean distance
    idd = np.argsort(life[:, ii])
    kappa = np.cumsum(life[idd, 0])
    plt.figure();
    pltx = life[idd, ii]
    plty = kappa
    if pltx[-1] < Rmax:
        pltx = np.hstack((life[idd, ii], Rmax))
        plty = np.hstack((kappa, kappa[-1]))
    plt.step(pltx, plty, where='post')
    plt.xlabel('Geodesic distance')
    plt.ylabel('Connectivity')
    '''

    all_life = []
    all_iclos = []
    all_dclos = []

    G = ig.Graph(edges)

    indices = G.get_adjlist()    #Could be done without igraph
    d = np.asarray(G.degree())   #Could be computed without igraph
    weights = na.compute_weights(F, edges)
    G.es.set_attribute_values('weight', weights)

    kdt = cKDTree(F)

    if display_progress_bar:
        pbar = tqdm(total=len(pos), initial=0, position=0, leave=True)

    for (i,p) in enumerate(pos):
        nodes_done = []
        next_nodes = []
        life = []

        #Find the closest node in F from position pos
        dclos, iclos = kdt.query(p)

        nodes_done.extend([iclos])
        next_nodes.extend(indices[iclos])

        #Initially, the connectivity is just the node degree
        life.append((d[iclos], 0, 0))   #death/birth, Dg, De

        while len(next_nodes) > 0:
            n = next_nodes[0]

            de = np.linalg.norm(F[iclos] - F[n])     #Euclidean distance from starting point

            dg = np.sum(weights[G.get_shortest_paths(iclos, n, output='epath')])   #Geodesic distance from starting point

            if de < Rmax[i] and d[n] == 1:   #Extremity kills a connectivity (Record the event only if de < R)
                life.append((-1, dg, de))
            elif de < Rmax[i] and d[n] > 2:  #Bifurcations creates connectivity (Record the event only if de < R)
                life.append((d[n]-2, dg, de))

            #Check next nodes to process and add them to path if still in bound
            nodes_done.extend([n])
            idxToAdd = indices[n]
            idxToAdd = np.delete(idxToAdd, np.argwhere(np.in1d(idxToAdd, nodes_done)).T[0])    #If the node is already done

            ll = np.argwhere(np.in1d(idxToAdd, next_nodes))
            if ll.shape[0] > 0:
                idxToAdd = np.delete(idxToAdd, ll[0])    #If node is already in the list

            if de < Rmax[i]:
                next_nodes.extend(idxToAdd)

            del next_nodes[0]

        life = np.vstack(life)
        idd = np.argsort(life[:,2])     #Sort events by increasing euclidean distance
        life = life[idd, :]

        all_life.append(life)
        all_dclos.append(dclos)
        all_iclos.append(iclos)

        if display_progress_bar:
            pbar.update(1)

    if display_progress_bar:
        pbar.close()

    all_life = np.array(all_life)
    all_dclos = np.array(all_dclos)
    all_iclos = np.array(all_iclos)

    return all_iclos, all_dclos, all_life



def radial_distance_skeleton(X, F, edges):

    D = X.shape[1]

    #for each position x, project it on the filamnetary structure
    pbar = tqdm(total=len(X), initial=0, position=0, leave=True)
    residuals = np.zeros(len(X))

    arr_edges = np.array(edges)  #From list to numpy array to allow indexing

    nn = NearestNeighbors(n_neighbors=10).fit(F)

    for (index, p) in enumerate(X):

        #Restrict to only a limited number of edges
        knn = nn.kneighbors(np.array([p]))[1][0]
        idx1 = np.argwhere(np.in1d(arr_edges[:,0], knn)).T[0]
        idx2 = np.argwhere(np.in1d(arr_edges[:,1], knn)).T[0]
        idx = np.unique(np.hstack((idx1, idx2)))

        edg = arr_edges[idx,:].T

        if D == 2:
                dist, posNearest = utility.vectorized_pnt2line(np.hstack((p, 0)),
                                                               np.hstack((F[edg[0]], np.array([np.zeros(len(edg[1]))]).T)),
                                                np.hstack((F[edg[1]], np.array([np.zeros(len(edg[1]))]).T)))
        elif D == 3:
                dist, posNearest = utility.vectorized_pnt2line(p, F[edg[0]], F[edg[1]])

        dist = np.array(dist)
        posNearest = np.array(posNearest)

        argmin = np.argmin(dist)
        d = dist[argmin]

        residuals[index] = d

        pbar.update(1)

    pbar.close()

    return residuals



#==============================================================================
#       READ, WRITE AND CONVERT FILAMENTS
#==============================================================================
def convertToDataFrame(filaments):
    '''
    Convert a list of filaments into a pandas DataFrame

    [+] filaments: list
            Set of filaments_ class instances

    [-] df: pandas DataFrame
            Contains the same information as filament_ class but in a DataFrame format
    '''

    fil_nodes = []
    fil_edges = []
    fil_length = []
    fil_curvature = []
    fil_residuals = []
    fil_datapoints = []
    fil_type = []
    fil_uncertainty = []

    for f in filaments:
        fil_length.append(f.length)
        fil_curvature.append(f.curvature)
        fil_type.append(f.type)
        fil_nodes.append(f.nodes)
        fil_edges.append(f.edges)
        fil_residuals.append(f.residuals)
        fil_datapoints.append(f.datapoints)
        fil_uncertainty.append(f.spatial_uncertainty)

    fil_nodes = np.array(fil_nodes)
    fil_edges = np.array(fil_edges)
    fil_length = np.array(fil_length)
    fil_curvature = np.array(fil_curvature)
    fil_residuals = np.array(fil_residuals)
    fil_datapoints = np.array(fil_datapoints)
    fil_type = np.array(fil_type)
    fil_uncertainty = np.array(fil_uncertainty)

    df = pd.DataFrame()
    df.insert(0, 'length', fil_length)
    df.insert(1, 'curvature', fil_curvature)
    df.insert(2, 'type', fil_type)
    df.insert(3, 'nodes', fil_nodes)
    df.insert(4, 'edges', fil_edges)
    df.insert(5, 'residuals', fil_residuals)
    df.insert(6, 'assignment', fil_datapoints)
    df.insert(7, 'uncertainty', fil_uncertainty)

    return df



def convertToFilaments(df):
    '''
    Convert a pandas DataFrame into a list of filaments instances

    [+] df: pandas DataFrame
            Contains the same information as filaments_ class but in a DataFrame format

    [-] filaments: list
            Set of filaments_ class instances
    '''

    filaments = []

    for i in range(len(df)):
        filament = filaments_()
        filament.length = df['length'][i]
        filament.curvature = df['curvature'][i]
        filament.type = df['type'][i]
        filament.nodes = df['nodes'][i]
        filament.edges = df['edges'][i]
        filament.residuals = df['residuals'][i]
        filament.datapoints = df['assignment'][i]
        filament.uncertainty = df['uncertainty'][i]
        filaments.append(filament)

    return filaments



def uncertaintyBalls(data, F, param, display_progress_bar):
    '''
    Compute uncertainty associated to each points in F using a bootstrap approach

    [+] data: NxD numpy array
            Array of D-dimensional datapoints
    [+] F: KxD numpy array
            Positions of reference graph nodes
    [+] param: Instance of 'param_' class
            The set of parameters to use
    [+] display_progress_bar=0: float
            If set to 1, a progress bar is displayed

    [-] rho: Kx1 numpy array
            The mean distance from each points in F to the other param.B realizations of regularized graph
    '''

    verbose = param.verbose_level

    if verbose >= 1:
        print('uncertaintyBalls -- Starting bootstrap')

    B = param.B
    allF, allEdges = bootstrap(data, param, display_progress_bar=display_progress_bar)

#    md = []

    if verbose >= 1:
        print('')
        print('uncertaintyBalls -- Starting computing distances')

    if display_progress_bar:
        pbar = tqdm(total=param.B, initial=0, position=0, leave=True)

    rho = np.zeros(len(F))
    for b in range(B):
        #Build a cKDTree on the current graph
        kdtree_b = cKDTree(allF[b])

        #Find the minimum distance from all references in the bootstrap graph
        mindist = kdtree_b.query(F, k=1)[0]
#        md.append(mindist)
        rho += 1/B * mindist**2

        if display_progress_bar:
            pbar.update(1)

#    md = np.vstack(md)
#    rho = np.sqrt(np.mean(md, axis=0))
    rho = np.sqrt(rho)

    if display_progress_bar:
        pbar.close()

    return rho



def writeCatalog_h5(data, regGraph, filaments, param, filename):
    '''
    Write a catalog of branches in h5 format

    [+] data: NxD numpy array
            Datapoints positions
    [+] F: KxD numpy array
            Nodes position
    [+] edges: Kx2 numpy array
            Indices representing connections between nodes
    [+] filaments: list
            Regularization_GMM.filaments_ objects of detected filaments
    [+] param: Instance of 'param_' class
            The set of parameters to use
    [+] filename: string
            Name of the h5 file (followed by extension .hdf5)
    '''

    df = convertToDataFrame(filaments)

    #Create h5 file
    file = h5py.File(filename, "w")

    #Parameters
    file.attrs[u'file_name'] = filename
    file.attrs[u'N_datapoints'] = len(data)
    file.attrs[u'N_nodes'] = len(regGraph.F)
    file.attrs[u'N_edges'] = len(regGraph.edges)

    params = file.create_group("param")
    params.attrs[u'lambda'] = param.lam
    params.attrs[u'pruning_level'] = param.denoisingParameter
    params.attrs[u'max_iter'] = param.maxIter
    params.attrs[u'A0'] = param.A0
    params.attrs[u'variance'] = param.sig
    params.attrs[u'topology_function'] = param.topology.__name__
    params.attrs[u'verbose_level'] = param.verbose_level
    params.attrs[u'B'] = param.B
    params.attrs[u'perc'] = param.perc
    params.attrs[u'minSigma'] = param.minSigma
    params.attrs[u'background_noise'] = param.background_noise
    params.attrs[u'domain_area'] = param.domain_area
    params.attrs[u'alpha'] = param.alpha
    params.attrs[u'covariance_update'] = param.covariance_update

    #Datapoints and graph data
    file.create_dataset("datapoints_pos", data=data)
#    file.create_dataset("nodes_pos", data=F)
#    file.create_dataset("edges", data=edges)

    graph = file.create_group("graph")
    graph.create_dataset("nodes_pos", data=regGraph.F)
    graph.create_dataset("edges", data=regGraph.edges)
    graph.create_dataset("pi_k", data=regGraph.pi_k)
    graph.create_dataset("variances", data=regGraph.sig)
    graph.create_dataset("alpha", data=regGraph.alpha)


    #Filaments group
    Nfil = len(filaments)
    fil = file.create_group("filaments")
    fil.attrs[u'N_filaments'] = Nfil

    if Nfil > 0:
        dt_float = h5py.special_dtype(vlen=np.dtype('float64'))
        dt_int = h5py.special_dtype(vlen=np.dtype('int'))

        fil.create_dataset("length", data=df['length'])
        fil.create_dataset("curvature", data=df['curvature'])
        fil.create_dataset("type", data=df['type'])

        fil.create_dataset("nodes", (len(filaments),), dtype=dt_int)
        fil["nodes"][...] = df['nodes']

        fil.create_dataset("edges", (len(filaments),), dtype=dt_int)
        fil["edges"][...] = df['edges']

        fil.create_dataset("residuals", (len(filaments),), dtype=dt_float)
        fil["residuals"][...] = df['residuals']

        fil.create_dataset("assignment", (len(filaments),), dtype=dt_int)
        fil["assignment"][...] = df['assignment']

        fil.create_dataset("uncertainty", (len(filaments),), dtype=dt_float)
        fil["uncertainty"][...] = df['uncertainty']

    file.close()

    return



def readCatalog_h5(filename):
    '''
    Read a catalog of branches in h5 format

    [+] filename: string
            Name of the h5 file (followed by extension .hdf5)

    [-] data: NxD numpy array
            Datapoints positions
    [-] F: KxD numpy array
            Nodes position
    [-] edges: Kx2 numpy array
            Indices representing connections between nodes
    [-] filaments: list
            Regularization_GMM.filaments_ objects of detected filaments
    '''

    file = h5py.File(filename, "r")
    t0 = time.time()

    data = np.array(file["datapoints_pos"])

    #Reconstruct regGraph object
    regGraph = regGraph_()
    regGraph.F = np.array(file['graph']["nodes_pos"])
    regGraph.edges = np.array(file['graph']["edges"])
    regGraph.pi_k = np.array(file['graph']['pi_k'])
    if regGraph.pi_k.shape == ():
        regGraph.pi_k = float(regGraph.pi_k)
    regGraph.alpha = float(np.array(file['graph']['alpha']))
    regGraph.sig = np.array(file['graph']['variances'])
    if regGraph.sig.shape == ():
        regGraph.sig = float(regGraph.sig)

    Nfil = file['filaments'].attrs[u'N_filaments']

    if Nfil > 0:
        filaments = pd.DataFrame()
        filaments.insert(0, 'length', np.array(file['filaments']['length']))
        filaments.insert(1, 'curvature', np.array(file['filaments']['curvature']))
        filaments.insert(2, 'type', np.array(file['filaments']['type']))
        filaments.insert(3, 'nodes', np.array(file['filaments']['nodes']))
        filaments.insert(4, 'edges', np.array(file['filaments']['edges']))
        filaments.insert(5, 'residuals', np.array(file['filaments']['residuals']))
        filaments.insert(6, 'assignment', np.array(file['filaments']['assignment']))
        filaments.insert(7, 'uncertainty', np.array(file['filaments']['uncertainty']))
    else:
        filaments = []

    param = param_()
    param.lam = file['param'].attrs['lambda']
    param.denoisingParameter = file['param'].attrs['pruning_level']
    param.maxIter = file['param'].attrs['max_iter']
    param.A0 = file['param'].attrs['A0']
    param.sig = file['param'].attrs['variance']
    param.topology = file['param'].attrs['topology_function']
    param.verbose_level = file['param'].attrs['verbose_level']
    param.B = file['param'].attrs['B']
    param.perc = file['param'].attrs['perc']
    param.alpha = file['param'].attrs['alpha']
    param.background_noise = file['param'].attrs['background_noise']
    param.domain_area = file['param'].attrs['domain_area']
    param.minSigma = file['param'].attrs['minSigma']
    param.covariance_update = file['param'].attrs['covariance_update']

    tr = time.time() - t0

    print('File read in {:.4f} seconds with {:d} datapoints and {:d} filaments'.format(tr, len(data), len(filaments)))
    print()
    print('Parameters used:')
    print('-----------------')
    print('lambda = {:.1f}'.format(param.lam))
    print('denoising = {:.1f}'.format(param.denoisingParameter))
    print('variance = {:.3f}'.format(param.sig))
    print('topology = {}'.format(param.topology))
    print('Covariance = {}'.format(param.covariance_update))
    print('Uniform background = {}'.format(param.background_noise))

    return data, regGraph, filaments, param
