import numpy as np
import scipy.stats as st
from tqdm import tqdm
import scipy.sparse as sp


def _computeResponsability(x, F, sigs):
    '''
    -- INTERNAL FUNCTION --
    Compute the responsability of a datapoint x given a set of centroids F
    WARNING: Assumes an isotropic diagonal covariance for each cluster K (sigs is a Kx1 numpy array)
    '''

    dst = np.linalg.norm(x - F, axis=1)**2
    pr = np.exp(-dst / 2 / sigs)
    resp = pr / np.sum(pr)
    return resp


def _mvn(mu, sig, X):
    '''
    -- INTERNAL FUNCTION --
    Gaussian pdf computed of a position X for several gaussian clusters.
    The gaussian is assumed to be spherical with scalar covariance.

    [+] mu: (K,D) numpy array
            Mean of gaussian clusters
    [+] sig: (K,) numpy array
            Variance of gaussian clusters
            If scalar, assumes the same for all clusters
    [+] X (D,) numpy array
            Contains position to evaluate the pdf for each gaussian

    [-] pdf: (K,) numpy array
            Value of the gaussian pdf computed at position X for all different gaussians
    '''
    D = len(X)
    A = 1 / (2*np.pi*sig)**(D/2)
    pdf = A * np.exp(-1/2 * np.linalg.norm(X-mu, axis=1)**2 / sig)
    return pdf


def _mvnX(mu, sig, X):
    '''
    -- INTERNAL FUNCTION --
    Gaussian pdf computed of a position X for several gaussian clusters.
    The gaussian is assumed to be spherical with scalar covariance.

    [+] mu: (D,) numpy array
            Mean of gaussian
    [+] sig: scalar
            Variance of gaussian clusters
    [+] X (N,D) numpy array
            Contains positions to evaluate the pdf

    [-] pdf: (N,) numpy array
            Value of the gaussian pdf computed at positions X
    '''

    D = len(mu)
    if isinstance(sig, np.ndarray):
        A = 1 / ((2*np.pi)**(D/2)*np.linalg.det(sig)**(1/2))
        invSig = np.linalg.inv(sig)
        pdf = A * np.exp(-1/2 * (X-mu).T @ invSig @ (X-mu))
    else:
        A = 1 / (2*np.pi*sig)**(D/2)
        pdf = A * np.exp(-1/2 * np.linalg.norm(X-mu, axis=1)**2  / sig)

    return pdf


def _normalize(R, cte=0):
    '''
    -- INTERNAL FUNCTION --
    Normalize a scipy sparse matrix over rows

    [+] R: scipy sparse matrix
            The matrix to be normalized
    [+] cte: scalar
            Constant to add to the normalization constant
    '''
    R = R.tocsr()

    norm = cte + np.repeat(np.add.reduceat(R.data, R.indptr[:-1]), np.diff(R.indptr))
    R.data = R.data / norm
    return R, norm


def updatePositionsGMM(data, R, A):
    '''
    Update positions of centers of gaussian clusters by solving linear system in each dimension.

    [+] data: NxD numpy array
            Array of D-dimensional datapoints
    [+] R: NxK numpy array
            Responsability matrix (support scipy sparse matrix)
    [+] A: KxK numpy array
            Matrix A = lambda*L + Lambda (support scipy sparse matrix)

    [-] F: KxD numpy array
            Updated clusters means positions
    '''

    K = R.shape[1]
    D = len(data.T)

    #Solve linear system for each dimension
    lu_obj = sp.linalg.splu(A)
    XR = data.T @ R
    F = np.zeros((K, D))
    for j in range(D):
        F[:,j] = lu_obj.solve(XR[j,:])
    return F


#import time
def M_step(data, V, R, A, Lambda, update_pos_only=True, background_noise=False, resp_background=[]):
    '''
    Compute the M-step of the regularized GMM problem

    [+] data: NxD numpy array
            Array of D-dimensional datapoints
    [+] R: NxK numpy array
            Responsability matrix (support scipy sparse matrix)
    [+] A: KxK numpy array
            Matrix A = lambda*L + Lambda (support scipy sparse matrix)
    [+] Lambda: KxK sparse diagonal matrix
            Diagonal matrix containing sum of responsabilities for each cluster
    [+] update_pos_only: boolean
            If True, update only the cluster means and leave rest unchanged
    [+] background_noise: boolean
            If True, also update alpha, the amplitude of background noise
    [+] resp_background: (N,) numpy array
            Responsabilities of the background for each datapoint

    [-] F: KxD numpy array
            Updated cluster mean positions
    [-] variance: scalar or (K,) numpy array
            Updated cluster variance
    [-] pi_k: empty list or (K,) numpy array
            Updated amplitudes associated to each cluster
    [-] alpha: scalar
            Updated background noise amplitude
    '''

    N = data.shape[0]
    D = data.shape[1]
    pi_k = []
    variance = 0
    alpha = 0

    #Position update
#    t0 = time.time()
    F = updatePositionsGMM(data, V, A)

    if not update_pos_only:
        #Variance update
        nk = Lambda.data[0]
#        t1 = time.time()
        if D == 2:
            diagCov, antiDiag = updateCov2D(data, F, R, nk, diag_only=True)
        elif D == 3:
            diagCov, covXY, covXZ, covYZ = updateCov3D(data, F, R, nk, diag_only=True)
        variance = np.mean(diagCov, axis=1)
#        t2 = time.time()

        #Amplitudes update
        pi_k = 1/N * nk

        #Background amplitude update
        alpha = 0
        if background_noise:
            alpha = 1/N * np.sum(resp_background)

#        t3 = time.time()

#        print('Time position : {:.3f}'.format(t1-t0))
#        print('Time variance : {:.3f}'.format(t2-t1))
#        print('Time amplitudes : {:.3f}'.format(t3-t2))
    return F, variance, pi_k, alpha



def evaluateObjective(data, F, R, Lambda, L, lam, sigs, pi_k, full=1, background_noise=False, alpha=0, domain_area=0):
    ''' Evaluate the objective function computed as the lower bound of the EM algorithm
    Basically, it corresponds to the sum of 3 terms: Q, the expectation function, H the log entropy and the prior log(p(\theta))

    [+] data: NxD numpy array
            Array of D-dimensional datapoints
    [+] F: KxD numpy array
            Clusters means positions
    [+] R: NxK numpy array
            Responsability matrix (support sparse matrix)
    [+] Lambda: KxK diagonal sparse matrix
            The expected number of datapoints in each clusters (Lambda.data=R.sum(axis=0))
    [+] L: KxK sparse matrix
            Laplacian matrix of the current graph structure
    [+] lam: float > 0
            Regularization parameter
    [+] sigs: Kx1 numpy array
            Current estimate of variance for each spherical cluster

    [-] obj: float
            Estimated value of the objective function
    '''
    if full:
        log_ptheta = - lam*np.trace(F.T@L@F)

        #Greedy computation of the log-likelihood
#        Loglikelihood = 0
#        for i in range(len(data)):
#            pdf = _mvn(F, sigs, data[i])
#            if background_noise:
#                Loglikelihood += np.log(np.sum(pi_k*pdf) + alpha/domain_area)
#            else:
#                Loglikelihood += np.log(np.sum(pi_k*pdf))

        ss = np.ones(len(F))*sigs

        #Compute the full log-likelihood
        D = data.shape[1]
        log_prob = (np.sum((F ** 2), axis=1)*(1/ss) - 2. * np.dot(data, (F.T*(1/ss))) + np.outer(np.linalg.norm(data, axis=1)**2, (1/ss).T))
        log_det = D * np.log(1/ss**0.5)
        ll = -.5 * (data.shape[1] * np.log(2 * np.pi) + log_prob) + log_det

        probs = ll+np.log(pi_k)
        if background_noise:
            log_prob_norm = np.log(np.sum(np.exp(probs)+alpha/domain_area, axis=1))
        else:
            log_prob_norm = np.log(np.sum(np.exp(probs), axis=1))
        loglikelihood = np.sum(log_prob_norm)

        regLog = loglikelihood + log_ptheta
        lower_bound = regLog
#        print('LogL = '+ str(loglikelihood))
#        print('Prior= '+ str(log_ptheta))
#        print('===========================')

    else:
#        Q = R.data*np.log(pi_k) - np.trace(data.T@data) - np.trace(F.T@Lambda@F) + 2*np.trace(data.T@R@F)  #Expectation term
        log_ptheta = - lam*np.trace(F.T@L@F)
##        H = sigs**2*np.sum(R.data*np.log(R.data))        #entropy term
#        lower_bound = Q + log_ptheta #- H  #Q + log(P(theta)) - H is the full posterior


        #2nd attempt
#        ksi = sp.diags(np.array(R.sum(0))[0])
        D = data.shape[1]

#        ksi = sp.diags(np.array(R.sum(1)).T[0])
#        Q = np.sum(R.multiply(sigs)*np.log(pi_k)) - D/2*np.sum(R.multiply(sigs)*np.log(2*np.pi)) - D/2*np.sum(R.multiply(np.log(sigs)))
#        Q = -1/2*(np.trace(data.T@ksi@data) + np.trace(F.T@Lambda@F) - 2*np.trace(data.T@R@F))  #what is inside the exponential
#        lower_bound = Q

        #Compute the full log-likelihood (thanks to scikit-learn)
        log_prob = (np.sum((F ** 2), axis=1)*(1/sigs) - 2. * np.dot(data, (F.T*(1/sigs))) + np.outer(np.linalg.norm(data, axis=1)**2, (1/sigs).T))
        log_det = D * np.log(1/sigs**0.5)
        ll = -.5 * (data.shape[1] * np.log(2 * np.pi) + log_prob) + log_det
        log_prob = np.log(np.sum(np.exp(ll+np.log(pi_k)), axis=1))
        loglikelihood = np.sum(log_prob)

        lower_bound = loglikelihood + log_ptheta


    return lower_bound


def evaluateLL(data, F, sigs, pi_k):
    D = data.shape[1]
    log_prob = (np.sum((F ** 2), axis=1)*(1/sigs) - 2. * np.dot(data, (F.T*(1/sigs))) + np.outer(np.linalg.norm(data, axis=1)**2, (1/sigs).T))
    log_det = D * np.log(1/sigs**0.5)
    ll = -.5 * (data.shape[1] * np.log(2 * np.pi) + log_prob) + log_det
    log_prob_norm = np.log(np.sum(np.exp(ll+np.log(pi_k)), axis=1))
    loglikelihood = np.sum(log_prob_norm)

    return loglikelihood


def evaluateObjective_local(data, F, R, Lambda, L, lam, sigs, pi_k, full=1, background_noise=False, alpha=0, domain_area=0, grid=[]):

#    gridx = grid[0]
#    gridy = grid[1]

#    map_obj = np.zeros((64, 64))
    map_obj = np.zeros(len(F))

    h = 1 #0.2

    if full:
#        log_ptheta = - lam*np.trace(F.T@L@F)
        all_pdf = []

        for i in range(len(data)):
            pdf = _mvn(F, sigs, data[i])
            all_pdf.append(np.log(np.sum(pi_k*pdf)))

        all_pdf = np.array(all_pdf)

        #Evaluate on the grid
#        for (ix, xx) in enumerate(gridx):
#            for (iy, yy) in enumerate(gridy):
#                Loglikelihood = 0
#                weights = _mvnX(np.array([xx, yy]), h**2, data)
#                Loglikelihood += np.sum(weights * all_pdf)
#
#                map_obj[ix, iy] = Loglikelihood #+ log_ptheta

        #Evaluate at nodes positions
        for (ii, ff) in enumerate(F):
            Loglikelihood = 0
            weights = _mvnX(ff, h**2, data)
            Loglikelihood += np.sum(weights*all_pdf)

            map_obj[ii] = Loglikelihood

    return map_obj


def neighborsCov(kdtree, data, centers, k):
    D = len(data.T)
    dst, nbrs = kdtree.query(centers, k=k)
    x = data[nbrs]

    covs = []
    cov00 = []
    cov11 = []
    cov01 = []

    cov02 = []
    cov12 = []
    cov22 = []

    for x in data[nbrs]:
        cov = np.cov(x.T)
        covs.append(cov)
        cov00.append(cov[0,0])
        cov11.append(cov[1,1])
        cov01.append(cov[0,1])

        if D == 3:  #In 3D, add other terms
            cov02.append(cov[0,2])
            cov12.append(cov[1,2])
            cov22.append(cov[2,2])

    cov00 = np.array(cov00)
    cov11 = np.array(cov11)
    cov22 = np.array(cov22)
    cov01 = np.array(cov01)
    cov02 = np.array(cov02)
    cov12 = np.array(cov12)

    return cov00, cov11, cov22, cov01, cov02, cov12, covs


def diagCov(data, F, R, nk):
    ''' Estimate the diagonal elements of the covariance matrices

    [+] data: NxD numpy array
            Array of D-dimensional datapoints
    [+] F: KxD numpy array
            Clusters means positions
    [+] R: NxK numpy array
            Responsability matrix (support sparse matrix)
    [+] nk: Kx1 numpy array
            Current number of data associated to kernels (=R.sum(axis=0))

    [-] diagCov: KxD numpy array
            Diagonal elements of all covariance matrices

    Note: For example, in 2D, the covariance matrix of cluster k is then C[k] = [[diagCov[k, 0], 0], [0, diagCov[k, 1]]]
    '''
    avg_X2 = R.T@data**2 / nk[:, np.newaxis]
    avg_F2 = F**2
    avg_XF = R.T@data / nk[:, np.newaxis] * F
    diagCov = avg_X2 - 2 * avg_XF + avg_F2
    return diagCov



def updateCov2D(data, F, R, nk, diag_only=0):
    ''' Estimate all elements of all covariance matrices in the 2D case

    [+] data: NxD numpy array
            Array of D-dimensional datapoints
    [+] F: KxD numpy array
            Clusters means positions
    [+] R: NxK numpy array (support scipy sparse matrix)
            Responsability matrix
    [+] nk: Kx1 numpy array
            Current number of data associated to kernels (=R.sum(axis=0))
    [+] diag_only: Boolean
            To only return the diagonal elements of the matrix

    [-] diagCov: Kx2 numpy array
            Diagonal elements of all covariance matrices
    [-] antiDiag: Kx2 numpy array
            Antidiagonal elements of all covariance matrices

    Note: The full covariance matrix of cluster k is then C[k] = [[diagCov[k,0], antiDiag[k]], [antiDiag[k], diagCov[k,1]]]
    '''
    diag = diagCov(data, F, R, nk)

    antiDiag = []
    if diag_only == False:
        #Antidiagonal terms
        avg_Xxy = R.T @ (data.T[0] * data.T[1]) / nk
        avg_Fxy = (F.T[0] * F.T[1])
        avg_XyFx = (R.T @ data).T[1] * F.T[0] / nk
        avg_XxFy = (R.T @ data).T[0] * F.T[1] / nk
        antiDiag = avg_Xxy - avg_XyFx - avg_XxFy + avg_Fxy

    return diag, antiDiag



def updateCov3D(data, F, R, nk, diag_only=False):
    ''' Estimate all elements of all covariance matrices in the 3D case
    The full symmetric covariance matrix can be retrieved as:
            [[diag[0], xy, xz],
             [xy, diag[1], yz],
             [xz, yz, diag[2]]]

    [+] data: NxD numpy array
            Array of D-dimensional datapoints
    [+] F: KxD numpy array
            Clusters means positions
    [+] R: NxK numpy array (support scipy sparse matrix)
            Responsability matrix
    [+] nk: Kx1 numpy array
            Current number of data associated to kernels (=R.sum(axis=0))
    [+] diag_only: Boolean
            To only return the diagonal elements of the matrix

    [-] diagCov: Kx2 numpy array
            Diagonal elements of all covariance matrices
    [-] xy: Kx1 numpy array
            XY covariances
    [-] xz: Kx1 numpy array
            XZ covariances
    [-] yz: Kx1 numpy array
            YZ covariances
    '''
    #Diagonal elements
    diag = diagCov(data, F, R, nk)

    xy = []
    xz = []
    yz = []
    if diag_only == False:
        #XY
        avg_Xxy = R.T @ (data.T[0] * data.T[1]) / nk
        avg_Fxy = (F.T[0] * F.T[1])
        avg_XyFx = (R.T @ data).T[1] * F.T[0] / nk
        avg_XxFy = (R.T @ data).T[0] * F.T[1] / nk
        xy = avg_Xxy - avg_XyFx - avg_XxFy + avg_Fxy

        #XZ
        avg_Xxz = R.T @ (data.T[0] * data.T[2]) / nk
        avg_Fxz = (F.T[0] * F.T[2])
        avg_XzFx = (R.T @ data).T[2] * F.T[0] / nk
        avg_XxFz = (R.T @ data).T[0] * F.T[2] / nk
        xz = avg_Xxz - avg_XzFx - avg_XxFz + avg_Fxz

        #YZ
        avg_Xzy = R.T @ (data.T[2] * data.T[1]) / nk
        avg_Fzy = (F.T[2] * F.T[1])
        avg_XyFz = (R.T @ data).T[1] * F.T[2] / nk
        avg_XzFy = (R.T @ data).T[2] * F.T[1] / nk
        yz = avg_Xzy - avg_XyFz - avg_XzFy + avg_Fzy
    return diag, xy, xz, yz



def eigvals2D(a, b, c):
    ''' Compute all eigenvalues of K 2x2 symmetric matrix
    This implementation is faster than using np.linalg.eigvals on each matrix

    [+] a: Kx1 numpy array
            K first diagonal elements of K matrices
    [+] b: Kx1 numpy array
           K second diagonal elements of K matrices
    [+] c: Kx1 numpy array
            K antidiagonal elements of K matrices

    [-] eigs: 2x1 numpy array
            Eigenvalues in decreasing order
    '''
    trace = a + b
    det = a*b - c*c

    l1 = trace/2 + np.sqrt(trace**2/4 - det)
    l2 = trace/2 - np.sqrt(trace**2/4 - det)

    eigs = np.vstack((l1, l2)).T

    return eigs



def eigvals3D(a, b, c, d, e, f):
    ''' Estimate all eigenvalues of K 3x3 symmetric matrix
    This implementation is faster than using np.linalg.eigvals on each matrix

    [+] a: Kx1 numpy array
            K first diagonal elements of K matrices
    [+] b: Kx1 numpy array
            K second diagonal elements of K matrices
    [+] c: Kx1 numpy array
            K third diagonal elements of K matrices
    [+] d: Kx1 numpy array
            K elements [1, 2] of K matrices
    [+] e: Kx1 numpy array
            K elements [1, 3] of K matrices
    [+] f: Kx1 numpy array
            K elements [2, 3] of K matrices

    [-] eigs: 3x1 numpy array
            Eigenvalues in decreasing order
    '''
    q = (a + b + c)/3   #Trace/3
    q2 = (a-q)**2 + (b-q)**2 + (c-q)**2 + 2*d**2 + 2*e**2 + 2*f**2   #Trace of (A-q*I)^2
    p = np.sqrt(q2 / 6)
    detB = 1/p**3*((a-q)*(b-q)*(c-q) + 2*d*f*e - e**2*(b-q) - d**2*(c-q) - f**2*(a-q))
    r = detB / 2

    phi = np.ones(len(r))*np.arccos(r)/3
    phi[r<=-1] = np.pi /3
    phi[r>=1] = 0

    #Eigenvalues in decreasing order
    eig1 = q + 2 * p * np.cos(phi)
    eig3 = q + 2 * p * np.cos(phi + (2*np.pi/3))
    eig2 = 3 * q - eig1 - eig3

    eigs = np.vstack((eig1, eig2, eig3)).T

    #    eigs[eigs<0] = 0

    return eigs



def computeDensity(x, mu, S, alpha=0):
    '''
    Compute the density on x positions from a mixture model by assuming
        a Gaussian Mixture with centers mu, covariances S and weights alpha

    [+] x: positions to compute the density on (Nxd)
    [+] mu: positions of centroids (Kxd)
    [+] S: covariance of each component (Kx(dxd) or scalar hence assuming S*I(d,d) for all clusters)
    [+] alpha: mixtures' weights (if 0, assumes equidistributed weights)

    [-] density_norm: Value of the density at each grid position
    '''

    N, d = x.shape

#    d = len(x)
#    if d == 2:
#        N = x[0].shape[0]
#        M = x[1].shape[0]
#    elif d == 3:
#        N, M, L = x[0].shape

    K = len(mu)

    try:
        if len(S.shape) == 0:
            scalar_cov = 1
        else:
            scalar_cov = 0
    except:
        scalar_cov = 1

    try:
        len(alpha)
        equidistributed_weights = 0
    except:
        equidistributed_weights = 1
        alpha = 1/K

#    if d == 2:
#        xx, yy = x
#        pos = np.empty(xx.shape + (2,))
#        pos[:, :, 0] = xx; pos[:, :, 1] = yy
#        density = np.zeros((N, M))
#    elif d == 3:
#        xx, yy, zz = x
#        pos = np.empty(xx.shape + (3,))
#        pos[:, :, :, 0] = xx; pos[:, :, :, 1] = yy; pos[:, :, :, 2] == zz;
#        density = np.zeros((N, M, L))

    pos = x
    density = np.zeros(N)

    components = []
    pbar = tqdm(total=K, initial=0, position=0, leave=True)
    for k in range(K):
        if scalar_cov:
            components.append(st.multivariate_normal(mu[k], S*np.eye(d)))
        else:
            components.append(st.multivariate_normal(mu[k], S[k]))

        if equidistributed_weights:
            density += alpha*components[k].pdf(pos)
        else:
            density += alpha[k]*components[k].pdf(pos)
        pbar.update(1)

    pbar.close()

    #Normalization step
    density_norm = density / np.sum(density) * len(mu)

    return density_norm

