#Tony BONNAIRE, created on November, 2018
import scipy.stats as sp_st
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import warnings
import matplotlib as mpl

#Convert numeric value to .3 string to put it in title and so on
def nice_looking_str(x, n=2):
    return str(round(x*10**n)/10**n)

from scipy.stats import chi2
def get_cov_ellipse(cov, centre, ns, **kwargs):
    """
    Return a matplotlib Ellipse patch representing the covariance matrix
    cov centred at centre and scaled by the factor nstd.

    """

    # Find and sort eigenvalues and eigenvectors into descending order
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # The anti-clockwise angle to rotate our ellipse by
    vx, vy = eigvecs[:,0][0], eigvecs[:,0][1]
    theta = np.arctan2(vy, vx)

    # Width and height of ellipse to draw
    s = chi2(2).ppf(ns) #4.605   #95%: 5.991  90%: 4.605
    width, height = np.sqrt(eigvals*s)*2

    return mpl.patches.Ellipse(xy=centre, width=width, height=height,
                   angle=np.degrees(theta), **kwargs)


def confidence_ellipse(centre, cov, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = centre[0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = centre[1]

    transf = mpl.transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)



def statisical_summary(x):
    return dict(zip(['Mean', 'Max', 'Min', 'Value'], [np.mean(x), max(x), min(x), x]))


def get_boundingBox(data):
    N, D = data.shape

    boundingBox = np.asarray([np.min(data.T[0]), np.max(data.T[0])])

    for d in range(1, D):
        boundingBox = np.vstack((boundingBox, np.asarray([np.min(data.T[d]), np.max(data.T[d])])))

    ranges = boundingBox[:, 1] - boundingBox[:, 0]
    shortestRange = np.min(ranges, axis=0)
    boundingBox = boundingBox.T
    return boundingBox, shortestRange


def weighted_median(data, weights):
    """
    Args:
      data (list or numpy.array): data
      weights (list or numpy.array): weights
    """
    data, weights = np.array(data).squeeze(), np.array(weights).squeeze()
#    s_data, s_weights = map(np.array, zip(*sorted(zip(data, weights))))
    s_data = data
    s_weights = weights
    midpoint = 0.5 * sum(s_weights)
    if any(weights > midpoint):
        w_median = (data[weights == np.max(weights)])[0]
    else:
        cs_weights = np.cumsum(s_weights)
        idx = np.where(cs_weights <= midpoint)[0][-1]
        if cs_weights[idx] == midpoint:
            w_median = np.mean(s_data[idx:idx+2])
        else:
            w_median = s_data[idx+1]
    return w_median


def plot_covariance_ellipse(ax, mu, sigma, p=0.9, color="k"):
    s = sp_st.chi2.ppf(p, 2)
    sigma = s*sigma

    #Compute eigenvalues/eigenvectors of the covariance matrix
    eigvals, eigvects = np.linalg.eigh(sigma)

    x, y = eigvects[:, 0]
    theta = np.degrees(np.arctan2(y, x))

    # Eigenvalues gives access to great and little axes of ellipse
    w, h = 2 * np.sqrt(eigvals)
    ellipse = Ellipse(mu, w, h, theta, color=color)
    ellipse.set_clip_box(ax.bbox)
    ellipse.set_alpha(0.5)
    ax.add_artist(ellipse)



def plot_ellipsoide(ax, mu, sigma, color="k"):
    # Find the rotation matrix and the great axis values by computing an SVD
    _, s, rot = np.linalg.svd(sigma)
    radii = np.sqrt(s)

    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, 10)
    v = np.linspace(0, np.pi, 10)

    # Cartesian coordinates that correspond to the spherical angles:
    # (this is the equation of an ellipsoid):
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    #Apply the rotation matrix and center the ellipsoid on mu
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rot) + mu

    # Plot:
    ax.plot_surface(x, y, z, alpha=0.5, color=color)



def exportGraphAsTxt(file, edges):
    f = open(file, "w+")
    for e in edges:
        f.write("%d\t%d\n" %(e[0], e[1]))
    f.close()
    return


def getFigure(figSize, axSize):
    fig = plt.figure(figsize=figSize)
    ax = fig.add_axes(axSize, aspect='equal')

    return fig, ax


def importGraphAsTxt(file):
    f = open(file, "r")
    e = []
    for line in f:
        a, b = line.split()
        e.append(tuple((int(a), int(b))))
    f.close()
    return e



def density_map_NGP(posx,posy,Nx,Ny,wgt=None):
    if wgt == None:
        wgt = np.ones(len(posx))
    map_2D=np.zeros((Nx,Ny))
    npart=len(posx)
    xmin=min(posx)-1
    xmax=max(posx)+1
    ymin=min(posy)-1
    ymax=max(posy)+1

    dx=(xmax-xmin)/(1.0*Nx)
    dy=(ymax-ymin)/(1.0*Ny)

    print('xmin xmax (Mpc/h) {0} {1}'.format(xmin,xmax))
    print('ymin ymax (Mpc/h) {0} {1}'.format(ymin,ymax))
    print('dx dy (kpc/h) {0} {1}'.format(dx*1e3,dy*1e3))

    idx=(posx-xmin)/(dx)
    idy=(posy-ymin)/(dy)

    idx= np.rint(idx).astype(int)
    idy= np.rint(idy).astype(int)

    idx[idx>(Nx-1)]=Nx-1
    idx[idx<0]=0
    idy[idy>(Ny-1)]=Ny-1
    idy[idy<0]=0

    for ip in range(npart):
#          print('ip {0} / {1} test {2} , {3}'.format(ip,npart,idx[ip],idy[ip]))
         ix=idx[ip]
         iy=idy[ip]
         map_2D[ix,iy]=map_2D[ix,iy]+wgt[ip]

    map_2D=map_2D/dx/dy
    return map_2D



def plot_map(map_2D, ax=None, cbarTitle='$1+\delta$', cmap='viridis', extent=[], fig=None, cbarPos=[]):
     from matplotlib.colors import LogNorm
     if ax == None:
         fig, ax = plt.subplots()
     if len(extent)>0:
         image = ax.imshow(map_2D, origin='lower', cmap=cmap, extent=extent, aspect='auto')
         image = ax.imshow(map_2D, origin='lower', norm=LogNorm(vmin=np.min(map_2D[map_2D>0]), vmax=1), cmap=cmap, extent=extent, aspect='auto')
     else:
         image = ax.imshow(map_2D, origin='lower', cmap=cmap, aspect='auto')
         image = ax.imshow(map_2D, origin='lower', norm=LogNorm(vmin=np.min(map_2D[map_2D>0]), vmax=1), cmap=cmap, aspect='auto')

     if len(cbarPos) > 0:
         colorbar_ax = fig.add_axes(cbarPos)
         cbar = plt.colorbar(image, cax=colorbar_ax)
         cbar.set_label(cbarTitle) #, rotation=270)
     else:
         cbar = plt.colorbar(image, ax=ax)
         cbar.set_label(cbarTitle) #, rotation=270)
#         cbar.ax.get_yaxis().labelpad = 15
     return ax



def hessian(x):
    """
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x)
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype)
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k)
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return x_grad, hessian.T


def dot(v,w):
    x,y,z = v
    X,Y,Z = w
    return x*X + y*Y + z*Z

def length(v):
    x,y,z = v
    return np.sqrt(x**2 + y**2 + z**2)

def vector(b,e):
    x,y,z = b
    X,Y,Z = e
    return (X-x, Y-y, Z-z)

def unit(v):
    x,y,z = v
    mag = length(v)
    return (x/mag, y/mag, z/mag)

def distance(p0,p1):
    return length(vector(p0,p1))

def scale(v,sc):
    x,y,z = v
    return (x * sc, y * sc, z * sc)

def add(v,w):
    x,y,z = v
    X,Y,Z = w
    return (x+X, y+Y, z+Z)

#Thanks to http://www.fundza.com/vectors/point2line/index.html
def pnt2line(pnt, start, end):
    #Not vector form
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt.T)
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0/line_len)
    t = dot(line_unitvec, pnt_vec_scaled)

    if np.size(t) > 1:
        t[t<0.0] = 0.0
        t[t>1.0] = 1.0
    else:
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0

    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)
    nearest = add(nearest, start)

    return (dist, nearest)


def vectorized_pnt2line(pnt, start, end):

    #Vector form
    line_vec = end - start
    pnt_vec = pnt.T - start
    line_len = np.array([np.linalg.norm(line_vec, axis=1)]).T
    line_unitvec = line_vec / line_len
    pnt_vec_scaled = pnt_vec * 1.0 / line_len
    t = np.array([np.sum(line_unitvec*pnt_vec_scaled, axis=1)]).T

    if np.size(t) > 1:
        t[t<0.0] = 0.0
        t[t>1.0] = 1.0
    else:
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0

    #Vector form
    nearest = line_vec * t
    dist = np.linalg.norm(pnt_vec - nearest, axis=1)
    nearest = nearest + start

    return (dist, nearest)


def create_ranges(starts, stops, N):
    steps = (1.0/N).reshape(len(starts),1)*(stops - starts)
    interpolations = []
    for (i,w) in enumerate(N):
        vec = np.kron(np.arange(int(w)+1), steps[i]).reshape((int(w)+1, len(starts.T)))
        res = np.vstack((vec + starts[i], stops[i]))
        interpolations.append(res)
    return np.vstack(interpolations)

