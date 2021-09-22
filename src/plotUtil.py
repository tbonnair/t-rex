from sklearn.neighbors import KDTree
import plotly.graph_objects as go
import numpy as np
import networkAlgo


def rotate_z(x, y, z, theta):
    w = x+1j*y
    return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z


def plotFilaments(data, m, F, edges, filaments, idxFil, radius, opacity=0):
    n = filaments[idxFil].nodes
    datapoints_fil = filaments[idxFil].datapoints

    kdtree = KDTree(data)
    meanFil = np.mean(F[n], axis=0)
    d = kdtree.query_radius([meanFil], radius)[0]

    #Positions of the particular filament
    Xe_filPart, Ye_filPart, Ze_filPart = networkAlgo.compute_XYZ_of_edges(edges[filaments[idxFil].edges], F)

    eOk = []
    fal = np.linalg.norm(F[edges] - meanFil, axis=2)
    for (it, e) in enumerate(fal):
        if e[0] < radius or e[1] < radius:
            eOk.append(it)
    eOk = np.array(eOk)
    eOk = edges[eOk]

    Xe_fil, Ye_fil, Ze_fil = networkAlgo.compute_XYZ_of_edges(eOk, F)

#    ttl = 'Filament #{:d} - Length = {:2f} - Curvature = {:2f} - #Datapoints = {:d}'.format(idxFil,
#            filaments[idxFil].length, filaments[idxFil].curvature, len(filaments[idxFil].datapoints))

    if opacity == 0:
        op = 0.8 - len(d) / len(data) + 0.05
        if op > 1:
            op = 1
        elif op < 0:
            op = 0.05
    else:
        op = opacity

    #Tracing nodes
    traceDP=go.Scatter3d(x=data[d].T[0],
                   y=data[d].T[1],
                   z=data[d].T[2],
                   mode='markers',
                   marker=dict(symbol='circle',
                                 size=m[d],
                                 line_width = 0.1,
                                 opacity = op,  #0.05
                                 color='rgb(255, 255, 255)',
                                 colorscale='Greys',
                                 ),
                   name='Datapoints',
                   )

    traceAssign=go.Scatter3d(x=data[datapoints_fil].T[0],
                   y=data[datapoints_fil].T[1],
                   z=data[datapoints_fil].T[2],
                   mode='markers',
                   marker=dict(symbol='circle',
                                 size=m[datapoints_fil],
                                 line_width = 0.1,
                                 opacity = 1,  #0.05
                                 color='rgb(0, 255, 0)',
                                 colorscale='Greys',
                                 ),
                   name='Assigned datapoints'
                   )

    #Tracing edges
    traceEdges=go.Scatter3d(x=Xe_fil,
                   y=Ye_fil,
                   z=Ze_fil,
                   mode='lines',
                   line=dict(color='rgb(255, 0, 0)',
                             width=10),
                   hoverinfo='none',
                   name='Branches'
                   )

    traceEdgesOfInterest=go.Scatter3d(x=Xe_filPart,
                   y=Ye_filPart,
                   z=Ze_filPart,
                   mode='lines',
                   line=dict(color='rgb(37, 253, 233)',
                             width=10),
                   hoverinfo='none',
                   name='Branch of interest'
                   )

    finalData = [traceDP, traceAssign, traceEdges, traceEdgesOfInterest]
    fig=go.Figure(data=finalData)

    fig.update_layout(
            scene = dict(xaxis = {'title':'X', 'color':'white', 'showbackground':False, 'showline':False, 'showgrid':False, 'zeroline':False},
                         yaxis = {'title':'Y', 'color':'white', 'showbackground':False, 'showline':False, 'showgrid':False, 'zeroline':False},
                         zaxis = {'title':'Z', 'color':'white', 'showbackground':False, 'showline':False, 'showgrid':False, 'zeroline':False}),
            paper_bgcolor = 'rgb(0, 0, 0)',
            plot_bgcolor = 'rgb(0, 0, 0)')

    return fig, meanFil



def plotAroundCenter(data, m, F, edges, center, radius, labels_data=[], colors_labels=[], labels_id=[], F_other=[], edges_other=[], additional_dp=[], size_additional=1):
    kdtree = KDTree(data)
    d = kdtree.query_radius([center], radius)[0]

    eOk = []
    fal = np.linalg.norm(F[edges] - center, axis=2)
    for (it, e) in enumerate(fal):
        if e[0] < radius or e[1] < radius:
            eOk.append(it)
    eOk = np.array(eOk)
    if len(eOk) > 0:
        eOk = edges[eOk]
    else:
        eOk = []

    Xe_fil, Ye_fil, Ze_fil = networkAlgo.compute_XYZ_of_edges(eOk, F)


    if len(F_other)>0:
        eOk_other = []
        fal_other = np.linalg.norm(F_other[edges_other] - center, axis=2)
        for (it, e) in enumerate(fal_other):
            if e[0] < radius or e[1] < radius:
                eOk_other.append(it)
        eOk_other = np.array(eOk_other)
        if len(eOk_other) > 0:
            eOk_other = edges_other[eOk_other]
        else:
            eOk_other = []

        Xe_fil_other, Ye_fil_other, Ze_fil_other = networkAlgo.compute_XYZ_of_edges(eOk_other, F_other)


    op = 0.8 - len(d) / len(data) + 0.05
    print()
    print(op)
    if op > 1:
        op = 1
    elif op < 0:
        op = 0.25

    #Tracing center
    traceCenter=go.Scatter3d(x=[center[0]],
               y=[center[1]],
               z=[center[2]],
               mode='markers',
               marker=dict(symbol='circle',
                             size=5,
                             line_width = 0.1,
                             opacity = op,
                             color='rgb(0, 255, 0)',
                             colorscale='Greys',
                             ),
               name='Center',
               )

    #Tracing edges
    traceEdges=go.Scatter3d(x=Xe_fil,
                   y=Ye_fil,
                   z=Ze_fil,
                   mode='lines',
                   line=dict(color='rgb(255, 0, 0)',
                             width=10),
                   hoverinfo='none',
                   name='Branches'
                   )

    finalData = []

    if len(labels_data) == len(data):
#        diff_lab = np.unique(labels_data[labels_data>=0])
        diff_lab = np.arange(0, max(labels_data)+1).astype(int)
        if len(diff_lab) > 4:
            raise Exception('No more than 4 labels allowed, plotting no colors')

            traceDP=go.Scatter3d(x=data[d].T[0],
                   y=data[d].T[1],
                   z=data[d].T[2],
                   mode='markers',
                   marker=dict(symbol='circle',
                                 size=m[d],
                                 line_width = 0.1,
                                 opacity = op,
                                 color='rgb(255, 255, 255)',
                                 colorscale='Greys',
                                 ),
                   name='Datapoints',
                   )
            finalData.append(traceDP)

        else:
            for ilab in diff_lab:
                idx = np.where(labels_data==ilab)[0]
                idx_tokeep = np.intersect1d(idx, d)

                traceDP = go.Scatter3d(x=data[idx_tokeep].T[0],
                   y=data[idx_tokeep].T[1],
                   z=data[idx_tokeep].T[2],
                   mode='markers',
                   marker=dict(symbol='circle',
                                 size=m[idx_tokeep],
                                 line_width = 0.1,
                                 opacity = op,
                                 color=colors_labels[ilab],
                                 colorscale='Greys',
                                 ),
                   name=labels_id[ilab],
                   )
                finalData.append(traceDP)
    else:
        traceDP=go.Scatter3d(x=data[d].T[0],
               y=data[d].T[1],
               z=data[d].T[2],
               mode='markers',
               marker=dict(symbol='circle',
                             size=m[d],
                             line_width = 0.1,
                             opacity = op,
                             color='rgb(255, 255, 255)',
                             colorscale='Greys',
                             ),
               name='Datapoints',
               )
        finalData.append(traceDP)

    if len(F_other) > 0:
        traceEdges_other=go.Scatter3d(x=Xe_fil_other,
                   y=Ye_fil_other,
                   z=Ze_fil_other,
                   mode='lines',
                   line=dict(color='rgb(255, 127, 0)',
                             width=10),
                   hoverinfo='none',
                   name='Branches'
                   )
        finalData.append(traceEdges_other)

    if len(additional_dp) > 0:
        traceAddDP=go.Scatter3d(x=additional_dp.T[0],
               y=additional_dp.T[1],
               z=additional_dp.T[2],
               mode='markers',
               marker=dict(symbol='circle',
                             size=size_additional,
                             line_width = 1,
                             opacity = 1,
                             color='rgb(37, 253, 233)',
                             colorscale='Greys',
                             ),
               name='Datapoints',
               )
        finalData.append(traceAddDP)

    finalData.append(traceEdges)
    finalData.append(traceCenter)

    fig=go.Figure(data=finalData)

    fig.update_layout(
            scene = dict(xaxis = {'title':'X', 'color':'white', 'showbackground':False, 'showline':False, 'showgrid':False, 'zeroline':False},
                         yaxis = {'title':'Y', 'color':'white', 'showbackground':False, 'showline':False, 'showgrid':False, 'zeroline':False},
                         zaxis = {'title':'Z', 'color':'white', 'showbackground':False, 'showline':False, 'showgrid':False, 'zeroline':False}),
            paper_bgcolor = 'rgb(0, 0, 0)',
            plot_bgcolor = 'rgb(0, 0, 0)')

    return fig