import numpy as np
import pandas as pd
from PreProcessing import image_preprocessings as PreProcImg

def connectedComponents_byDistance(df_all_hits, strides_used, max_strides_for_colocated_points=1., debug=False):
    """ Basically a variant of Single Link Clustering
    all points which were identified as possible candidates by CNN and non-Maximum suppression strategy"""
    # - all_hits (dataframe):
    #   - locs: x,y locations
    #   - N: number of digits
    #   - digs: digits
    #   - bbox: bounding boxes
    vSt = lambda x: np.vstack(x)
    locs,N,digs,bbox = vSt(df_all_hits['locs'].values), df_all_hits['N'].values, vSt(df_all_hits['digits'].values), vSt(df_all_hits['bbox'].values)
    x, y = locs[:,0], locs[:,1]
    bx, by = np.vstack(bbox)[:,0], np.vstack(bbox)[:,1]
    x_centers, y_centers = x + bx//2, y + by//2
    assert x.size>=2

    x_max, y_max = strides_used
    def colocated(pt1, pt2):
        x0, y0 = pt1
        x1, y1 = pt2
        return((abs(x0 - x1) <= max_strides_for_colocated_points * x_max) and
               (abs(y0 - y1) <= max_strides_for_colocated_points * y_max))

    # Algorithm for localization:
    # - go over all pairs (O(n^2))
    #   - if distance criteria satisfied
    #     - parent claim child (tag child as belonging to parent)
    # - go over all points linearly (O(n.lg(n))) (this is unoptimized currently)
    #   - if child==parent
    #     - add this node as key of dictionary and add itself as first element of list, which is value of that key
    #   - else
    #     - append this node to root of connections up the chain (parent-of-parents)
    # - output dictionary
    # Could be extended with additional EM on found largest cluster via above algo (SLC), to find if point distribution is multi-modal
    # (e.g. 17 and 45, although nearby as 1745, are detected separately due to detection window not being large enough. This could hint at extending window.)

    # - Initialize
    parents = [i for i in range(0, x.size)] # all points are their parents initially
    # - go over all pairs (O(n^2))
    for i in range(0, x.size):
        for j in range(i+1, x.size):
            if colocated((x_centers[i], y_centers[i]),
                         (x_centers[j], y_centers[j])):
                parents[j] = i

    def root(j):
        while j!=parents[j]: j=parents[j] # could use path compression
        return j

    clusters={}
    for i in range(0, x.size):
        if i==root(i):
            assert i not in clusters
            clusters[i]=[[x[i], y[i], N[i], digs[i], bbox[i]]]
        else:
            assert root(i) in clusters
            clusters[root(i)].append([x[i], y[i], N[i], digs[i], bbox[i]])

    if debug:
        for k in clusters.keys():
            print('Cluster:', k)
            for i in range(len(clusters[k])):
                print(clusters[k][i][0], clusters[k][i][1])
            print()
    return clusters



def getPredictions(model, test_images):
    colV = lambda x: x.reshape(-1, 1)
    test_images_processed = PreProcImg.img_meanSubtraction_compressTo1p0(test_images)
    out_probas = model.predict(test_images_processed)

    p_N, p_y_1, p_y_2, p_y_3, p_y_4, p_y_5 = [proba.max(axis=-1) for proba in out_probas]  # max probabilities
    p_outputs = np.hstack((colV(p_N), colV(p_y_1), colV(p_y_2), colV(p_y_3), colV(p_y_4), colV(p_y_5)))

    N, y_1, y_2, y_3, y_4, y_5 = [proba.argmax(axis=-1) for proba in out_probas]
    outputs = np.hstack((colV(N), colV(y_1), colV(y_2), colV(y_3), colV(y_4), colV(y_5)))

    N, y = outputs[:, 0], outputs[:, 1:]
    y[y == 10] = -1
    y_labels = np.array([''.join([str(y[j, i]) for i in range(N[j])]) for j in range(y.shape[0])])

    return (N, y, p_outputs, y_labels)


def get_hit_miss(N, y, probs, pEachDig_LL=0.3, pNdig_LL=0.98):
    hit_miss = []
    for i in np.arange(y.shape[0]):
        if N[i] > 0:
            true_or_not = np.all(y[i, :N[i]] >= 0) and \
                          np.all(probs[i, 1:N[i] + 1] > pEachDig_LL) and \
                          probs[i, 0] > pNdig_LL  # earlier 0.4, 0.98 worked well
            hit_miss.append(true_or_not)
        else:
            hit_miss.append(False)
    return (np.array(hit_miss))
