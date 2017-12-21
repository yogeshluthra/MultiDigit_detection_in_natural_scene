"""localization via clustering and detection"""
from PreProcessing import image_preprocessings as PreProcImg
from utilities import segmentation as Seg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
from glob import glob

def findMostFreq(vec):
    Yunique, Yindices = np.unique(vec, return_inverse=True)
    return(Yunique[np.argmax(np.bincount(Yindices))] ) # finds most likely (or frequent)

def findMostLikelyLabelIn(clusters):

    # 1 - find biggest cluster
    # 2 - find biggest N within biggest cluster (as we believe Ndigits is most accurately found)
    # 3 - find cluster center as mean of x,y of points having that largest N
    # 4 - find mean bbox from points having that largest N
    # 5 - for all digits having that biggest N
    # 6   - find most likely digit_i at position i
    biggest_cluster, n_members = None, -np.inf
    for k, v in clusters.items(): # 1
        if len(v) > n_members: biggest_cluster, n_members = k, len(v)

    N = np.array([clusters[biggest_cluster][i][2] for i in range(n_members)])
    Nlargest = np.max(N) # 2

    xs_largestN = np.array([clusters[biggest_cluster][i][0] for i in range(n_members)])[N == Nlargest]
    ys_largestN = np.array([clusters[biggest_cluster][i][1] for i in range(n_members)])[N == Nlargest]
    x_centroid, y_centroid = np.int(np.round(np.mean(xs_largestN))), np.int(np.round(np.mean(ys_largestN))) # 3

    bbox_largestN = np.round(np.vstack([clusters[biggest_cluster][i][4] for i in range(n_members)])[N == Nlargest].mean(axis=0)).astype(np.int) # 4

    digit_largestN = np.vstack([clusters[biggest_cluster][i][3] for i in range(n_members)])[N == Nlargest] # 5
    digit_mostLikely = list(findMostFreq(digit_largestN[:,i]) for i in range(Nlargest))

    return(Nlargest, (x_centroid, y_centroid), bbox_largestN, digit_mostLikely)

def localize_digits(model, test_image, debug=False,
                    minDim_size = 500,
                    strides = (2, 4),
                    min_bSize_for_pyramids=24,
                    expansion_factor=1.5, n_expansions=2,
                    max_strides_for_colocated_points=3.):
    """localize digits in test_image using model. Note the image is compressed such that min dimension=mDim_size"""
    """find clusters where digits could be"""
    # Algorithm for pyramids:
    # - Initialize bbox_queue to whole image frame
    # - for bbox_size in bbox sizes:
    # -   while bbox_queue:
    # -     bbox = bbox_queue.pop()
    # -     get crops and crop_locs using bbox
    # -     getPredictions on those crops
    # -     append all hits to dataframe
    # -   use dataframe of all hits so far to form clusters (SLC)
    # -   bbox_queue filled with bounding box of top 3 clusters

    if debug:
        if os.path.exists('./debug'):
            for aFile in glob('./debug/*'): os.remove(aFile)
        else:                           os.makedirs('./debug')

    df_hits = pd.DataFrame(columns=['locs', 'N', 'digits', 'bbox'])

    image_for_prediction = PreProcImg.compress_image(test_image, minDim_size=minDim_size)
    h,w,_=image_for_prediction.shape
    bbox_queue = [(0,0,w,h)] # x,y,w,h

    bSize = min_bSize_for_pyramids
    clusters=None
    for i in range(n_expansions+1):
        bbox_size = [bSize, bSize]
        print('for box size ',bbox_size)
        j=0
        while bbox_queue:
            print('boxes in queue ', bbox_queue)
            bbox = bbox_queue.pop()
            crops, crop_locs = PreProcImg.sliding_window_crops(image_for_prediction,
                                                               bbox, bbox_size,
                                                               resizeTO=(32, 32),
                                                               strides=strides,
                                                               minDim_size=minDim_size, debug=debug)

            N, y, p_outputs, y_labels = Seg.getPredictions(model, crops)
            hit_miss = Seg.get_hit_miss(N, y, p_outputs, pEachDig_LL=0.3, pNdig_LL=0.98)  # earlier 0.4, 0.98 worked well

            if debug:
                img_with_hit_miss = PreProcImg.display_hits(image_for_prediction, crop_locs[hit_miss], bbox_size)
                plt.imsave('debug/test_out_allhits_bSize{0}_{1}.png'.format(bSize, j),
                           img_with_hit_miss[..., [2, 1, 0]]) # TODO: remove


            if debug:
                print('for box size: ',bSize)
                for lab, prob, crop_loc in zip(y_labels[hit_miss], p_outputs[hit_miss], crop_locs[hit_miss]):
                    print(lab, prob, crop_loc)
                print(np.hstack((y_labels[hit_miss].reshape(-1,1),p_outputs[hit_miss,0:])))

            for i_th_hit in range(N[hit_miss].size):  # stack hits one by one in pandas df
                df_hits = df_hits.append(pd.DataFrame(data={'locs':[(crop_locs[hit_miss])[i_th_hit]],
                                                            'N':[(N[hit_miss])[i_th_hit]],
                                                            'digits':[(y[hit_miss])[i_th_hit]],
                                                            'bbox':[bbox_size]}))
            j += 1
        clusters = Seg.connectedComponents_byDistance(df_hits, strides,
                                                      max_strides_for_colocated_points=3.,
                                                      debug=debug)  # x,y,N,digits,bbox
        bSize = int(round(bSize * expansion_factor))  # new box size
        bbox_queue = get_bboxes_from_topdense_clusters(clusters, img_UpperLimits=(h,w), max_clusters=3, new_bSize=bSize)

        if debug:
            test_bbox_img = image_for_prediction.copy() # TODO: remove
            for bbox in bbox_queue:
                x_min, y_min, w_box, h_box = bbox
                test_bbox_img = cv2.rectangle(test_bbox_img, (x_min, y_min), (x_min+w_box, y_min+h_box),(0,0,255),1)
            plt.imsave('debug/test_bbox_at_{}.png'.format(i), test_bbox_img[...,[2,1,0]])

    return(clusters, image_for_prediction, df_hits)

def get_bboxes_from_topdense_clusters(clusters, img_UpperLimits=None, max_clusters=3, new_bSize=32):
    if img_UpperLimits: h,w=img_UpperLimits
    else: h,w = np.inf, np.inf

    bSize = new_bSize
    # x,y,N,digs,box
    cluster_names, cluster_lengths = [], []
    for k in clusters.keys():
        cluster_names.append(k)
        cluster_lengths.append(len(clusters[k]))
    cluster_names,cluster_lengths = np.array(cluster_names), np.array(cluster_lengths)
    top_n_cluster_names = cluster_names[np.argsort(cluster_lengths)[-max_clusters:]]
    bboxes=[]
    for cluster_name in top_n_cluster_names:
        xs_in_this_cluster = np.array([item[0] for item in clusters[cluster_name]])
        ys_in_this_cluster = np.array([item[1] for item in clusters[cluster_name]])
        x_box_sizes_in_this_cluster = np.array([item[4][0] for item in clusters[cluster_name]])
        y_box_sizes_in_this_cluster = np.array([item[4][1] for item in clusters[cluster_name]])
        x_min, y_min = np.min(xs_in_this_cluster), np.min(ys_in_this_cluster)
        x_max, y_max = np.max(xs_in_this_cluster + x_box_sizes_in_this_cluster), \
                       np.max(ys_in_this_cluster + y_box_sizes_in_this_cluster)
        x_min, y_min = max(0, x_min - bSize), max(0, y_min - bSize)
        x_max, y_max = min(w, x_max + bSize), min(h, y_max + bSize)
        bboxes.append((x_min, y_min, x_max-x_min, y_max-y_min))
    return(bboxes)



def detect_digits(clusters, compressed_image, save_in_file, debug=False):
    """detect digits using clusters and output final image with bounding box"""
    Nlargest, (x_centroid, y_centroid), bbox_largestN, digits_mostLikely = findMostLikelyLabelIn(clusters)
    print('centroid',x_centroid, y_centroid,'\n')
    label_mostLikely = ''.join(str(i) for i in digits_mostLikely)

    compressed_image_with_rect = cv2.rectangle(compressed_image,
                                               (x_centroid, y_centroid),
                                               (x_centroid + bbox_largestN[0], y_centroid + bbox_largestN[1]),
                                               (0, 255, 0), 2)
    # x_textLoc, y_textLoc = compressed_image_with_rect.shape[1] // 2, \
    #                        compressed_image_with_rect.shape[0] // 2
    x_textLoc, y_textLoc = x_centroid, y_centroid-10
    compressed_image_with_rect = cv2.putText(compressed_image_with_rect,
                                             label_mostLikely,
                                             (x_textLoc, y_textLoc),
                                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    plt.imsave(save_in_file, compressed_image_with_rect[..., [2, 1, 0]])
    if debug:
        if not os.path.exists('./debug'): os.makedirs('./debug')
        plt.imsave('debug/test_out_digitsDetected.png', compressed_image_with_rect[..., [2, 1, 0]])
        print(digits_mostLikely)