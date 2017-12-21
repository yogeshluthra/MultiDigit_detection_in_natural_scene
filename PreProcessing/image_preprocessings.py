import numpy as np
import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt

class ProcessImageDF(object):
    """Process image dataframe"""
    def __init__(self, df, max_recognizable_digits=5):
        self.df=df
        self.columns = ['N', 'bb', 'image', 'y']
        self.max_recognizable_digits=max_recognizable_digits
        for column in self.columns: assert column in self.df.columns.values

    def get_bbox_corners(self, i_image):
        """finds bbox top left and bottom right coords from image indexed at i_image in dataframe
        :returns list(bbox_xmin, bbox_xmax, bbox_ymin, bbox_ymax)"""
        bboxes = self.df.iloc[i_image]['bb']
        image = self.df.iloc[i_image]['image']
        N = self.df.iloc[i_image]['N']
        bbox_xmin, bbox_xmax = image.shape[1], 0
        bbox_ymin, bbox_ymax = image.shape[0], 0
        for j in range(N):
            if j < self.max_recognizable_digits:
                x_left, x_right = bboxes[j, 0], bboxes[j, 0] + bboxes[j, 2]
                y_top, y_bot = bboxes[j, 1], bboxes[j, 1] + bboxes[j, 3]
                bbox_xmin = min(bbox_xmin, x_left)
                bbox_xmax = max(bbox_xmax, x_right)
                # TODO: find bbox_ymin, bbox_ymax
                bbox_ymin = min(bbox_ymin, y_top)
                bbox_ymax = max(bbox_ymax, y_bot)

        # - extract a square (or close to square) cropped image from original image
        bbox_xmin, bbox_xmax = int(round(min(bbox_xmin, bbox_xmax))), int(round(max(bbox_xmin, bbox_xmax)))
        bbox_ymin, bbox_ymax = int(round(min(bbox_ymin, bbox_ymax))), int(round(max(bbox_ymin, bbox_ymax)))
        return([bbox_xmin, bbox_xmax, bbox_ymin, bbox_ymax])

    def get_bboxCoords_afterRotation(self, M, bbox_xmin, bbox_xmax, bbox_ymin, bbox_ymax):
        """M: rotation matrix"""
        bbox_org = np.array([[bbox_xmin,    bbox_xmin,  bbox_xmax,  bbox_xmax],
                             [bbox_ymin,    bbox_ymax,  bbox_ymin,  bbox_ymax],
                             [1.,           1.,         1.,         1.]]).astype(np.float32)

        # New coords under rotation
        bbox_rotated = (np.round(M.dot(bbox_org))).astype(np.int)
        bbox_xmin, bbox_ymin = np.min(bbox_rotated, axis=1)
        bbox_xmax, bbox_ymax = np.max(bbox_rotated, axis=1)
        return ([bbox_xmin, bbox_xmax, bbox_ymin, bbox_ymax])

    def getDF_with_resized_images_aroundBB(self, size=32, max_recognizable_digits=5, max_randomExpansion=(0.4,0.6), augment_With_randomExpansion=False, rotationRange=0):
        """find bbox around sequence as bbox_xmin = min left coord bbox_xmax = max of all x+w.
        New bbox_w = bbox_xmax - bbox_xmin
        x_center = bbox_xmin + bbox_w/2
        then extend bbox w between 100-125% of original width (check edge cases)
        resize to 32x32
        augment_With_randomExpansion: Add randomly expanded boxes"""
        x_lefts, x_rights = [], []

        # - construct new dataframe
        columns_new = ['image', 'N', 'y', 'box_size', 'aspect_WbyH']
        df_img_resized = pd.DataFrame(columns=columns_new)

        w_streches = np.arange(max_randomExpansion[0], max_randomExpansion[1]+0.05, 0.05) + 1.0
        for i in range(self.df.shape[0]): # for each dataset
            N = self.df.iloc[i]['N']
            image = self.df.iloc[i]['image']
            y = self.df.iloc[i]['y']
            bbox_xLL, bbox_xUL = 0, image.shape[1]
            bbox_yLL, bbox_yUL = 0, image.shape[0]

            bbox_xmin, bbox_xmax, bbox_ymin, bbox_ymax = self.get_bbox_corners(i)

            if rotationRange>0:
                h, w = bbox_ymax - bbox_ymin, bbox_xmax - bbox_xmin
                x_rotation_center, y_rotation_center = int(round(bbox_xmin + w / 2.)), int(round(bbox_ymin + h / 2.)) # rotate about digits center
                rotationAngle = np.random.choice(np.arange(-rotationRange, rotationRange)) # random rotation
                cols, rows = bbox_xUL - bbox_xLL, bbox_yUL - bbox_yLL
                M = cv2.getRotationMatrix2D((x_rotation_center, y_rotation_center), rotationAngle, 1)
                image = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
                bbox_xmin, bbox_xmax, bbox_ymin, bbox_ymax = self.get_bboxCoords_afterRotation(M,
                                                                                               bbox_xmin, bbox_xmax,
                                                                                               bbox_ymin, bbox_ymax)

            h, w = bbox_ymax-bbox_ymin, bbox_xmax-bbox_xmin
            x_center, y_center = bbox_xmin + w/2., bbox_ymin + h/2.
            #region - Find a random cropped image 1x-1.45x of found bbox
            box_size = np.int(np.round(1.0 * max(h, w))) # find bounding box for cropping and randomly expand it between 0-25%
            x_tl, y_tl = int(round(max(bbox_xLL, x_center-box_size/2.))), int(round(max(bbox_yLL, y_center-box_size/2.)))
            x_br, y_br = int(round(min(bbox_xUL, x_center+box_size/2.))), int(round(min(bbox_yUL, y_center+box_size/2.)))
            aspect_WbyH = (x_br - x_tl)*1. / (y_br - y_tl) # w by h
            image_resized = cv2.resize(image[y_tl:y_tl+box_size, x_tl:x_tl+box_size], (size, size))

            df2 = pd.DataFrame(dict(zip(columns_new, [[item] for item in [image_resized, N, y, box_size, aspect_WbyH]])))
            df_img_resized = df_img_resized.append(df2)
            #endregion
            # region - Augment with Random box expansion
            if augment_With_randomExpansion:
                box_size = np.int(np.round(np.random.choice(w_streches) * max(h, w)))  # find bounding box for cropping and randomly expand it between 0-25%
                x_tl, y_tl = int(round(max(bbox_xLL, x_center - box_size / 2.))), int(round(max(bbox_yLL, y_center - box_size / 2.)))
                x_br, y_br = int(round(min(bbox_xUL, x_center + box_size / 2.))), int(round(min(bbox_yUL, y_center + box_size / 2.)))
                aspect_WbyH = (x_br - x_tl) * 1. / (y_br - y_tl)  # w by h
                image_resized = cv2.resize(image[y_tl:y_tl + box_size, x_tl:x_tl + box_size], (size, size))

                df2 = pd.DataFrame(dict(zip(columns_new, [[item] for item in [image_resized, N, y, box_size, aspect_WbyH]])))
                df_img_resized = df_img_resized.append(df2)
            # endregion

        return df_img_resized

def img_meanSubtraction_compressTo1p0(images):
    """ subtract mean from each image and compress values between 0-1.0"""
    images = images.astype('float32') / 255.
    return images - images.mean(axis=(-3,-2,-1), keepdims=1)

def rotate_image(image, rotationAngle):
    rows, cols, n_channels = image.shape
    M = cv2.getRotationMatrix2D((cols // 2, rows // 2), rotationAngle, 1)
    return(cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE))

def get_rotated_images(df_32x32, rotationRange):
    """
    rotate all images in input dataframe df_32x32 while sampling rotation angle uniformly between +-rotationRange
    :param df_32x32: input data frame
    :param rotationRange: rotation range
    :return: dataframe
    """
    columns = ['image', 'N', 'y', 'box_size', 'aspect_WbyH']
    assert len(columns) == len(df_32x32.columns)  # checks
    for attr in columns: assert attr in df_32x32.columns
    df_new = pd.DataFrame(columns=columns)
    for i_image in range(df_32x32.shape[0]):
        image = df_32x32.iloc[i_image]['image']
        N = df_32x32.iloc[i_image]['N']
        y = df_32x32.iloc[i_image]['y']
        box_size = df_32x32.iloc[i_image]['box_size']
        aspect_WbyH = df_32x32.iloc[i_image]['aspect_WbyH']

        rotationAngle = np.random.choice(np.arange(-rotationRange, rotationRange))
        rows, cols, n_channels = image.shape
        M = cv2.getRotationMatrix2D((cols // 2, rows // 2), rotationAngle, 1)
        dst = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)

        df2 = pd.DataFrame(dict(zip(columns,
                                    [[item] for item in [dst, N, y, box_size, aspect_WbyH]])))
        df_new = df_new.append(df2)

    return(df_new)

# - Compress image to min_dim=200 but no aspect ratio change
def compress_image(image, minDim_size=200):
    """minDim_size: size of minimum dimension"""
    h, w, d = image.shape
    need_resize = False
    if h < w:
        if h > minDim_size:
            aspect_rat = w * 1. / h
            h, w = minDim_size, np.int(np.round(minDim_size*1. * aspect_rat))
            need_resize = True
    else:
        if w > minDim_size:
            aspect_rat = h * 1. / w
            h, w = np.int(np.round(minDim_size*1. * aspect_rat)), minDim_size
            need_resize = True
    if need_resize: image = cv2.resize(image, (w, h))
    return (image)


def sliding_window_crops(image, bbox, box_size, resizeTO=(32,32), strides=(4, 4), minDim_size=200, debug=False):
    """stride: in x,y directions. minDim_size: size of minimum dimension"""
    x0,y0,w,h=bbox
    bx, by = box_size
    w,h=max(w,bx+1), max(h,by+1)
    assert h > by and w > bx
    if debug: plt.imsave('debug/crop__bbox_{0}_{1}_{2}_{3}__atSize_{4}_{5}.png'.format(x0,y0,w,h,bx,by), image[y0:y0+h, x0:x0+w][...,[2,1,0]]) # TODO: remove

    def sliding_window_crop_locations(h, w, box_size, strides):
        """height, width, box_size, strides. Returns crop locations"""
        sx, sy = strides
        bx, by = box_size
        xx, yy = np.meshgrid((x0+np.arange(w - bx))[::sx], (y0+np.arange(h - by))[::sy])
        return (np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1))))  # [[x,y]] locations

    crop_locs = sliding_window_crop_locations(h, w, box_size, strides)
    crops = [cv2.resize(image[y:y + by, x:x + bx], resizeTO) for x, y in crop_locs]

    if debug:
        test_image=image.copy() # TODO: Remove
        for x, y in crop_locs:
            test_image = cv2.rectangle(test_image, (x,y),(x+bx,y+by),(0,0,255),1)
        plt.imsave('debug/slides__bbox_{0}_{1}_{2}_{3}__atSize_{4}_{5}.png'.format(x0,y0,w,h,bx,by), test_image[...,[2,1,0]])

    w,h=resizeTO
    return (np.vstack(crops).reshape(-1, h, w, 3), crop_locs)

def sliding_window_animation(image, crop_locs, box_size, hit_miss):
    """
    crop_locs: [[x,y]] locations
    hit_miss: boolean array. If true, green bbox. Else blue bbox"""
    for i in range(crop_locs.shape[0]):
        img = image.copy()
        pt1 = crop_locs[i]
        pt1, pt2 = tuple(pt1),     tuple(pt1+box_size)
        color = (0,255,0) if hit_miss[i] else (255,0,0)
        img = cv2.rectangle(img, pt1, pt2, color, 1)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def display_hits(image, bboxes_tl, bbox_size):
    img=image.copy()
    w,h=bbox_size
    for x,y in bboxes_tl:
        img=cv2.rectangle(img, (x,y), (x+w, y+h),(0,255,0),1)
    return(img)

if __name__=="__main__":
    test_image = cv2.imread('../data/test_images/house1.jpg', 1)

    box_size=[40,40]
    compressed_image = compress_image(test_image, minDim_size=200)
    crops, crop_locs = sliding_window_crops(compressed_image, box_size, minDim_size=200)

    hit_miss = np.zeros(crops.shape[0], dtype=np.bool)
    sliding_window_animation(compressed_image, crop_locs, box_size, hit_miss)






