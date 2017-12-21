#!/usr/bin/python

# Ref:https://confluence.slac.stanford.edu/display/PSDM/How+to+access+HDF5+data+from+Python 

"""Courtesy: https://github.com/prijip/Py-Gsvhn-DigitStruct-Reader/blob/master/digitStruct.py"""

import h5py
import numpy as np
import cv2
import os
import pandas as pd
import sys


#
# Bounding Box
#
class BBox:
    def __init__(self):
        self.label = ""     # Digit
        self.left = 0
        self.top = 0
        self.width = 0
        self.height = 0

class DigitStruct:
    def __init__(self):
        self.name = None    # Image file name
        self.bboxList = None # List of BBox structs

# Function for debugging
def printHDFObj(theObj, theObjName):
    isFile = isinstance(theObj, h5py.File)
    isGroup = isinstance(theObj, h5py.Group)
    isDataSet = isinstance(theObj, h5py.Dataset)
    isReference = isinstance(theObj, h5py.Reference)
    print("{}".format(theObjName))
    print("    type(): {}".format(type(theObj)))
    if isFile or isGroup or isDataSet:
        # if theObj.name != None:
        #    print "    name: {}".format(theObj.name)
        print("    id: {}".format(theObj.id))
    if isFile or isGroup:
        print("    keys: {}".format(theObj.keys()))
    if not isReference:
        print("    Len: {}".format(len(theObj)))

    if not (isFile or isGroup or isDataSet or isReference):
        print(theObj)

#
# Reads a string from the file using its reference
#
def readString(strRef, dsFile):
    strObj = dsFile[strRef]
    str = ''.join(chr(i) for i in strObj)
    return str

#
# Reads an integer value from the file
#
def readInt(intArray, dsFile):
    intRef = intArray[0]
    isReference = isinstance(intRef, h5py.Reference)
    intVal = 0
    if isReference:
        intObj = dsFile[intRef]
        intVal = int(intObj[0])
    else: # Assuming value type
        intVal = int(intRef)
    return intVal

def yieldNextInt(intDataset, dsFile):
    for intData in intDataset:
        intVal = readInt(intData, dsFile)
        yield intVal 

def yieldNextBBox(bboxDataset, dsFile):
    for bboxArray in bboxDataset:
        bboxGroupRef = bboxArray[0]
        bboxGroup = dsFile[bboxGroupRef]
        labelDataset = bboxGroup["label"]
        leftDataset = bboxGroup["left"]
        topDataset = bboxGroup["top"]
        widthDataset = bboxGroup["width"]
        heightDataset = bboxGroup["height"]

        left = yieldNextInt(leftDataset, dsFile)
        top = yieldNextInt(topDataset, dsFile)
        width = yieldNextInt(widthDataset, dsFile)
        height = yieldNextInt(heightDataset, dsFile)

        bboxList = []

        for label in yieldNextInt(labelDataset, dsFile):
            bbox = BBox()
            bbox.label = label
            bbox.left = next(left)
            bbox.top = next(top)
            bbox.width = next(width)
            bbox.height = next(height)
            bboxList.append(bbox)

        yield bboxList

def yieldNextFileName(nameDataset, dsFile):
    for nameArray in nameDataset:
        nameRef = nameArray[0]
        name = readString(nameRef, dsFile)
        yield name

# dsFile = h5py.File('../data/gsvhn/train/digitStruct.mat', 'r')
def yieldNextDigitStruct(dsFileName):
    dsFile = h5py.File(dsFileName, 'r')
    dsGroup = dsFile["digitStruct"]
    nameDataset = dsGroup["name"]
    bboxDataset = dsGroup["bbox"]

    bboxListIter = yieldNextBBox(bboxDataset, dsFile)
    for name in yieldNextFileName(nameDataset, dsFile):
        bboxList = next(bboxListIter)
        obj = DigitStruct()
        obj.name = name
        obj.bboxList = bboxList
        yield obj

def getMain_DigitStruct_DataFrame(max_recognizable_digits=5, grayScale=False, transform_label=None):
    """Convert DigitStruct from primary dataset to Data Frame, which will have columns
    [image, N, y, bb]
    where,
    image:  image numpy array (gray-scale images but H x W x 1 dimensional)
    N:      Number of digits in this image
    y[i]:    ith digit name
    bb[i]:   ith bounding box (list of x,y,w,h)

    Note: for i>N-1, y[i]=-1 and bb[i]=[0,0,w,h] , where -1 is just an arbitrary choice signifying invalid digit
    Later y_i needs to be one-hot encoded with 11 places. 11th place = 1 for y_i=-1
    """
    dsFileName = 'data/gsvhn/train/digitStruct.mat'
    imageDir = 'data/gsvhn/train'

    columns = ['image', 'N', 'y', 'bb']
    df_main = pd.DataFrame(columns=columns)

    for dsObj in yieldNextDigitStruct(dsFileName):
        # testCounter += 1
        if grayScale:
            image = cv2.imread(os.path.join(imageDir, dsObj.name), 0) # read gray scale image
            image = image.reshape(image.shape[0], image.shape[1], 1) # convert image into single channel image. For tf channel is last
        else:
            image = cv2.imread(os.path.join(imageDir, dsObj.name), 1) # read as colored image

        N=0
        y= [-1] * max_recognizable_digits
        bb= [[0,0,image.shape[1], image.shape[0]]] * max_recognizable_digits
        for bbox in dsObj.bboxList:
            if N < max_recognizable_digits:
                y[N] = bbox.label if transform_label is None else transform_label(bbox.label)
                bb[N] = [bbox.left, bbox.top, bbox.width, bbox.height]
            N+=1
        y, bb = np.array(y), np.array(bb)

        df2 = pd.DataFrame(dict(zip(columns,[[item] for item in [image, N, y, bb]])))
        df_main = df_main.append(df2)

    return df_main

def getFalseImages_DataFrame(cropSize_dist, size=32, Ncrops_PerImage=3, total_crops=15000, max_recognizable_digits=5, grayScale=False, rotationRange=0):
    """ will cut out square crops from false_image data set, according to provided cropSize_dist
    outputs dataframe with:
    [image, N, y, bb]
    where,
    image:  image numpy array (gray-scale images but H x W x 1 dimensional)
    N:      Number of digits in this image
    y[i]:    ith digit name
    bb[i]:   ith bounding box (list of x,y,w,h)

    Note: for i>N-1, y[i]=-1 and bb[i]=[0,0,w,h] , where -1 is just an arbitrary choice signifying invalid digit
    Later y_i needs to be one-hot encoded with 11 places. 11th place = 1 for y_i=-1
    """
    dir_falseImages = 'data/false_images'
    selected_files = np.random.permutation([file for file in os.listdir('data/false_images/') if file.endswith('.jpg')]) # randomly select images
    selected_files = selected_files[:total_crops//Ncrops_PerImage] # select only files from which crops will be cut out

    p, bins = cropSize_dist # assumed to come from numpy histogram
    p=p/p.sum()
    p[-1] = 1. - np.sum(p[:-1])
    assert p[-1]>=-1e-6
    p[-1]=0.0 if p[-1]<0.0 else p[-1]
    new_points=[]
    for i in np.arange(bins.size-1):
        new_points.append(np.int(np.round(np.random.uniform(bins[i],bins[i+1]))))
    bins=np.array(new_points)


    columns = ['image', 'N', 'y', 'box_size', 'aspect_WbyH']
    df_false = pd.DataFrame(columns=columns)

    for filename in selected_files:
        if grayScale:
            image = cv2.imread(os.path.join(dir_falseImages, filename), 0) # read gray scale image
            image = image.reshape(image.shape[0], image.shape[1], 1) # convert image into single channel image. For tf channel is last
        else:
            image = cv2.imread(os.path.join(dir_falseImages, filename), 1) # read as colored image

        image = image[image.shape[0] // 2:, :]  # cut top half as it is highly blurred in false images

        if rotationRange > 0:
            rotationAngle = np.random.choice(np.arange(-rotationRange, rotationRange))  # random rotation
            cols, rows = image.shape[1], image.shape[0]
            M = cv2.getRotationMatrix2D((cols//2, rows//2), rotationAngle, 1)
            image = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)


        h, w, d = image.shape
        def getRandomLoc(cSize):
            t1=np.meshgrid(np.arange(w-cSize), np.arange(h-cSize), indexing='xy')
            t2=np.ravel_multi_index(t1, (w,h))
            t3=np.random.choice(t2.flatten())
            return np.unravel_index(t3, (h,w))

        n_crops=0
        while n_crops<Ncrops_PerImage:
            aCropSize = np.random.choice(bins, p=p)

            if aCropSize>=min(h,w)*0.9: continue
            y,x = getRandomLoc(aCropSize)
            aCrop = cv2.resize(image[y:y + aCropSize, x:x + aCropSize], (size, size))

            N = 0
            y = -1 * np.ones(max_recognizable_digits, dtype=np.int)  # there are no digits for these images

            df2 = pd.DataFrame(dict(zip(columns, [[item] for item in [aCrop, N, y, 32, 1.0]])))
            df_false = df_false.append(df2)

            n_crops +=1

    return df_false

def testMain():
    dsFileName = 'data/gsvhn/train/digitStruct.mat'
    testCounter = 0
    for dsObj in yieldNextDigitStruct(dsFileName):
        # testCounter += 1
        print(dsObj.name)
        for bbox in dsObj.bboxList:
            print("    {}:{},{},{},{}".format(
                bbox.label, bbox.left, bbox.top, bbox.width, bbox.height))
        if testCounter >= 5:
            break

if __name__ == "__main__":
    testMain()

