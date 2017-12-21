"""Get 10000 street view images.
then we will randomly find 5 small patches in each image as all false digit dataset to augment training"""

import urllib.request
import re
import os
import random

dataDir='../data/false_images'
webFolder='http://www.cs.ucf.edu/~aroshan/index_files/Dataset_PitOrlManh/images/'
urlpath =urllib.request.urlopen(webFolder)
string = urlpath.read().decode('utf-8')

pattern = re.compile(r"\w*_[1-4].jpg")


filelist = pattern.findall(string)
filelist = random.sample(filelist, 10000)
print(filelist)
print('total images considered...{}'.format(len(filelist)))

for filename in filelist:
    if filename not in set(os.listdir(dataDir)):
        print(filename)
        urllib.request.urlretrieve(webFolder + filename,
                                   os.path.join(dataDir, filename))