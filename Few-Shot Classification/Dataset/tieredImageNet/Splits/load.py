# Author : Bryce Xu
# Time : 2020/2/1
# Function: 

'''
import pickle as pkl
f = open('/mnt/datadev_2/std/TieredImageNet/tiered-imagenet/train_images_png.pkl', 'rb')
data = pkl.load(f)
print(len(data[2]))
'''

'''
import numpy as np

imagepath = '/mnt/datadev_2/std/TieredImageNet/images/train.npz'

with np.load(imagepath, mmap_mode="r", encoding='latin1') as data:
    images = data["images"]
    print(len(images)) # 448695
'''

import pickle as pkl

labelpath = '/mnt/datadev_2/std/TieredImageNet/images/train_labels.pkl'

with open(labelpath, "rb") as f:
    data = pkl.load(f)
    label_specific = data["label_specific"]
    label_general = data["label_general"]
    label_specific_str = data["label_specific_str"]
    label_general_str = data["label_general_str"]
    print(len(label_specific_str))
    print(len(label_specific))
















