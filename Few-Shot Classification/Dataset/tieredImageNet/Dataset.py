# Author : Bryce Xu
# Time : 2020/2/1
# Function: 

import torch.utils.data as data
import os
import numpy as np
import pickle as pkl
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import cv2

def decompress(path, output):
  with open(output, 'rb') as f:
    array = pkl.load(f)
  images = np.zeros([len(array), 84, 84, 3], dtype=np.uint8)
  for ii, item in tqdm(enumerate(array), desc='decompress'):
    im = cv2.imdecode(item, 1)
    images[ii] = im
  np.savez(path, images=images)

class TieredImageNet(data.Dataset):

    def __init__(self, mode, root):
        super(TieredImageNet, self).__init__()
        image_path = os.path.join(root, mode + '.npz')
        label_path = os.path.join(root, mode + '_labels.pkl')

        with np.load(image_path, mmap_mode="r", encoding='latin1') as data:
            self.images = data["images"]

        with open(label_path, "rb") as f:
            data = pkl.load(f)
            self.label_specific = data["label_specific"]
            self.label_specific_str = data["label_specific_str"]

        self.transform = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = self.transform(Image.fromarray(self.images[item]).convert('RGB'))
        label = self.label_specific
        return image, label

