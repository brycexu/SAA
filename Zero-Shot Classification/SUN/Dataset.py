# Author : Bryce Xu
# Time : 2020/2/29
# Function:

import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
from WordEmbedding import load_model, load_embeddings
import numpy as np
import torch

class SUNDataset(data.Dataset):

    def __init__(self, mode, data_root, attribute_root, wmodel_root):
        super(SUNDataset, self).__init__()
        allClasses_path = os.path.join(data_root, 'classes.txt')
        allClasses_lines = [x.strip('\n') for x in open(allClasses_path, 'r').readlines()]
        self.allClasses = {}
        for l in allClasses_lines:
            index, name = l.split(' ')
            self.allClasses[name] = int(index)-1
        class_path = os.path.join(data_root, mode + 'classes.txt')
        class_lines = [x.strip() for x in open(class_path, 'r').readlines()]
        attribute_lines = np.loadtxt(attribute_root)
        self.data = []
        self.label = []
        self.label_name = []
        self.attribute = []
        labelidx = -1
        self.word_model = load_model(wmodel_root)
        for class_name in class_lines:
            currentPath = os.path.join(data_root, 'images', class_name)
            labelidx += 1
            for _, _, files in os.walk(currentPath):
                for file in files:
                    self.data.append(os.path.join(currentPath, file))
                    self.label.append(labelidx)
                    self.label_name.append(class_name)
                    self.attribute.append(np.float32(attribute_lines[self.allClasses[class_name]]))
        self.wordEmbeddings = load_embeddings(self.label_name, self.word_model)
        self.attribute = np.array(self.attribute, dtype='float32')
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        path, label = self.data[item], self.label[item]
        image = self.transform(Image.open(path).convert('RGB'))
        attribute = self.attribute[item]
        wordEmbedding = self.wordEmbeddings[item]
        return image, label, attribute, wordEmbedding

