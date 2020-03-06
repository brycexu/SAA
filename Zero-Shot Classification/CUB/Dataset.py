# Author : Bryce Xu
# Time : 2020/2/18
# Function: 

from PIL import Image
from torch.utils import data
import os
from WordEmbedding import *

class CUBDataset(data.Dataset):
    def __init__(self, parser, examples, labels, attributes, names, transform):
        self.labels = labels
        self.examples = examples
        self.transform = transform
        self.image_dir = parser.image_dir
        self.data = []
        self.label = []
        self.attribute = []
        self.label_name = []
        self.word_model = load_model(parser.word_dir)
        for i in range(len(examples)):
            id = self.examples[i]
            path = os.path.join(self.image_dir, id)
            self.data.append(path)
            self.label.append(self.labels[id])
            self.attribute.append(attributes[self.labels[id]])
            self.label_name.append(names[self.labels[id]])
        self.embedding = load_embeddings(self.label_name, self.word_model)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        path, label, attribute, embedding = self.data[item], self.label[item], self.attribute[item], self.embedding[item]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label, attribute, embedding