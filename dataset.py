import os
import math
import random
import torch.nn as nn
from torch.utils import data
import argparse
import numpy as np
from torchvision import transforms
from PIL import Image
import torch
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F

import models as model
# from config import get_config
import utils as ut
import utils as ut

args, _ = ut.get_config()
device = torch.device("cuda:{}".format(args.gpu_id) if (torch.cuda.is_available()) else "cpu")


class Dataset:
    def __init__(self, data_dir, item_size, num_negatives):
        self.item_size = item_size
        self.num_negatives = num_negatives
        self.data, self.user2item = self.fileread(data_dir)

    def fileread(self, data_dir):
        print(data_dir)
        user2item = {}
        with open(data_dir, 'rb') as f:
            reader = f.readlines()
            data = []
            for line in reader:
                line = line.decode().strip().split(',')
                user, item = int(float(line[0])), int(float(line[1]))
                if user not in user2item.keys():
                    user2item[user] = [item]
                    data.append([user, item])
                else:
                    if item not in user2item[user]:
                        user2item[user].append(item)
                        data.append([user, item])
        # print(len(data))
        data = np.array(data, dtype=float)
        return data, user2item

    def generate_triple_data(self):
        users = []
        positives = []
        negatives = []
        for i in range(self.data.shape[0]):
            # for i in range(100):
            user = self.data[i, 0]
            item = self.data[i, 1]
            users += [user] * self.num_negatives
            positives += [item] * self.num_negatives

            for i in range(self.num_negatives):
                neg = np.random.randint(0, self.item_size, size=1)[0]
                while neg in self.user2item[user]:
                    neg = np.random.randint(0, self.item_size, size=1)[0]
                negatives.append(neg)

        return np.array(users, dtype=np.int64), np.array(positives, dtype=np.int64), np.array(negatives, dtype=np.int64)

    def sample_negatives(self, users, positives, num_negatives=2):
        users_new = []
        pos_new = []
        neg_new = []

        for i in range(users.shape[0]):
            user = users[i, 0]
            item = positives[i, 0]
            users_new += [user] * num_negatives
            pos_new += [item] * num_negatives

            for i in range(num_negatives):
                neg = np.random.randint(0, self.item_size, size=1)[0]
                while neg in self.user2item[user]:
                    neg = np.random.randint(0, self.item_size, size=1)[0]
                neg_new.append(neg)

        users_new = np.array(users_new, dtype=np.int64).reshape((-1, 1))
        pos_new = np.array(pos_new, dtype=np.int64).reshape((-1, 1))
        neg_new = np.array(neg_new, dtype=np.int64).reshape((-1, 1))

        return users_new, pos_new, neg_new

    def generate_pairwise_data(self):
        users = []
        items = []
        labels = []
        for i in range(self.data.shape[0]):
            # for i in range(100):
            user = self.data[i, 0]
            item = self.data[i, 1]
            users += [user] * (self.num_negatives + 1)
            items += [item]
            labels += [1]

            for i in range(self.num_negatives):
                neg = np.random.randint(0, self.item_size, size=1)[0]
                while neg in self.user2item[user]:
                    neg = np.random.randint(0, self.item_size, size=1)[0]
                items.append(neg)
                labels.append(0)

        return np.array(users, dtype=np.int64), np.array(items, dtype=np.int64), np.array(labels, dtype=np.int64)

    def __str__(self):
        # return string representation of 'Dataset' class
        # print(Dataset) or str(Dataset)
        ret = '======== [Dataset] ========\n'
        # ret += 'Train file: %s\n' % self.train_file
        # ret += 'Test file : %s\n' % self.test_file
        ret += 'Number of Users : %d\n' % len(self.user2item)
        ret += 'Number of items : %d\n' % self.item_size
        ret += '\n'
        return ret


# ---------------------------------------------------
# SelectModel
def load_impression_list_file(filename):
    users = []
    items = []
    labels = []
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = [int(float(i)) for i in line.split(",")]
            users.append(arr[0])
            items.append(arr[1:6])
            labels.append(arr[6:])
            line = f.readline()
    return np.array(users, dtype=np.int32), np.array(items, dtype=np.int32), np.array(labels, dtype=np.int32)


# ---------------------------------------------------


if __name__ == '__main__':
    args, _ = get_config()
    select_model = model.SelectModel(args).to(device)
    x, a, y = load_impression_list_file('./hw/dev/list_impression.csv')
    print('------------------------------------')
    print('x', x[:10])
    print('a', a[:10])
    print('y', y[:10])
    print('------------------------------------')
    acc = select_model.evaluate(x, a, y)
    print('Scatch Accuracy:', acc)

    ut.load_model_by_name(args.model_dir, select_model, args.train_epoch_max - 1)
    acc = select_model.evaluate(x, a, y)
    print('Pretrained Accuracy:', acc)
