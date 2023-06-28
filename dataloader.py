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
import utils as ut

args, _ = ut.get_config()
device = torch.device("cuda:{}".format(args.gpu_id) if (torch.cuda.is_available()) else "cpu")


def fileread(data_dir):
    print(data_dir)
    with open(data_dir, 'rb') as f:
        reader = f.readlines()
        data = []
        for line in reader:
            line = [int(float(i)) for i in line.decode().strip().split(',')]
            data.append(line)
    # print(len(data))
    data = np.array(data, dtype=np.int64)
    return data  # .reshape([len(data),4,1])


class DataLoad(data.Dataset):
    def __init__(self, root):
        # self.user_item = fileread(os.path.join(root,'data'))
        self.user_item = fileread(root)
        self.transforms = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        data = torch.from_numpy(self.user_item[idx])
        # print(data[0])
        if data.shape[0] == 4:  # learning causal model
            return data[0], data[1], data[2], data[3]
        if data.shape[0] == 3:  # learning causal model
            return data[0], data[1], data[2]

    def __len__(self):
        return len(self.user_item)


class ListDataLoad(data.Dataset):
    def __init__(self, root):
        # self.user_item = fileread(os.path.join(root,'data'))
        self.user_item = fileread(root)
        self.transforms = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        data = torch.from_numpy(self.user_item[idx])
        # print(data[0])
        return data[0], data[1:6], data[6:11]

    def __len__(self):
        return len(self.user_item)


def dataload(dataset_dir, batch_size, shuffle=False, num_workers=8, pin_memory=True):
    dataset = DataLoad(dataset_dir)
    dataset = Data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    return dataset


def list_dataload(dataset_dir, batch_size, shuffle=False, num_workers=8, pin_memory=True):
    dataset = ListDataLoad(dataset_dir)
    dataset = Data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    return dataset
