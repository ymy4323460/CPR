'''
Created on Apr 15, 2016
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
    (Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16)

@author: hexiangnan
'''
import os
import math
import heapq # for retrieval topK
import multiprocessing
import numpy as np
from sklearn.metrics import roc_auc_score, ndcg_score
import time

import torch
import torch.nn.functional as F
from torch import autograd, nn, optim
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear

import models as model
import config as config
import warnings
import utils as ut

args, _ = ut.get_config()
device = torch.device("cuda:{}".format(args.gpu_id) if(torch.cuda.is_available()) else "cpu")
device_cpu = torch.device("cpu")

class Evaluator:
    def __init__(self,
                 model,
                 K,
                 item_size,
                 train_file='./hw/train/bpr_data.csv',
                 test_file='./hw/dev/bpr_data.csv',
                 test_neg_file='./hw/dev/bpr_data_neg.csv',
                 neg_num=100,
                 flag=False):

        # Global variables that are shared across processes
        self.model = model
        # self.model.to(device_cpu)
        # self.model.to(device)
        self.train_file = train_file
        self.test_file = test_file
        self.K = K
        self.item_size = item_size
        self.neg_num = neg_num

        self.testRatings = self.load_rating_file_as_list(test_file)
#        if not os.path.exists(test_neg_file):
        self.test_reformatting(test_neg_file)
        self.testNegatives = self.load_negative_file(test_neg_file)

    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(",")
                user, item = int(float(arr[0])), int(float(arr[1]))
                ratingList.append([user, item])
                line = f.readline()
        return ratingList

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def load_postives(self):
        user2posItem = {}
        testUser = {}

        with open(self.train_file, "r") as f:
            line = f.readline()
            while line != None and line != "":
                user, item = int(float(line.split(",")[0])), int(float(line.split(",")[1]))
                if user in user2posItem.keys():
                    user2posItem[user].append(item)
                else:
                    user2posItem[user] = [item]
                line = f.readline()

        with open(self.test_file, "r") as f:
            line = f.readline()
            while line != None and line != "":
                user, item = int(float(line.split(",")[0])), int(float(line.split(",")[1]))
                if user in user2posItem.keys():
                    user2posItem[user].append(item)
                else:
                    user2posItem[user] = [item]
                if user not in testUser.keys():
                    testUser[user] = 1
                line = f.readline()

        return user2posItem, testUser

    def test_reformatting(self, out_file):
        np.random.seed(9021)

        user2posItem, testUser = self.load_postives()
        user2negItem = {}

        for user in testUser.keys():
            negs = []
            if user not in user2negItem.keys():
                for _ in range(self.neg_num):
                    neg = np.random.randint(0, self.item_size, size=1)[0]
                    while neg in negs or neg in user2posItem[user]:
                        neg = np.random.randint(0, self.item_size, size=1)[0]
                    negs.append(neg)
                user2negItem[user] = negs

        fid = open(out_file, 'w')
        for ui in self.testRatings:
            info = '({},{})'.format(ui[0], ui[1])
            for neg in user2negItem[ui[0]]:
                info = info + '\t' + '{}'.format(neg)
            fid.write(info+'\n')
        fid.close()

    def evaluate_model(self, num_thread):
        """
        Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
        Return: score of each test rating.
        """
        # begin_time = time.time()
        hits, ndcgs, aucs = [],[],[]
        if(num_thread > 1): # Multi-thread
            pool = multiprocessing.Pool(processes=num_thread)
            res = pool.map(self.eval_one_rating, range(len(self.testRatings)))
            # res = pool.map(self.eval_one_rating, range(1))
            pool.close()
            pool.join()
            hits = [r[0] for r in res]
            ndcgs = [r[1] for r in res]
            aucs = [r[2] for r in res]
            hr, ndcg, auc = np.array(hits).mean(), np.array(ndcgs).mean(), np.array(aucs).mean()
            # print("Evaluate time: %f sec" % (time.time() - begin_time))
            return hr, ndcg, auc
        # Single thread
        for idx in range(len(self.testRatings)):
            (hr, ndcg, auc) = self.eval_one_rating(idx)
            hits.append(hr)
            ndcgs.append(ndcg)
            aucs.append(auc)

        hr, ndcg, auc = np.array(hits).mean(), np.array(ndcgs).mean(), np.array(aucs).mean()
        # print("Evaluate time: %f sec" % (time.time() - begin_time))
        return hr, ndcg, auc

    def eval_one_rating(self, idx):
        rating = self.testRatings[idx]
        items = self.testNegatives[idx]
        u = rating[0]
        gtItem = rating[1]
        items.append(gtItem)
        # Get prediction scores
        map_item_score = {}
        users = np.full(len(items), u, dtype='int32')
        predictions = self.model.predict(users, np.array(items), mode='test')

        labels = np.array([0]*self.neg_num+[1], dtype='float32')

        # print('***'*10)
        # print('predictions', predictions.shape)
        # print('labels', labels.shape)
        # print('***'*10)
        # print('user', users)
        # print('item', items)
        # print('prediction', predictions.shape)
        # print('prediction', predictions)
        # print('labels', labels.shape)
        # print('labels', labels)
        auc = self.getAUC(labels, predictions)

        for i in range(len(items)):
            item = items[i]
            map_item_score[item] = predictions[i]
        items.pop()

        # Evaluate top rank list
        ranklist = heapq.nlargest(self.K, map_item_score, key=map_item_score.get)
        hr = self.getHitRatio(ranklist, gtItem)
        ndcg = self.getNDCG(ranklist, gtItem)
        return (hr, ndcg, auc)

    def getHitRatio(self, ranklist, gtItem):
        for item in ranklist:
            if item == gtItem:
                return 1
        return 0

    def getNDCG(self, ranklist, gtItem):
        for i in range(len(ranklist)):
            item = ranklist[i]
            if item == gtItem:
                return math.log(2) / math.log(i+2)
        return 0

    def getAUC(self, y_true, y_pred):
        return roc_auc_score(y_true, y_pred)

def eval_auc_ndcg_hr(y_score, y_label):

    # print('***'*10)
    # print(y_score.shape, y_label.shape)
    # print('***'*10)
    indices = (np.sum(y_label, axis=1)>0)
    y_label = y_label[indices]
    y_score = y_score[indices]

    # print('***'*10)
    # print(y_score.shape, y_label.shape)
    # print('***'*10)

    indices = (np.sum(y_label, axis=1)<5)
    y_label = y_label[indices]
    y_score = y_score[indices]

    # print('***'*10)
    # print(y_score.shape, y_label.shape)
    # print('***'*10)

    aucs = []
    hrs = []

    for i in range(y_score.shape[0]):
        aucs.append(roc_auc_score(y_label[i, :], y_score[i, :]))
    auc = sum(aucs)/len(aucs)
    ndcg = ndcg_score(y_label, y_score)

    return auc, ndcg

if __name__=='__main__':
    args, _ = config.get_config()
    # bpr_model = model.BPR(args)
    bpr_model = model.NeuBPR(args)
    evaluator = Evaluator(bpr_model, 10, args.item_size,
                          train_file='./real/mind/train/bpr_data.csv',
                          test_file='./real/mind/dev/bpr_data.csv',
                          test_neg_file='./real/mind/dev/bpr_data_neg.csv',
                          neg_num=100,
                          flag=False)

    begin_time = time.time()
    hr, ndcg, auc = evaluator.evaluate_model(1)
    print("HR:{}, NDCG:{}, Evaluate time: {:.0f} sec".format(hr, ndcg, time.time() - begin_time))


    # args, _ = get_config()
    # bpr_model = model.BPR(args)
    # evaluator = Evaluator(bpr_model, 10, args.item_size,
    #                       train_file='./hw/train/bpr_data.csv',
    #                       test_file='./hw/dev/bpr_data_demo.csv',
    #                       test_neg_file='./hw/dev/bpr_data_neg_demo.csv',
    #                       neg_num=100,
    #                       flag=False)

    # hr, ndcg, auc = evaluator.evaluate_model(1)
    # print('HR:{}, NDCG:{}'.format(hr, ndcg))


    # args, _ = get_config()
    # r_model = model.RecommendModel(args)
    # evaluator = Evaluator(r_model, 10, args.item_size,
    #                       train_file='./hw/train/data.csv',
    #                       test_file='./hw/dev/data.csv',
    #                       test_neg_file='./hw/dev/data_neg.csv',
    #                       neg_num=100,
    #                       flag=True)

    # begin_time = time.time()
    # hr, ndcg, auc = evaluator.evaluate_model(20)
    # print("HR:{}, NDCG:{}, AUC:{}, Evaluate time: {:.0f} sec".format(hr, ndcg, auc, time.time() - begin_time))
