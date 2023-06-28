import os
import math
import time
import random
import torch
import argparse
import numpy as np
import utils as ut
import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
from torch.utils import data
import matplotlib.pyplot as plt
import torch.utils.data as Data
from torch.autograd import Variable
import dataloader
import models as model
import learner
import evaluator
import dataset
# from codebase.debias.models import get_config as get_cc_config
import logging
import warnings
warnings.filterwarnings('ignore')


def train_rModel(args):

    #-----------------------------------------------------------
    # log
    logger = logging.getLogger()
    fh = logging.FileHandler(args.log_file)
    if args.log_level:
        logger.setLevel(logging.getLevelName(args.log_level))
    else:
        logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - line:%(lineno)d]: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    #-----------------------------------------------------------

    device = torch.device("cuda:{}".format(args.gpu_id) if(torch.cuda.is_available()) else "cpu")
    device_cpu = torch.device("cpu")
    logger.info(torch.cuda.is_available())
    logger.info(device)

    logger.info('\n\n\n\n\n')
    logger.info('***'*20)
    logger.info('##### rModel Pretraining #####')
    logger.info(args)
    logger.info('***'*20)

    # workstation_path = './'
    # args.model_dir = os.path.join(workstation_path, args.model_dir)
    # dataset_dir = os.path.join(workstation_path, args.dataset)

    dataset_dir = os.path.join(args.dataset)

    mode = 'dynamic_negative_sampling'
    # mode = 'original'
    logger.info('negative-sampling-mode: {}'.format(mode))

    if args.learning_mode == 'train':
        tr_rs_data = os.path.join(dataset_dir, 'train', 'data.csv')
        ts_rs_data = os.path.join(dataset_dir, 'dev', 'data.csv')
        ts_rs_data_neg = os.path.join(dataset_dir, 'dev', 'data_neg.csv')

        tr_s_list = os.path.join(dataset_dir, 'train', 'list_impression.csv')
        ts_s_list = os.path.join(dataset_dir, 'dev', 'list_impression.csv')

        if args.pair_wise_selection:
            if mode=='original':
                train_r_dataloader = dataloader.dataload(tr_rs_data, args.batch_size) # a size is 1
                test_r_dataloader = dataloader.dataload(ts_rs_data, args.batch_size)
            else:
                train_r_dataset = dataset.Dataset(tr_rs_data, args.item_size, 1)

            train_s_dataloader = dataloader.dataload(tr_rs_data, args.batch_size)
            test_s_dataloader = dataloader.dataload(ts_rs_data, args.batch_size)
        else:
            if mode=='original':
                train_r_dataloader = dataloader.dataload(tr_rs_data, args.batch_size) # a size is 1
                test_r_dataloader = dataloader.dataload(ts_rs_data, args.batch_size)
            else:
                train_r_dataset = dataset.Dataset(tr_rs_data, args.item_size, 1)

            train_s_dataloader = dataloader.list_dataload(tr_s_list, args.batch_size)
            # test_s_dataloader = dataloader.list_dataload(os.path.join(dataset_dir, 'dev', 'list_impression.csv'), args.batch_size)
            test_s_x, test_s_a, test_s_y = dataset.load_impression_list_file(ts_s_list)

    # select_dataloader = ut.SelectedDataLoader(dataset_dir, args.batch_size) # a size is n

    # learning_rates = [1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5]
    # learning_rates.reverse()
    learning_rates = [2e-05]


    for lr in learning_rates:
        logger.info('###'*10)
        args.lr_r = lr
        logger.info('Learning rate of recommendModel'.format(lr))
        logger.info('###'*10)

        if args.learning_mode == 'train':
            recommed_model = model.RecommendModel(args).to(device)
            select_model = model.SelectModel(args).to(device)

            r_evaluator = evaluator.Evaluator(recommed_model, 10, args.item_size,
                                              train_file=tr_rs_data,
                                              test_file=ts_rs_data,
                                              test_neg_file=ts_rs_data_neg,
                                              neg_num=100,
                                              flag=False)
            #--------------------------------------------------------------------------

            # RecommendModel
            # train_PQ_XY
            # epsilon lookup table should be fixed as item_size*[0,1], which means the prior of epsilon is N(0,I)
            recommed_model.args.learn_epsilon = False
            for i in recommed_model.parameters():
                i.requires_grad = True
            for i in recommed_model.er_embedding_lookup.parameters():
                i.requires_grad = False
            recommend_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, recommed_model.parameters()), lr=args.lr_r, betas=(0.9, 0.999))
            #--------------------------------------------------------------------------

            best_hr = 0
            best_ndcg = 0
            best_auc = 0
            best_epoch = 0
            count = 0
            for epoch in range(args.train_epoch_max_r):
                recommed_model.to(device)
                recommed_model.train()
                recommend_loss = 0
                total_recommend_loss = 0
                begin_time = time.time()
                if mode=='original':
                    for x, a, r, _ in train_r_dataloader:
                        # train recommended loss
                        recommend_optimizer.zero_grad()
                        x, a, r = x.to(device), a.to(device), r.to(device)
                        recommend_loss = recommed_model.loss(x, a, r)
                        recommend_loss.backward()
                        recommend_optimizer.step()
                        total_recommend_loss += recommend_loss.item()
                        mr = len(train_r_dataloader)
                    total_recommend_loss /= mr
                else:
                    total_recommend_loss = recommed_model.train_one_epoch(train_r_dataset, recommend_optimizer, args.batch_size)
                tr_time = time.time() - begin_time

                if epoch % 1 == 0:
                    count += 1
                    begin_time = time.time()
                    # bpr_model.to(device_cpu)
                    # bpr_model.eval()
                    r_evaluator.model = recommed_model
                    r_evaluator.model.to(device_cpu)
                    # r_evaluator.model.to(device)
                    hr, ndcg, auc = r_evaluator.evaluate_model(20)
                    r_evaluator.model.to(device)
                    logger.info("epoch:{}, total_recommend_loss: {:.3f}, test_hr:{:.4f}, test_ndcg:{:.4f}, test_auc:{:.4f}, train_time:{:.0f}, test_time:{:.0f}"
                        .format(epoch+1, total_recommend_loss, hr, ndcg, auc, tr_time, time.time()-begin_time))

                    if hr+ndcg+auc > best_hr+best_ndcg+best_auc:
                        count = 0
                        best_hr = hr
                        best_auc = auc
                        best_ndcg = ndcg
                        best_epoch = epoch
                        ut.save_model_by_name(model_dir=args.model_dir, model=recommed_model, global_step=str(args.magic_num)+'-'+str(args.user_emb_dim_r))

                # early stop
                if count>=3:
                    break

                # # if epoch % args.iter_save == 0:
                # if epoch==args.train_epoch_max_r-1:
                #     ut.save_model_by_name(model_dir=args.model_dir, model=recommed_model, global_step=epoch)
            #--------------------------------------------------------------------------

            #----------------------------------------------------------------------------
            # train_epsilon: RecommendModel
            # ut.load_model_by_name(args.model_dir, recommed_model, args.train_epoch_max_r-1)
            ut.load_model_by_name(args.model_dir, recommed_model, str(args.magic_num)+'-'+str(args.user_emb_dim_r))

            recommed_model.args.learn_epsilon = True
            # only mu sigma in epsilon's posterior should be learned in this turn
            for i in recommed_model.parameters():
                i.requires_grad = False
            for i in recommed_model.er_embedding_lookup.parameters():
                i.requires_grad = True
            recommend_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, recommed_model.parameters()), lr=1e-3, betas=(0.9, 0.999))
            # turn to epsilon prior
            #recommed_model.set_epsilon()
            for epoch in range(args.train_epsilon_epoch_max_r):
                recommed_model.to(device)
                recommed_model.train()
                recommend_loss = 0
                total_recommend_loss = 0

                begin_time = time.time()
                if mode=='original':
                    for x, a, r, _ in train_r_dataloader:
                        # train recommended loss
                        recommend_optimizer.zero_grad()
                        x, a, r = x.to(device), a.to(device), r.to(device)
                        recommend_loss = -recommed_model.epsilon_loss(x, a, r)
                        recommend_loss.backward()
                        recommend_optimizer.step()
                        total_recommend_loss += recommend_loss.item()
                        mr = len(train_r_dataloader)
                    total_recommend_loss /= mr
                else:
                    total_recommend_loss = recommed_model.train_one_epoch(train_r_dataset, recommend_optimizer, args.batch_size)

                tr_time = time.time() - begin_time

                if epoch % 1 == 0:
                    begin_time = time.time()
                    # bpr_model.to(device_cpu)
                    # bpr_model.eval()
                    r_evaluator.model = recommed_model
                    r_evaluator.model.to(device_cpu)
                    # r_evaluator.model.to(device)
                    hr, ndcg, auc = r_evaluator.evaluate_model(20)
                    r_evaluator.model.to(device)
                    logger.info("epoch:{}, total_recommend_loss: {:.3f}, test_hr:{:.4f}, test_ndcg:{:.4f}, test_auc:{:.4f}, train_time:{:.0f}, test_time:{:.0f}"
                        .format(epoch+1, total_recommend_loss, hr, ndcg, auc, tr_time, time.time()-begin_time))

                # if epoch % args.iter_save == 0:
                if epoch==args.train_epsilon_epoch_max_r-1:
                    ut.save_model_by_name(model_dir=args.model_dir, model=recommed_model, global_step=str(args.magic_num)+'-'+str(args.user_emb_dim_r))
            #----------------------------------------------------------------------------

        logger.info('\n\n\n')
