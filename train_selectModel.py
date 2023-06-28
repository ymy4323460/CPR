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


def train_selectModel(args):

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
    logger.info('##### selectModel Pretraining #####')
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

    # learning_rates = [1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 1.5e-5, 9e-6, 8e-6, 5e-6, 3e-6, 1e-6]  # best: 1e-5
    # learning_rates = [1.5e-5, 9e-6, 8e-6, 5e-6, 3e-6, 1e-6]  # best: 9e-06
    # learning_rates = [args.lr_s]
    # embedding_sizes = [8, 16, 32, 64, 128, 256]

    learning_rates = [args.lr_s]
    embedding_sizes = [args.user_emb_dim_s]

    for lr in learning_rates:
        logger.info('###'*10)
        args.lr_s = lr
        logger.info('Learning rate of selectModel {}'.format(lr))
        logger.info('###'*10)

        for dim in embedding_sizes:
            logger.info('###'*10)
            args.user_emb_dim_s = dim
            args.item_emb_dim_s = dim
            logger.info('Embedding size of selectModel {}'.format(dim))
            logger.info('###'*10)

            if args.learning_mode == 'train':
                select_model = model.SelectModel(args).to(device)

                # selection model
                select_model.args.learn_epsilon = False
                for i in select_model.parameters():
                    i.requires_grad = True
                for i in select_model.es_embedding_lookup.parameters():
                    i.requires_grad = False
                select_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, select_model.parameters()), lr=args.lr_s, betas=(0.9, 0.999))

                best_acc = 0
                best_auc = 0
                best_ndcg = 0
                best_epoch = 0
                count = 0
                for epoch in range(args.train_epoch_max_s):

                    select_model.train()
                    select_loss = 0
                    total_select_loss = 0

                    begin_time = time.time()
                    if args.pair_wise_selection:
                        for x, a, _, y in train_s_dataloader:
                            select_optimizer.zero_grad()
                            x, a, y = x.to(device), a.to(device), y.to(device)
                            select_loss = select_model.loss(x, a, y)
                            select_loss.backward()
                            select_optimizer.step()
                            total_select_loss += select_loss.item()
                            ms = len(train_s_dataloader)
                    else:
                        for x, a, y in train_s_dataloader:
                            select_optimizer.zero_grad()
                            x, a, y = x.to(device), a.to(device), y.to(device)
                            # select_loss = select_model.loss(x, a, y)
                            select_loss = select_model.bce_loss(x, a, y)
                            select_loss.backward()
                            select_optimizer.step()
                            total_select_loss += select_loss.item()
                            ms = len(train_s_dataloader)
                    total_select_loss /= ms
                    tr_time = time.time() - begin_time

                    if (epoch+1) % 1 == 0:
                        count += 1
                        acc, y_score, y_label = select_model.evaluate(test_s_x, test_s_a, test_s_y)
                        auc, ndcg = evaluator.eval_auc_ndcg_hr(y_score, y_label)
                        logger.info("Epoch:{}, Train total_select_loss:{:.3f}, Test acc:{:.3f}, AUC:{:.3f}, nDCG:{:.3f}".format(epoch, total_select_loss, acc, auc, ndcg))

                        if acc+auc+ndcg > best_acc+best_auc+best_ndcg:
                            count = 0
                            best_acc = acc
                            best_auc = auc
                            best_ndcg = ndcg
                            best_epoch = epoch
                            ut.save_model_by_name(model_dir=args.model_dir, model=select_model, global_step=str(args.magic_num)+'-'+str(args.user_emb_dim_s))

                        # logger.info("Train total_select_loss:{}, Test acc:{:.3f}".format(total_select_loss, acc))

                    # early stop
                    if count>=3:
                        break

                    # # if epoch % args.iter_save == 0:
                    # if epoch==args.train_epoch_max_s-1:
                    #     ut.save_model_by_name(model_dir=args.model_dir, model=select_model, global_step=epoch)
                #--------------------------------------------------------------------------

                #----------------------------------------------------------------------------
                # train_epsilon: SelectModel
                # ut.load_model_by_name(args.model_dir, select_model, args.train_epoch_max_s-1)
                ut.load_model_by_name(args.model_dir, select_model, str(args.magic_num)+'-'+str(args.user_emb_dim_s))

                select_model.args.learn_epsilon = True
                # only mu sigma in epsilon's posterior should be learned in this turn
                for i in select_model.parameters():
                    i.requires_grad=False
                for i in select_model.es_embedding_lookup.parameters():
                    i.requires_grad=True

                logger.info('***'*20)
                logger.info('##### selectModel train_epsilon #####')
                logger.info(args)
                logger.info('***'*20)

                # for name, pram in select_model.named_parameters():
                #     logger.info('requires_grad:{}:{}'.format(name, pram.requires_grad))
                select_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, select_model.parameters()),  lr=args.lr_s, betas=(0.9, 0.999))
                # turn to epsilon prior
                #select_model.set_epsilon()
                for epoch in range(args.train_epsilon_epoch_max_s):
                    select_model.train()
                    select_loss = 0
                    total_select_loss = 0

                    if args.pair_wise_selection:
                        for x, a, _, y in train_s_dataloader:
                            select_optimizer.zero_grad()
                            x, a, y = x.to(device), a.to(device), y.to(device)
                            select_loss = -select_model.epsilon_loss(x, a, y)
                            select_loss.backward()
                            select_optimizer.step()
                            total_select_loss += select_loss.item()
                            ms = len(train_s_dataloader)
                    else:
                        for x, a, y in train_s_dataloader:
                            select_optimizer.zero_grad()
                            x, a, y = x.to(device), a.to(device), y.to(device)
                            select_loss = -select_model.epsilon_loss(x, a, y)
                            select_loss.backward()
                            select_optimizer.step()
                            total_select_loss += select_loss.item()
                            ms = len(train_s_dataloader)
                    if (epoch+1) % 1 == 0:
=                        acc, y_score, y_label = select_model.evaluate(test_s_x, test_s_a, test_s_y)
                        auc, ndcg = evaluator.eval_auc_ndcg_hr(y_score, y_label)
                        logger.info("Train epsilon turn:{} total_select_loss:{:.3f}, Test acc:{:.3f}, AUC:{:.3f}, nDCG:{:.3f}".format(epoch, total_select_loss/ms, acc, auc, ndcg))
                    # if epoch % args.iter_save == 0:
                    if epoch==args.train_epsilon_epoch_max_s-1:
                        ut.save_model_by_name(model_dir=args.model_dir, model=select_model, global_step=str(args.magic_num)+'-'+str(args.user_emb_dim_s))
                #----------------------------------------------------------------------------
            logger.info('\n\n\n')
