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


def train_intervention(args):
    # -----------------------------------------------------------
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
    # -----------------------------------------------------------

    device = torch.device("cuda:{}".format(args.gpu_id) if (torch.cuda.is_available()) else "cpu")
    device_cpu = torch.device("cpu")
    logger.info(torch.cuda.is_available())
    logger.info(device)

    logger.info('\n\n\n\n\n')
    logger.info('***' * 20)
    logger.info('##### Intervention {} #####'.format(args.intervent_type))
    logger.info(args)
    logger.info('***' * 20)

    # workstation_path = './'
    # args.model_dir = os.path.join(workstation_path, args.model_dir)
    # dataset_dir = os.path.join(workstation_path, args.dataset)

    dataset_dir = os.path.join(args.dataset)

    # mode = 'dynamic_negative_sampling'
    mode = 'original'
    logger.info('negative-sampling-mode: {}'.format(mode))

    if args.learning_mode == 'intervention':
        # tr_bpr_data = os.path.join(dataset_dir, 'train', 'bpr_data.csv')
        # ts_bpr_data = os.path.join(dataset_dir, 'dev', 'bpr_data.csv')
        # ts_bpr_data_neg = os.path.join(dataset_dir, 'dev', 'bpr_data_neg.csv')
        tr_bpr_data = os.path.join(dataset_dir, 'train', 'bpr_data.csv')
        ts_bpr_data = os.path.join(dataset_dir, 'dev', 'bpr_data.csv')
        ts_bpr_data_neg = os.path.join(dataset_dir, 'dev', 'bpr_data_neg.csv')
        pretrain_dataset = dataset.Dataset(tr_bpr_data, args.item_size, args.negative_num)

        # for LightGCN
        if args.model_name == 'LightGCN':
            tr_matrix_file = os.path.join(dataset_dir, 'train', 'train_matrix.npz')
            logger.info('Load train matrix...')
            train_matrix = ut.load_train_matrix(tr_bpr_data, tr_matrix_file, args.user_size, args.item_size)
            logger.info('Done loading.')

    # select_dataloader = ut.SelectedDataLoader(dataset_dir, args.batch_size) # a size is n

    if args.learning_mode == 'intervention':

        # logging = logger('', mf.name, args.dataset)

        # load pretrained model
        if args.model_name == 'BPR':
            bpr_pretrained = model.BPR(args).to(device)
        elif args.model_name in ['NeuBPR', 'gmfBPR', 'mlpBPR', 'bprBPR']:
            bpr_pretrained = model.NeuBPR(args).to(device)
        elif args.model_name == 'LightGCN':
            bpr_pretrained = model.LightGCN(args, train_matrix, device, device)

        r_pretrained = model.RecommendModel(args).to(device)
        s_pretrained = model.SelectModel(args).to(device)

        if args.intervent_type == 'pretrained':
            ut.load_model_by_name(model_dir=args.model_dir, model=bpr_pretrained, global_step=args.magic_num)
        elif args.intervent_type == 'finetuned':
            bpr_pretrained.name = '{}_Finetuned'.format(args.model_name)
            ut.load_model_by_name(model_dir=args.model_dir, model=bpr_pretrained, global_step=args.magic_num)

        ut.load_model_by_name(model_dir=args.model_dir, model=r_pretrained,
                              global_step=str(args.magic_num) + '-' + str(args.user_emb_dim_r))
        ut.load_model_by_name(model_dir=args.model_dir, model=s_pretrained,
                              global_step=str(args.magic_num) + '-' + str(args.user_emb_dim_s))

        args.learn_intervention = True  # with intervention
        intervention_model = learner.Intervention(args, r_model=r_pretrained, s_model=s_pretrained,
                                                  bpr_model=bpr_pretrained, dataset=pretrain_dataset).to(device)

        bpr_evaluator = evaluator.Evaluator(bpr_pretrained, 10, args.item_size,
                                            train_file=tr_bpr_data,
                                            test_file=ts_bpr_data,
                                            test_neg_file=ts_bpr_data_neg,
                                            neg_num=100,
                                            flag=False)
        # test the pretrained model
        begin_time = time.time()
        bpr_evaluator.model = intervention_model.bpr_model
        if args.model_name == 'LightGCN':
            bpr_evaluator.model.to(device)
            hr, ndcg, auc = bpr_evaluator.evaluate_model(1)
            bpr_evaluator.model.to(device)
        else:
            bpr_evaluator.model.to(device_cpu)
            hr, ndcg, auc = bpr_evaluator.evaluate_model(20)
            bpr_evaluator.model.to(device)
        logger.info(
            "#Testing of pretrained BPR# test_hr:{:.4f}, test_ndcg:{:.4f}, test_auc:{:.4f}, test_time:{:.0f}".format(hr,
                                                                                                                     ndcg,
                                                                                                                     auc,
                                                                                                                     time.time() - begin_time))

        # set optimizer
        for i in intervention_model.parameters():
            i.requires_grad = False
        for i in intervention_model.tau_model.parameters():
            i.requires_grad = True
        tau_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, intervention_model.parameters()), lr=1e-3,
                                         betas=(0.9, 0.999))

        for i in intervention_model.parameters():
            i.requires_grad = False
        for i in intervention_model.bpr_model.parameters():
            i.requires_grad = True
        bpr_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, intervention_model.parameters()), lr=1e-3,
                                         betas=(0.9, 0.999))

        intervention_model.bpr_model.name += '_minmax'
        hr_best = 0
        ndcg_best = 0
        for epoch in range(args.intervent_epoch):
            intervention_model.to(device)
            intervention_model.train()
            max_tau_loss = 0
            bpr_loss = 0
            total_max_tau_loss = 0
            total_bpr_loss = 0

            begin_time = time.time()

            for step in range(args.max_step):
                # updating tau policy model
                tau_optimizer.zero_grad()
                tau_loss, tau_constraint = intervention_model.train_tau()
                if args.policy_mode == 'constraint':
                    L = tau_loss + args.tau_lambda_ * tau_constraint
                elif args.policy_mode == 'normalized':
                    L = tau_loss
                L.backward()
                tau_optimizer.step()
                total_max_tau_loss += L.item()
            total_max_tau_loss /= args.max_step

            for step in range(args.min_step):
                # updating bpr model
                bpr_optimizer.zero_grad()
                bpr_loss = intervention_model.train_bpr(flag=False)
                bpr_loss.backward()
                bpr_optimizer.step()
                total_bpr_loss += bpr_loss.item()
            total_bpr_loss /= args.min_step

            tr_time = time.time() - begin_time
            # if (epoch+1)%1 == 0:
            #     logger.info("Epoch:{}, tau_loss:{:.3f}, bpr_loss:{:.3f}, train_time:{:.0f}"
            #             .format(epoch+1, total_max_tau_loss, total_bpr_loss, tr_time))

            # testing
            if (epoch + 1) % 1 == 0:
                begin_time = time.time()
                bpr_evaluator.model = intervention_model.bpr_model

                if args.model_name == 'LightGCN':
                    bpr_evaluator.model.to(device)
                    hr, ndcg, auc = bpr_evaluator.evaluate_model(1)
                    bpr_evaluator.model.to(device)
                else:
                    bpr_evaluator.model.to(device_cpu)
                    hr, ndcg, auc = bpr_evaluator.evaluate_model(20)
                    bpr_evaluator.model.to(device)
                print(
                    "Intervention_Epoch:{}, tau_loss:{:.3f}, bpr_loss:{:.3f}, test_hr:{:.4f}, test_ndcg:{:.4f}, test_auc:{:.4f}, train_time:{:.0f}, test_time:{:.0f}"
                    .format(epoch + 1, total_max_tau_loss, total_bpr_loss, hr, ndcg, auc, tr_time,
                            time.time() - begin_time))
                logger.info(
                    "Epoch:{}, tau_loss:{:.3f}, bpr_loss:{:.3f}, test_hr:{:.4f}, test_ndcg:{:.4f}, test_auc:{:.4f}, train_time:{:.0f}, test_time:{:.0f}"
                    .format(epoch + 1, total_max_tau_loss, total_bpr_loss, hr, ndcg, auc, tr_time,
                            time.time() - begin_time))

            if (epoch + 1) % 1 == 0:
                model_to_save = intervention_model.bpr_model
                model_to_save.name = '{}_{}_intervention'.format(args.model_name, args.intervent_type)
                ut.save_model_by_name(model_dir=args.model_dir, model=model_to_save,
                                      global_step=str(args.user_emb_dim_s) + '-' + str(epoch))

                if hr + ndcg > hr_best + ndcg_best:
                    hr_best = hr
                    ndcg_best = ndcg
                    logger.info('CURRENT BEST: HR:{:.4f}, NDCG:{:.4f}'.format(hr, ndcg))
                    ut.save_model_by_name(model_dir=args.model_dir, model=model_to_save,
                                          global_step=str(args.user_emb_dim_s) + '-' + str(args.magic_num))

