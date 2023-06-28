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
# from config import get_config
import logging
import warnings
warnings.filterwarnings('ignore')


def train_bpr_finetune(args):

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

    logger.info('\n\n\n\n\n')
    logger.info('***'*20)
    logger.info('#####BPR Finetuning#####')
    logger.info(args)
    logger.info('***'*20)

    device = torch.device("cuda:{}".format(args.gpu_id) if(torch.cuda.is_available()) else "cpu")
    device_cpu = torch.device("cpu")

    logger.info(torch.cuda.is_available())
    logger.info(device)

    # args, _ = get_config()
    # workstation_path = './'
    # args.model_dir = os.path.join(workstation_path, args.model_dir)
    dataset_dir = os.path.join(args.dataset)

    # mode = 'dynamic_negative_sampling'
    mode = 'original'
    logger.info('negative-sampling-mode: {}'.format(mode))

    if args.learning_mode == 'pretrain':
        tr_bpr_data = os.path.join(dataset_dir, 'train', 'bpr_data.csv')
        tr_bpr_triplet_truth = os.path.join(dataset_dir, 'train', 'bpr_data.csv')
        # ts_bpr_data = os.path.join(dataset_dir, 'dev', 'bpr_data.csv')
        # ts_bpr_data_neg = os.path.join(dataset_dir, 'dev', 'bpr_data_neg.csv')
        ts_bpr_data = os.path.join(dataset_dir, 'dev', 'bpr_data_loo.csv')
        ts_bpr_data_neg = os.path.join(dataset_dir, 'dev', 'bpr_data_loo_neg.csv')

        # for LightGCN
        if args.model_name == 'LightGCN':
            tr_matrix_file = os.path.join(dataset_dir, 'train', 'train_matrix.npz')
            logger.info('Load train matrix...')
            train_matrix = ut.load_train_matrix(tr_bpr_data, tr_matrix_file, args.user_size, args.item_size)
            logger.info('Done loading.')

        if mode=='original':
            pretrain_dataset = dataloader.dataload(tr_bpr_triplet_truth, args.batch_size, shuffle=True)
        else:
            pretrain_dataset = dataset.Dataset(tr_bpr_data, args.item_size, args.negative_num)


    # select_dataloader = ut.SelectedDataLoader(dataset_dir, args.batch_size) # a size is n

    if args.learning_mode == 'pretrain':
        '''
        Todo: learn BPR model here
        '''
        if args.model_name == 'BPR':
            bpr_model = model.BPR(args).to(device)
        elif args.model_name in ['NeuBPR', 'gmfBPR', 'mlpBPR', 'bprBPR']:
            bpr_model = model.NeuBPR(args).to(device)
        elif args.model_name == 'LightGCN':
            bpr_model = model.LightGCN(args, train_matrix, device, device)

        bpr_evaluator = evaluator.Evaluator(bpr_model, 10, args.item_size,
                                            train_file=tr_bpr_data,
                                            test_file=ts_bpr_data,
                                            test_neg_file=ts_bpr_data_neg,
                                            neg_num=100,
                                            flag=False)

        # test the pretrained model
        # ut.load_model_by_name(model_dir=args.model_dir, model=bpr_model, global_step=args.pretrain_epoch-1)
        ut.load_model_by_name(model_dir=args.model_dir, model=bpr_model, global_step=args.magic_num)

        begin_time = time.time()
        bpr_evaluator.model = bpr_model
        if args.model_name == 'LightGCN':
            bpr_evaluator.model.to(device)
            hr, ndcg, auc = bpr_evaluator.evaluate_model(1)
            bpr_evaluator.model.to(device)
        else:
            bpr_evaluator.model.to(device_cpu)
            hr, ndcg, auc = bpr_evaluator.evaluate_model(20)
            bpr_evaluator.model.to(device)
        logger.info("#Testing of pretrained BPR# test_hr:{:.4f}, test_ndcg:{:.4f}, test_auc:{:.4f}, test_time:{:.0f}".format(hr, ndcg, auc, time.time()-begin_time))

        intervention_model = learner.Intervention(args=args, bpr_model=bpr_model, dataset=pretrain_dataset).to(device)
        pretrain_optimizer = torch.optim.Adam(intervention_model.parameters(), lr=1e-3, betas=(0.9, 0.999))
        hr_best = 0
        ndcg_best = 0
        for epoch in range(args.finetune_epoch):

            # if (epoch+1) == 1:
            #     model_to_save = intervention_model.bpr_model
            #     model_to_save.name = '{}_Finetuned'.format(args.model_name)
            #     ut.save_model_by_name(model_dir=args.model_dir, model=model_to_save, global_step=epoch)

            intervention_model.train()
            bpr_loss = 0
            total_bpr_loss = 0

            begin_time = time.time()
            if mode=='original':
                step = 0
                step_loss = 0
                for x, a_prefer, a_not_prefer in pretrain_dataset:
                    pretrain_optimizer.zero_grad()
                    x, a_prefer, a_not_prefer = x.to(device), a_prefer.to(device), a_not_prefer.to(device)
                    bpr_loss = intervention_model.pretrain_bpr(x, a_prefer, a_not_prefer).mean()
                    bpr_loss.backward()
                    pretrain_optimizer.step()
                    loss = bpr_loss.item()
                    total_bpr_loss += loss
                    step_loss += loss
                    # m = len(pretrain_dataloader)

                    step += 1
                    if (step+1) % 100 == 0:
                        begin_t = time.time()
                        # bpr_model.to(device_cpu)
                        # bpr_model.eval()
                        bpr_evaluator.model = intervention_model.bpr_model

                        if args.model_name == 'LightGCN':
                            bpr_evaluator.model.to(device)
                            hr, ndcg, auc = bpr_evaluator.evaluate_model(1)
                            bpr_evaluator.model.to(device)
                        else:
                            bpr_evaluator.model.to(device_cpu)
                            hr, ndcg, auc = bpr_evaluator.evaluate_model(20)
                            bpr_evaluator.model.to(device)

                        logger.info("Step:{}, step_loss:{:.3f}, test_hr:{:.4f}, test_ndcg:{:.4f}, test_auc:{:.4f}, test_time:{:.0f}"
                            .format(step, step_loss, hr, ndcg, auc, time.time()-begin_t))
                        step_loss = 0

                        # model save
                        if hr+ndcg > hr_best+ndcg_best:
                            hr_best = hr
                            ndcg_best = ndcg
                            logger.info('CURRENT BEST: HR:{:.4f}, NDCG:{:.4f}'.format(hr, ndcg))
                            model_to_save = intervention_model.bpr_model
                            model_to_save.name = '{}_Finetuned'.format(args.model_name)
                            ut.save_model_by_name(model_dir=args.model_dir, model=model_to_save, global_step=args.magic_num)
            else:
                total_bpr_loss = intervention_model.pretrain_bpr_one_epoch(pretrain_dataset, pretrain_optimizer, args.batch_size)

            tr_time = time.time() - begin_time

            if (epoch+1) % 1 == 0:
                begin_time = time.time()
                # bpr_model.to(device_cpu)
                # bpr_model.eval()
                bpr_evaluator.model = intervention_model.bpr_model

                if args.model_name == 'LightGCN':
                    bpr_evaluator.model.to(device)
                    hr, ndcg, auc = bpr_evaluator.evaluate_model(1)
                    bpr_evaluator.model.to(device)
                else:
                    bpr_evaluator.model.to(device_cpu)
                    hr, ndcg, auc = bpr_evaluator.evaluate_model(20)
                    bpr_evaluator.model.to(device)

                logger.info("Epoch:{}, total_bpr_loss: {:.3f}, test_hr:{:.4f}, test_ndcg:{:.4f}, test_auc:{:.4f}, train_time:{:.0f}, test_time:{:.0f}"
                    .format(epoch+1, total_bpr_loss, hr, ndcg, auc, tr_time, time.time()-begin_time))

            # if (epoch+1) % args.iter_save == 0:
            if (epoch+1) == 1:
                model_to_save = intervention_model.bpr_model
                model_to_save.name = '{}_Finetuned'.format(args.model_name)
                ut.save_model_by_name(model_dir=args.model_dir, model=model_to_save, global_step=epoch)
