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
from config import get_config
import warnings

warnings.filterwarnings('ignore')

device = torch.device("cuda:1" if (torch.cuda.is_available()) else "cpu")
device_cpu = torch.device("cpu")

print(torch.cuda.is_available())
print(device)

args, _ = get_config()

print('***' * 10)
print(args)
print('***' * 10)

workstation_path = './'
# args.model_dir = os.path.join(workstation_path, args.model_dir)
dataset_dir = os.path.join(workstation_path, args.dataset)

mode = 'dynamic_negative_sampling'
# mode = 'original'
print('negative-sampling-mode:', mode)

if args.learning_mode == 'pretrain':
    # no dynamic negative sampling
    # -----------------------------------Quanyu Dai-------------------------------------
    # pretrain_dataloader = dataloader.dataload(os.path.join(dataset_dir, 'train', 'bpr_data.csv'), args.batch_size)
    # pretest_dataloader = dataloader.dataload(os.path.join(dataset_dir, 'dev', 'bpr_data.csv'), args.batch_size)
    if mode == 'original':
        pretrain_dataloader = dataloader.dataload(os.path.join(dataset_dir, 'train', 'bpr_data.csv'), args.batch_size)
    else:
        pretrain_dataset = dataset.Dataset(os.path.join(dataset_dir, 'train', 'bpr_data.csv'), args.item_size, 5)
    # ----------------------------------------------------------------------------------
if args.learning_mode == 'train':
    if args.pair_wise_selection:
        if mode == 'original':
            train_r_dataloader = dataloader.dataload(os.path.join(dataset_dir, 'train', 'data.csv'),
                                                     args.batch_size)  # a size is 1
            test_r_dataloader = dataloader.dataload(os.path.join(dataset_dir, 'dev', 'data.csv'), args.batch_size)
        else:
            train_r_dataset = dataset.Dataset(os.path.join(dataset_dir, 'train', 'data.csv'), args.item_size, 1)

        train_s_dataloader = dataloader.dataload(os.path.join(dataset_dir, 'train', 'data.csv'), args.batch_size)
        test_s_dataloader = dataloader.dataload(os.path.join(dataset_dir, 'dev', 'data.csv'), args.batch_size)
    else:
        if mode == 'original':
            train_r_dataloader = dataloader.dataload(os.path.join(dataset_dir, 'train', 'data.csv'),
                                                     args.batch_size)  # a size is 1
            test_r_dataloader = dataloader.dataload(os.path.join(dataset_dir, 'dev', 'data.csv'), args.batch_size)
        else:
            train_r_dataset = dataset.Dataset(os.path.join(dataset_dir, 'train', 'data.csv'), args.item_size, 1)

        train_s_dataloader = dataloader.list_dataload(os.path.join(dataset_dir, 'train', 'list_impression.csv'),
                                                      args.batch_size)
        # test_s_dataloader = dataloader.list_dataload(os.path.join(dataset_dir, 'dev', 'list_impression.csv'), args.batch_size)
        test_s_x, test_s_a, test_s_y = dataset.load_impression_list_file(
            os.path.join(dataset_dir, 'dev', 'list_impression.csv'))

if args.learning_mode == 'intervention':
    pretrain_dataset = dataset.Dataset(os.path.join(dataset_dir, 'train', 'bpr_data.csv'), args.item_size, 5)

# select_dataloader = ut.SelectedDataLoader(dataset_dir, args.batch_size) # a size is n

if args.learning_mode == 'pretrain':
    '''
    Todo: learn BPR model here
    '''
    bpr_model = model.BPR(args).to(device)

    bpr_evaluator = evaluator.Evaluator(bpr_model, 10, args.item_size,
                                        train_file='./hw/train/bpr_data.csv',
                                        test_file='./hw/dev/bpr_data.csv',
                                        test_neg_file='./hw/dev/bpr_data_neg.csv',
                                        neg_num=100,
                                        flag=False)

    intervention_model = learner.Intervention(args=args, bpr_model=bpr_model, dataset=pretrain_dataset).to(device)
    pretrain_optimizer = torch.optim.Adam(intervention_model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    for epoch in range(args.pretrain_epoch):
        intervention_model.train()
        bpr_loss = 0
        total_bpr_loss = 0

        begin_time = time.time()
        if mode == 'original':
            for x, a_prefer, a_not_prefer in pretrain_dataloader:
                pretrain_optimizer.zero_grad()
                x, a_prefer, a_not_prefer = x.to(device), a_prefer.to(device), a_not_prefer.to(device)
                bpr_loss = intervention_model.pretrain_bpr(x, a_prefer, a_not_prefer)
                bpr_loss.backward()
                pretrain_optimizer.step()
                total_bpr_loss += bpr_loss.item()
                # m = len(pretrain_dataloader)
        else:
            total_bpr_loss = intervention_model.pretrain_bpr_one_epoch(pretrain_dataset, pretrain_optimizer,
                                                                       args.batch_size)

        tr_time = time.time() - begin_time

        if (epoch + 1) % 1 == 0:
            begin_time = time.time()
            # bpr_model.to(device_cpu)
            # bpr_model.eval()
            bpr_evaluator.model = intervention_model.bpr_model
            bpr_evaluator.model.to(device_cpu)
            hr, ndcg, auc = bpr_evaluator.evaluate_model(20)
            bpr_evaluator.model.to(device)
            print(
                "epoch:{}, total_bpr_loss: {:.3f}, test_hr:{}, test_ndcg:{}, test_auc:{}, train_time:{:.0f}, test_time:{:.0f}"
                .format(epoch + 1, total_bpr_loss, hr, ndcg, auc, tr_time, time.time() - begin_time))

        # if (epoch+1) % args.iter_save == 0:
        if (epoch + 1) == args.pretrain_epoch:
            ut.save_model_by_name(model_dir=args.model_dir, model=intervention_model.bpr_model, global_step=epoch)

    # # test the pretrained model
    # bpr_model_load = model.BPR(args).to(device)
    # ut.load_model_by_name(model_dir=args.model_dir, model=bpr_model_load, global_step=args.pretrain_epoch-1)

    # begin_time = time.time()
    # bpr_evaluator.model = bpr_model_load
    # bpr_evaluator.model.to(device_cpu)
    # hr, ndcg, auc = bpr_evaluator.evaluate_model(20)
    # bpr_evaluator.model.to(device)
    # print("#Testing of pretrained BPR# test_hr:{}, test_ndcg:{}, test_auc:{}, test_time:{:.0f}".format(hr, ndcg, auc, time.time()-begin_time))

if args.learning_mode == 'train':
    recommed_model = model.RecommendModel(args).to(device)
    select_model = model.SelectModel(args).to(device)

    r_evaluator = evaluator.Evaluator(recommed_model, 10, args.item_size,
                                      train_file='./hw/train/data.csv',
                                      test_file='./hw/dev/data.csv',
                                      test_neg_file='./hw/dev/data_neg.csv',
                                      neg_num=100,
                                      flag=False)

    # RecommendModel
    # train_PQ_XY
    # epsilon lookup table should be fixed as item_size*[0,1], which means the prior of epsilon is N(0,I)
    recommed_model.args.learn_epsilon = False
    for i in recommed_model.parameters():
        i.requires_grad = True
    for i in recommed_model.er_embedding_lookup.parameters():
        i.requires_grad = False
    recommend_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, recommed_model.parameters()), lr=1e-3,
                                           betas=(0.9, 0.999))

    for epoch in range(args.train_epoch_max_r):
        recommed_model.to(device)
        recommed_model.train()
        recommend_loss = 0
        total_recommend_loss = 0
        begin_time = time.time()
        if mode == 'original':
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
            begin_time = time.time()
            # bpr_model.to(device_cpu)
            # bpr_model.eval()
            r_evaluator.model = recommed_model
            r_evaluator.model.to(device_cpu)
            hr, ndcg, auc = r_evaluator.evaluate_model(20)
            r_evaluator.model.to(device)
            print(
                "epoch:{}, total_recommend_loss: {:.3f}, test_hr:{}, test_ndcg:{}, test_auc:{}, train_time:{:.0f}, test_time:{:.0f}"
                .format(epoch + 1, total_recommend_loss, hr, ndcg, auc, tr_time, time.time() - begin_time))

        # if epoch % args.iter_save == 0:
        if epoch == args.train_epoch_max_r - 1:
            ut.save_model_by_name(model_dir=args.model_dir, model=recommed_model, global_step=epoch)

    # train_epsilon: RecommendModel
    ut.load_model_by_name(args.model_dir, recommed_model, args.train_epoch_max_r - 1)

    recommed_model.args.learn_epsilon = True
    # only mu sigma in epsilon's posterior should be learned in this turn
    for i in recommed_model.parameters():
        i.requires_grad = False
    for i in recommed_model.er_embedding_lookup.parameters():
        i.requires_grad = True
    recommend_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, recommed_model.parameters()), lr=1e-3,
                                           betas=(0.9, 0.999))
    # turn to epsilon prior
    # recommed_model.set_epsilon()
    for epoch in range(args.train_epsilon_epoch_max_r):
        recommed_model.to(device)
        recommed_model.train()
        recommend_loss = 0
        total_recommend_loss = 0

        begin_time = time.time()
        if mode == 'original':
            for x, a, r, _ in train_r_dataloader:
                # train recommended loss
                recommend_optimizer.zero_grad()
                x, a, r = x.to(device), a.to(device), r.to(device)
                recommend_loss = recommed_model.epsilon_loss(x, a, r)
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
            hr, ndcg, auc = r_evaluator.evaluate_model(20)
            r_evaluator.model.to(device)
            print(
                "epoch:{}, total_recommend_loss: {:.3f}, test_hr:{}, test_ndcg:{}, test_auc:{}, train_time:{:.0f}, test_time:{:.0f}"
                .format(epoch + 1, total_recommend_loss, hr, ndcg, auc, tr_time, time.time() - begin_time))

        # if epoch % args.iter_save == 0:
        if epoch == args.train_epsilon_epoch_max_r - 1:
            ut.save_model_by_name(model_dir=args.model_dir, model=recommed_model,
                                  global_step=args.train_epsilon_epoch_max_r - 1)

    # # selection model
    # select_model.args.learn_epsilon = False
    # for i in select_model.parameters():
    #     i.requires_grad=True
    # for i in select_model.es_embedding_lookup.parameters():
    #     i.requires_grad=False
    # select_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, select_model.parameters()), lr=1e-3, betas=(0.9, 0.999))

    # best_acc = 0
    # best_epoch = 0
    # count = 0
    # for epoch in range(args.train_epoch_max_s):

    #     select_model.train()
    #     select_loss = 0
    #     total_select_loss = 0

    #     begin_time = time.time()
    #     if args.pair_wise_selection:
    #         for x, a, _, y in train_s_dataloader:
    #             select_optimizer.zero_grad()
    #             x, a, y = x.to(device), a.to(device), y.to(device)
    #             select_loss = select_model.loss(x, a, y)
    #             select_loss.backward()
    #             select_optimizer.step()
    #             total_select_loss += select_loss.item()
    #             ms = len(train_s_dataloader)
    #     else:
    #         for x, a, y in train_s_dataloader:
    #             select_optimizer.zero_grad()
    #             x, a, y = x.to(device), a.to(device), y.to(device)
    #             select_loss = select_model.loss(x, a, y)
    #             select_loss.backward()
    #             select_optimizer.step()
    #             total_select_loss += select_loss.item()
    #             ms = len(train_s_dataloader)
    #     total_select_loss /= ms
    #     tr_time = time.time() - begin_time

    #     if (epoch+1) % 1 == 0:
    #         count += 1
    #         acc = select_model.evaluate(test_s_x, test_s_a, test_s_y)

    #         if acc > best_acc:
    #             count = 0
    #             best_acc = acc
    #             best_epoch = epoch
    #             ut.save_model_by_name(model_dir=args.model_dir, model=select_model, global_step=epoch)

    #         print("Train total_select_loss:{}, Test acc:{:.3f}".format(total_select_loss, acc))

    #     # early stop
    #     if count>=3:
    #         break

    #     # # if epoch % args.iter_save == 0:
    #     # if epoch==args.train_epoch_max_s-1:
    #     #     ut.save_model_by_name(model_dir=args.model_dir, model=select_model, global_step=epoch)
    # #--------------------------------------------------------------------------

    # #----------------------------------------------------------------------------
    # # train_epsilon: SelectModel
    # # ut.load_model_by_name(args.model_dir, select_model, args.train_epoch_max_s-1)
    # ut.load_model_by_name(args.model_dir, select_model, best_epoch)

    # select_model.args.learn_epsilon = True
    # # only mu sigma in epsilon's posterior should be learned in this turn
    # for i in select_model.parameters():
    #     i.requires_grad=False
    # for i in select_model.es_embedding_lookup.parameters():
    #     i.requires_grad=True
    # # for name, pram in select_model.named_parameters():
    # #     print('requires_grad:{}:{}'.format(name, pram.requires_grad))
    # select_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, select_model.parameters()),  lr=1e-3, betas=(0.9, 0.999))
    # # turn to epsilon prior
    # #select_model.set_epsilon()
    # for epoch in range(args.train_epsilon_epoch_max_s):
    #     select_model.train()
    #     select_loss = 0
    #     total_select_loss = 0

    #     if args.pair_wise_selection:
    #         for x, a, _, y in train_s_dataloader:
    #             select_optimizer.zero_grad()
    #             x, a, y = x.to(device), a.to(device), y.to(device)
    #             select_loss = select_model.epsilon_loss(x, a, y)
    #             select_loss.backward()
    #             select_optimizer.step()
    #             total_select_loss += select_loss.item()
    #             ms = len(train_s_dataloader)
    #     else:
    #         for x, a, y in train_s_dataloader:
    #             select_optimizer.zero_grad()
    #             x, a, y = x.to(device), a.to(device), y.to(device)
    #             select_loss = select_model.epsilon_loss(x, a, y)
    #             select_loss.backward()
    #             select_optimizer.step()
    #             total_select_loss += select_loss.item()
    #             ms = len(train_s_dataloader)
    #     if (epoch+1) % 1 == 0:
    #         acc = select_model.evaluate(test_s_x, test_s_a, test_s_y)
    #         print("Train epsilon turn : {} total_select_loss:{}, Test acc:{:.3f}".format(epoch, total_select_loss/ms, acc))
    #     # if epoch % args.iter_save == 0:
    #     if epoch==args.train_epsilon_epoch_max_s-1:
    #         ut.save_model_by_name(model_dir=args.model_dir, model=select_model, global_step=args.train_epsilon_epoch_max_s-1)

if args.learning_mode == 'intervention':

    if args.learn_intervention:
        # load pretrained model
        bpr_pretrained = model.BPR(args).to(device)
        r_pretrained = model.RecommendModel(args).to(device)
        s_pretrained = model.SelectModel(args).to(device)
        ut.load_model_by_name(model_dir=args.model_dir, model=bpr_pretrained, global_step=args.pretrain_epoch - 1)
        ut.load_model_by_name(model_dir=args.model_dir, model=r_pretrained,
                              global_step=args.train_epsilon_epoch_max_r - 1)
        ut.load_model_by_name(model_dir=args.model_dir, model=s_pretrained,
                              global_step=args.train_epsilon_epoch_max_s - 1)

        intervention_model = learner.Intervention(args, r_model=r_pretrained, s_model=s_pretrained,
                                                  bpr_model=bpr_pretrained, dataset=pretrain_dataset).to(device)

        bpr_evaluator = evaluator.Evaluator(bpr_pretrained, 10, args.item_size,
                                            train_file='./hw/train/bpr_data.csv',
                                            test_file='./hw/dev/bpr_data.csv',
                                            test_neg_file='./hw/dev/bpr_data_neg.csv',
                                            neg_num=100,
                                            flag=False)
        # test the pretrained model
        begin_time = time.time()
        bpr_evaluator.model = intervention_model.bpr_model
        bpr_evaluator.model.to(device_cpu)
        hr, ndcg, auc = bpr_evaluator.evaluate_model(20)
        bpr_evaluator.model.to(device)
        print(
            "#Testing of pretrained BPR# test_hr:{}, test_ndcg:{}, test_auc:{}, test_time:{:.0f}".format(hr, ndcg, auc,
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
        for epoch in range(args.minmax_epochs):
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
                L = tau_loss + 0.1 * tau_constraint
                L.backward()
                tau_optimizer.step()
                total_max_tau_loss += L.item()
            total_max_tau_loss /= args.max_step

            if (epoch + 1) >= 0:
                for step in range(args.min_step):
                    # updating bpr model
                    bpr_optimizer.zero_grad()
                    bpr_loss = intervention_model.train_bpr(flag=False)
                    bpr_loss.backward()
                    bpr_optimizer.step()
                    total_bpr_loss += bpr_loss.item()
                total_bpr_loss /= args.min_step

                tr_time = time.time() - begin_time
                if (epoch + 1) % 50 == 0:
                    print("epoch:{}, tau_loss:{:.3f}, bpr_loss:{:.3f}, train_time:{:.0f}"
                          .format(epoch + 1, total_max_tau_loss, total_bpr_loss, tr_time))

            # testing
            if (epoch + 1) % 100 == 0 and (epoch + 1) >= 1500:
                begin_time = time.time()
                bpr_evaluator.model = intervention_model.bpr_model
                bpr_evaluator.model.to(device_cpu)
                hr, ndcg, auc = bpr_evaluator.evaluate_model(20)
                bpr_evaluator.model.to(device)
                print(
                    "epoch:{}, tau_loss:{:.3f}, bpr_loss:{:.3f}, test_hr:{}, test_ndcg:{}, test_auc:{}, train_time:{:.0f}, test_time:{:.0f}"
                    .format(epoch + 1, total_max_tau_loss, total_bpr_loss, hr, ndcg, auc, tr_time,
                            time.time() - begin_time))

            # if step % args.iter_save == 0:
            #     ut.save_model_by_name(model_dir=args.model_dir, model=intervention_model, global_step=epoch)
