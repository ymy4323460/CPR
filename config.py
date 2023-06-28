from __future__ import print_function
import argparse
import time
import os
import train_bpr as train_bpr
import train_finetune as train_finetune
import train_selectModel as train_selectModel
import train_rModel as train_rModel
import train_intervention as train_intervention

import utils as ut


# def add_argument_group(name):
#     arg = parser.add_argument_group(name)
#     arg_lists.append(arg)
#     return arg

def bpr_grid_search(args):
    # learning_rates = [1e-3, 2*1e-3, 5*1e-3]
    # embed_dims = [64, 128, 256]
    # deep_layers = ['128|64', '256|128', '512|256']
    # weight_decays = [2e-2, 1e-2, 1e-3, 2e-3, 5e-3, 1e-4, 5e-4, 1e-5]
    # dropouts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    learning_rates = [1e-3]
    embed_dims = [64]
    deep_layers = ['128|64']
    weight_decays = [2e-3]
    dropouts = [0.1]

    for lr in learning_rates:
        args.lr_bpr = lr

        for i in range(len(embed_dims)):
            args.user_emb_dim = embed_dims[i]
            args.item_emb_dim = embed_dims[i]
            args.layers = deep_layers[i]

            for decay in weight_decays:
                args.weight_decay = decay

                for dropout in dropouts:
                    args.dropout = dropout

                    # print('\n*****************************')
                    # print('Learning rate:{}, Deep layers:{}, Weight:{} Dropout:{}'.format(lr, args.layers, decay, dropout))
                    # print('*****************************\n')
                    # train_bpr.train_bpr_pretrain(args)

                    return args


if __name__ == '__main__':
    # for debug of config
    args, _ = ut.get_config()
    timestamp = time.strftime("%Y_%m%d_%H%M", time.localtime())

    if args.dataset == 'synthetic':

        # # datasets = ['logitdata_5_16', 'logitdata_5_32', 'logitdata_10_16', 'logitdata_10_32', \
        # #     'logitdata_25_16', 'logitdata_25_32', 'logitdata_50_16', 'logitdata_50_32', 'logitdata_100_16', 'logitdata_100_32']
        datasets = args.datasets

        for dataset in [datasets]:
            args.model_dir = '{}/{}'.format('synthetic_bpr_pretrained', dataset)
            args.dataset = '{}/{}'.format('synthetic', dataset)
            if not os.path.exists('./results/{}/'.format(args.dataset)):
                os.makedirs('./results/{}/'.format(args.dataset))
            # args.log_file = './results/{}/{}_{}.ablation'.format(args.dataset, args.model_name, timestamp)
            args.log_file = './results/logs'
            if args.model_name == 'LightGCN':
                args.negative_num = 1
            args.user_size = 600
            args.item_size = 300
            args.user_batch = 600
            args.r_size = 300
            print(args)
            train_bpr.train_bpr_pretrain(args)
            # train_finetune.train_bpr_finetune(args)

            args.learning_mode = 'train'
            train_selectModel.train_selectModel(args)
            train_rModel.train_rModel(args)

            args.learning_mode = 'intervention'
            args.intervent_epoch = 20
            args.intervent_type = 'pretrained'
            train_intervention.train_intervention(args)

            # args.intervent_type = 'finetuned'
            # train_intervention.train_intervention(args)

    elif args.dataset == 'mind':
        datasets = ['mind']
        for dataset in datasets:
            args.model_dir = '{}/{}'.format('real_bpr_pretrained', dataset)
            args.dataset = '{}/{}'.format('real', dataset)

            if args.model_name == 'LightGCN':
                args.negative_num = 1
                args = bpr_grid_search(args)
            if args.model_name in ['mlpBPR', 'NeuBPR']:
                args = bpr_grid_search(args)

            # different embedding_size makes huge difference on finetuning results
            # the following are searched from [16, 32, 64, 128, 256]
            if args.model_name == 'gmfBPR':
                args.user_emb_dim = 64
                args.item_emb_dim = 64
            if args.model_name == 'mlpBPR':
                args.user_emb_dim = 64
                args.item_emb_dim = 64
            if args.model_name == 'NeuBPR':
                args.user_emb_dim = 32
                args.item_emb_dim = 32
                args.layers = '64|64'
            if args.model_name == 'LightGCN':
                args.user_emb_dim = 16
                args.item_emb_dim = 16

            # #-------------------------------------------
            # args.learning_mode = 'pretrain'
            # train_bpr.train_bpr_pretrain(args)
            # train_finetune.train_bpr_finetune(args)

            # #-----------------------------------------
            # args.learning_mode = 'train'
            # train_selectModel.train_selectModel(args)
            # train_rModel.train_rModel(args)

            # -----------------------------------------

            # -----------------------------------------
            # intervention on pretrained model
            args.learning_mode = 'intervention'
            args.intervent_type = 'pretrained'
            # embed_dims_s = [16, 32, 64, 128, 256, 512]
            embed_dims_s = [16, 32, 64, 128, 256, 512]

            if args.policy_mode == 'normalized':
                args.tau_lambda_ = 0.5
            elif args.policy_mode == 'constraint':
                args.tau_lambda_ = 0.1

            for dim in embed_dims_s:
                args.user_emb_dim_s = dim
                args.item_emb_dim_s = dim
                # args.max_step = 15
                # args.min_step = 1
                train_intervention.train_intervention(args)

            # #-----------------------------------------
            # # intervention on finetuned model
            # '''
            # best setting:
            # tau_lambda_:
            # mlp, gmf: 64 + 0.5, neumf: 32, 0.9
            # min_step = 1, max_step=15
            # '''
            # # embed_dims_s = [16, 32, 64, 128, 256, 512]
            # # embed_dims_s = [32, 64, 128, 256, 512]
            # embed_dims_s = [32]
            # # min_steps = [1, 3, 5, 7, 9, 15]
            # min_steps = [1]
            # # min_steps = [args.min_step]

            # # max_steps = [50, 30, 20, 15, 5]
            # max_steps = [15]
            # # max_steps = [args.max_step]

            # tau_lambdas = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]  # mlp, gmf: 64 + 0.5, neumf: 32, 0.9

            # for dim in embed_dims_s:
            #     args.user_emb_dim_s = dim
            #     args.item_emb_dim_s = dim

            #     for step in min_steps:
            #         args.min_step = step

            #         for max_step in max_steps:
            #             args.max_step = max_step

            #             for lbd in tau_lambdas:
            #                 args.tau_lambda_ = lbd

            #                 args.learning_mode = 'intervention'
            #                 args.intervent_type = 'finetuned'
            #                 train_intervention.train_intervention(args)

'''
commands:

nohup python config.py --model_name LightGCN --dataset mind --gpu_id 1 --intervent_epoch 200 >> mind_lightgcn.txt 2>&1 &


nohup python config.py --model_name bprBPR --dataset mind --gpu_id 0 > mind_bprbpr.txt 2>&1 &
nohup python config.py --model_name gmfBPR --dataset mind --gpu_id 1 --policy_mode constraint > mind_gmfbpr.txt 2>&1 &
nohup python config.py --model_name NeuBPR --dataset mind --gpu_id 0 --policy_mode constraint > mind_neubpr.txt 2>&1 &
nohup python config.py --model_name mlpBPR --dataset mind --gpu_id 1 --policy_mode constraint > mind_mlpbpr.txt 2>&1 &



# grid search
nohup python config.py --model_name mlpBPR --dataset mind --gpu_id 1 --layers '256|128' >> mind_mlpbpr_grid_search.txt 2>&1 &
nohup python config.py --model_name mlpBPR --dataset mind --gpu_id 1 --layers '256|128' --intervent_epoch 500 > mind_mlpbpr_gs_weight_decay.txt 2>&1 &
nohup python config.py --model_name mlpBPR --dataset mind --gpu_id 1 --layers '256|128' --intervent_epoch 500 > mind_mlpbpr_gs_dropout.txt 2>&1 &


ps -ef | grep config.py | grep -v grep | awk '{print $2}' | xargs kill -9
'''

'''
commands:

nohup python config.py --model_name BPR --dataset synthetic >> synthetic_bpr_pretrain_finetune.txt 2>&1 &
nohup python config.py --model_name gmfBPR > synthetic_gmfbpr_pretrain_finetune.txt 2>&1 &
nohup python config.py --model_name bprBPR > synthetic_bprbpr_pretrain_finetune.txt 2>&1 &

nohup python config.py --model_name BPR > mind_bpr_pretrain_finetune.txt 2>&1 &
nohup python config.py --model_name NeuBPR >> mind_neubpr_pretrain_finetune.txt 2>&1 &
nohup python config.py --model_name gmfBPR > mind_gmfbpr_pretrain_finetune.txt 2>&1 &

nohup python config.py --model_name NeuBPR --dataset mind --gpu_id 0 >> mind_neubpr_pretrain.txt 2>&1 &
nohup python config.py --model_name mlpBPR --dataset mind --gpu_id 1 >> mind_mlpbpr_pretrain.txt 2>&1 &

nohup python config.py --model_name NeuBPR --dataset mind --gpu_id 1 >> mind_neubpr_intervention_002.txt 2>&1 &
nohup python config.py --model_name mlpBPR --dataset mind --gpu_id 0 --pretrain_epoch 5 >> mind_mlpbpr_intervention_002.txt 2>&1 &

nohup python config.py --model_name NeuBPR --dataset mind --gpu_id 1 >> mind_neubpr_finetune.txt 2>&1 &
nohup python config.py --model_name mlpBPR --dataset mind --gpu_id 1 --pretrain_epoch 5 >> mind_mlpbpr_finetune.txt 2>&1 &


nohup python config.py --model_name bprBPR --dataset mind --gpu_id 0 >> mind_bprbpr.txt 2>&1 &
nohup python config.py --model_name bprBPR --dataset mind --gpu_id 0 >> mind_bprbpr_pretrain_intervention.txt 2>&1 &
nohup python config.py --model_name bprBPR --dataset mind --gpu_id 1 > mind_bprbpr_finetune_intervention_001.txt 2>&1 &


nohup python config.py --model_name mlpBPR --dataset mind --gpu_id 0 --pretrain_epoch 5 >> mind_mlpbpr_pretrain_intervention.txt 2>&1 &
nohup python config.py --model_name NeuBPR --dataset mind --gpu_id 1 >> mind_neubpr_pretrain_intervention.txt 2>&1 &
nohup python config.py --model_name gmfBPR --dataset mind --gpu_id 0 --pretrain_epoch 1 >> mind_gmfbpr_pretrain_intervention.txt 2>&1 &

nohup python config.py --model_name BPR --dataset mind --gpu_id 0 >> mind_bpr_pretrain_intervention.txt 2>&1 &

nohup python config.py --model_name BPR --dataset mind --gpu_id 1 >> mind_bpr_finetune_intervention.txt 2>&1 &


'''
