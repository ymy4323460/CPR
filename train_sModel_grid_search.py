from __future__ import print_function
import argparse
import time

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
    embed_dims = [16, 32, 64, 128, 256, 512]
    # embed_dims = [256, 512]
    weight_decays = [2e-3]
    dropouts = [0.1]

    for lr in learning_rates:
        args.lr_bpr = lr

        for i in range(len(embed_dims)):
            args.user_emb_dim_s = embed_dims[i]
            args.item_emb_dim_s = embed_dims[i]

            for decay in weight_decays:
                args.weight_decay = decay

                for dropout in dropouts:
                    args.dropout = dropout

                    args.learning_mode = 'train'
                    train_selectModel.train_selectModel(args)


if __name__ == '__main__':
    # for debug of config
    args, _ = ut.get_config()
    timestamp = time.strftime("%Y_%m%d_%H%M", time.localtime())
    args.log_file = './results/{}_{}.sModel'.format(args.dataset, timestamp)

    if args.dataset == 'synthetic':

        # # datasets = ['logitdata_5_16', 'logitdata_5_32', 'logitdata_10_16', 'logitdata_10_32', \
        # #     'logitdata_25_16', 'logitdata_25_32', 'logitdata_50_16', 'logitdata_50_32', 'logitdata_100_16', 'logitdata_100_32']
        datasets = ['logitdata_5_16']

        for dataset in datasets:
            args.model_dir = '{}/{}'.format('synthetic_bpr_pretrained', dataset)
            args.dataset = '{}/{}'.format('synthetic', dataset)

            pass

    elif args.dataset == 'mind':
        datasets = ['mind']
        for dataset in datasets:
            args.model_dir = '{}/{}'.format('real_bpr_pretrained', dataset)
            args.dataset = '{}/{}'.format('real', dataset)

            bpr_grid_search(args)

            # train_rModel.train_rModel(args)

'''
commands:
nohup python train_sModel_grid_search.py --dataset mind --gpu_id 0 --train_epoch_max_s 50 >> mind_sModel.txt 2>&1 &
'''