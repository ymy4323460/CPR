import os
import time
import argparse
import shutil
import torch
import numpy as np
from torch.nn import functional as F
from scipy import sparse
from scipy.sparse import csr_matrix


def get_config():
    def str2bool(v):
        return v is True or v.lower() in ('true', '1')

    arg_lists = []
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default='counterfactual', help='The main model name')
    parser.add_argument('--time', type=str, default='', help='Current time')
    # Learning process
    parser.add_argument('--pretrain_epoch', type=int, default=10, help='The learning epoches')
    parser.add_argument('--finetune_epoch', type=int, default=1,
                        help='The learning epoches of finetuning BPR with groundtruth impression list.')

    parser.add_argument('--train_epoch_max_r', type=int, default=8, help='The learning epoches')
    parser.add_argument('--train_epsilon_epoch_max_r', type=int, default=8, help='The learning epoches')
    parser.add_argument('--lr_r', type=float, default=1e-3, help="Learning rate of recommend model")

    parser.add_argument('--train_epoch_max_s', type=int, default=200, help='The learning epoches')
    parser.add_argument('--train_epsilon_epoch_max_s', type=int, default=6, help='The learning epoches')
    parser.add_argument('--lr_s', type=float, default=1e-3, help="Learning rate of select model")

    parser.add_argument('--minmax_epochs', type=int, default=120, help='The learning epoches')
    parser.add_argument('--max_step', type=int, default=25, help='The policy learning step')
    parser.add_argument('--min_step', type=int, default=5, help='The bpr learning step')

    parser.add_argument('--iter_save', type=int, default=30, help='The save turn')
    parser.add_argument('--pt_load_path', type=str, default='')

    # Pretrain network
    parser.add_argument('--learning_mode', type=str, default='pretrain',
                        choices=['pretrain', 'train', 'test', 'intervention'], help='Weighted learning')
    parser.add_argument('--learn_epsilon', type=str2bool, default=False,
                        help='learning epsilon posterior')  # false: 4.1, true: 4.2
    parser.add_argument('--learn_intervention', type=str2bool, default=True, help='tau learning')  # true: 4.4

    # train network
    parser.add_argument('--model_dir', type=str, default='', help='The model dir')

    # Used to be an option, but now is solved
    # pretrain_arg.add_argument('--pretrain_type',type=str,default='wasserstein',choices=['wasserstein','gan'])
    parser.add_argument('--user_dim', type=int, default=1, help="User feature dimension")
    parser.add_argument('--item_dim', type=int, default=1, help="Item feature dimension")
    parser.add_argument('--user_emb_dim', type=int, default=128, help="User embedding dimension")
    parser.add_argument('--item_emb_dim', type=int, default=128, help="Item embedding dimension")
    parser.add_argument('--user_size', type=int, default=50000, help="Size of user")
    parser.add_argument('--item_size', type=int, default=20288, help="Size of item")
    parser.add_argument('--pair_wise_selection', type=str2bool, default=False,
                        help='pair wise selection, if true, selection model is binary classification model')

    # selectModel & recommendModel
    parser.add_argument('--user_emb_dim_s', type=int, default=128, help="User embedding dimension of sModel")
    parser.add_argument('--item_emb_dim_s', type=int, default=128, help="Item embedding dimension of sModel")
    parser.add_argument('--user_emb_dim_r', type=int, default=128, help="User embedding dimension of rModel")
    parser.add_argument('--item_emb_dim_r', type=int, default=128, help="Item embedding dimension of rModel")

    parser.add_argument('--recommend_layer_dims', type=int, nargs='+', default=[64, 32, 16],
                        help='Hidden layer dimension of recommendation prediction model')
    parser.add_argument('--select_layer_dims', type=int, nargs='+', default=[64, 32, 16],
                        help='Hidden layer dimension of select prediction model')
    parser.add_argument('--tau_layer_dims', type=int, nargs='+', default=[64, 32, 16],
                        help='Hidden layer dimension of tau policy model')
    parser.add_argument('--weight_decay', type=float, default=0.001, help="weight_decay in BPR model")
    parser.add_argument('--tau_lambda_', type=float, default=0.5, help="upper constraint of tau")
    parser.add_argument('--policy_mode', type=str, default='normalized', choices=['normalized', 'constraint'],
                        help='Weighted learning')
    # data
    parser.add_argument('--dataset', type=str, default='synthetic')
    parser.add_argument('--datasets', type=str, default='logitdata_25_32')

    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--user_batch', type=int, default=1024, help='The size of users for conducting intervention')
    parser.add_argument('--r_size', type=int, default=1500, help='The size of candidate items for R model')
    parser.add_argument('--impression_length', type=int, default=5)
    parser.add_argument('--topK', type=int, default=1)
    parser.add_argument('--num_worker', type=int, default=8,
                        help='number of threads to use for loading and preprocessing data')

    # log
    parser.add_argument("--log_file", help="log file", default="./results/log.log")
    parser.add_argument('--log_level', default="INFO", type=str, help='log level:DEBUG INFO WARN ERROR',
                        choices=["DEBUG", "INFO", "WARN", "ERROR"])

    # -------------------------------------------------------------------------
    # NeuBPR
    parser.add_argument('--layers', type=str,
                        default='256|512|128')  # the first layer is the summation of user&item embedding dimensions
    # parser.add_argument('--layers', type=int, nargs='+', default=[128, 64], help='Hidden layer dimension of mlp of NeuBPR')
    parser.add_argument('--model_name', type=str, default='gmfBPR')
    parser.add_argument('--lr_bpr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--negative_num', type=int, default=5,
                        help='the number for negative items sampled for each positive item during training')

    parser.add_argument('--intervent_type', type=str, default='pretrained', choices=['pretrained', 'finetuned'])
    parser.add_argument('--intervent_epoch', type=int, default=100)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--magic_num', type=int, default=20210127)

    # -------------------------------------------------------------------------
    # LightGCN
    parser.add_argument('--data_name', type=str, default='synthetic')
    parser.add_argument('--graph_dir', type=str, default='./graph')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--node_dropout', type=float, default=0.0)
    parser.add_argument('--split', type=str2bool, default=False)
    parser.add_argument('--num_folds', type=int, default=100)
    parser.add_argument('--reg', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--early_stop', dest='early_stop', action='store_true')
    parser.add_argument('--no_early_stop', dest='early_stop', action='store_false')
    parser.set_defaults(early_stop=True)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--top_k', type=int, nargs='+', default=[10])
    parser.add_argument('--topk', type=int, default=100)
    parser.add_argument('--shrink', type=int, default=100)
    parser.add_argument('--feature_weighting', type=str, default='none')

    # CDAE
    parser.add_argument('--hidden_dim', type=int, default=50)
    parser.add_argument('--corruption_ratio', type=float, default=0.5)
    parser.add_argument('--act', type=str, default='tanh')
    parser.add_argument('--test_batch_size', type=int, default=4096)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # ablation study
    parser.add_argument('--intervent_mode', type=str, default='RL', choices=['RL', 'NO', 'RANDOM'])
    # -------------------------------------------------------------------------

    config, unparsed = parser.parse_known_args()
    current_time = time.localtime(time.time())
    config.time = '{}_{}_{}_{}'.format(current_time.tm_mon, current_time.tm_mday, current_time.tm_hour,
                                       current_time.tm_min)

    print('Loaded ./config.py')
    return config, unparsed


args, _ = get_config()
device = torch.device("cuda:{}".format(args.gpu_id) if (torch.cuda.is_available()) else "cpu")


def load_train_matrix(bpr_file, tr_matrix_file, user_num, item_num):
    if os.path.exists(tr_matrix_file):
        train_matrix = sparse.load_npz(tr_matrix_file)
    else:
        row = []
        col = []
        user2item = {}
        fid = open(bpr_file, 'r')
        line = fid.readline().strip()
        while line:
            uij = line.split(',')
            u, i = int(float(uij[0])), int(float(uij[1]))
            if u in user2item.keys():
                if i not in user2item[u]:
                    user2item[u].append(i)
                    row.append(u)
                    col.append(i)
            else:
                user2item[u] = [i]
                row.append(u)
                col.append(i)
            line = fid.readline().strip()

        data = np.ones(len(row))
        row = np.array(row)
        col = np.array(col)
        train_matrix = csr_matrix((data, (row, col)), shape=(user_num, item_num))

        sparse.save_npz(tr_matrix_file, train_matrix)

    return train_matrix


def sample_gaussian(m, v):
    """
    Element-wise application reparameterization trick to sample from Gaussian

    Args:
        m: tensor: (batch, ...): Mean
        v: tensor: (batch, ...): Variance

    Return:
        z: tensor: (batch, ...): Samples
    """
    ################################################################################
    # TODO: Modify/complete the code here
    # Sample z
    ################################################################################

    ################################################################################
    # End of code modification
    ################################################################################
    sample = torch.randn(m.shape).to(device)

    z = m + (v ** 0.5) * sample
    return z


def save_model_by_name(model_dir, model, global_step):
    save_dir = os.path.join('checkpoints', model_dir, model.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, 'model-{}.pt'.format(global_step))
    state = model.state_dict()
    torch.save(state, file_path)
    print('Saved to {}'.format(file_path))


def load_model_by_name(model_dir, model, global_step):
    """
    Load a model based on its name model.name and the checkpoint iteration step

    Args:
        model: Model: (): A model
        global_step: int: (): Checkpoint iteration
    """
    file_path = os.path.join('checkpoints', model_dir, model.name,
                             'model-{}.pt'.format(global_step))
    state = torch.load(file_path)
    model.load_state_dict(state)
    print("Loaded from {}".format(file_path))


ce = torch.nn.CrossEntropyLoss(reduction='none')


def cross_entropy_loss(x, logits):
    """
    Computes the log probability of a Bernoulli given its logits

    Args:
        x: tensor: (batch, dim): Observation
        logits: tensor: (batch, dim): Bernoulli logits

    Return:
        log_prob: tensor: (batch,): log probability of each sample
    """
    log_prob = ce(input=logits, target=x).sum(-1)
    return log_prob
