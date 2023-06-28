import logging
import numpy as np
import torch
import torch.nn.functional as F
import utils as ut
from torch import autograd, nn, optim
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear
import models
import utils as ut

args, _ = ut.get_config()
device = torch.device("cuda:{}".format(args.gpu_id) if (torch.cuda.is_available()) else "cpu")
device_cpu = torch.device("cpu")


class Intervention(nn.Module):
    def __init__(self, args, r_model=None, s_model=None, bpr_model=None, tau_model=None, dataset=None):
        super().__init__()
        self.args = args
        self.name = 'Intervention'
        self.dataset = dataset
        self.r_model = r_model
        self.s_model = s_model
        self.bpr_model = bpr_model
        self.tau_model = models.tau_policy(self.args).to(device)
        self.load_models(r_model=r_model, s_model=s_model, bpr_model=bpr_model)
        self.iter = 0
        self.sample_batch = args.user_batch
        self.topK = args.topK  # k should lower than impression len

        self.intervent_mode = args.intervent_mode  # for ablation study, ['RL', 'NO', 'RANDOM']

    def load_models(self, r_model=None, s_model=None, bpr_model=None, tau_model=None):
        if self.args.learning_mode == 'pretrain':
            if bpr_model is not None:
                print('-----------------------------------------------')
                print('BPR Model from Outside.')
                self.bpr_model = bpr_model
            else:
                print('-----------------------------------------------')
                print('BPR Model Newly Defined.')
                if args.model_name == 'BPR':
                    self.bpr_model = models.BPR(self.args).to(device)
                elif args.model_name in ['NeuBPR', 'gmfBPR', 'mlpBPR']:
                    self.bpr_model = models.NeuBPR(self.args).to(device)

        if self.args.learning_mode == 'train' or self.args.learning_mode == 'intervention':
            if bpr_model is not None:
                print('BPR Model from Pretrained.')
                self.bpr_model = bpr_model
            else:
                print('BPR Model from Scratch.')
                if args.model_name == 'BPR':
                    self.bpr_model = models.BPR(self.args).to(device)
                elif args.model_name in ['NeuBPR', 'gmfBPR', 'mlpBPR', 'bprBPR']:
                    self.bpr_model = models.NeuBPR(self.args).to(device)
                ut.load_model_by_name(self.args.model_dir, self.bpr_model, 0)
            if r_model is not None:
                print('R Model from Pretrained.')
                self.r_model = r_model
            else:
                print('R Model from Scratch.')
                self.r_model = models.RecommendModel(self.args).to(device)
                ut.load_model_by_name(self.args.model_dir, self.r_model, 0)
            if s_model is not None:
                print('S Model from Pretrained.')
                self.s_model = s_model
            else:
                print('S Model from Scratch.')
                self.s_model = models.SelectModel(self.args).to(device)
                ut.load_model_by_name(self.args.model_dir, self.s_model, 0)

            if tau_model is not None:  # only tau need to learn in this process (bpr do not have to learn)
                print('Tau Model from Pretrained.')
                self.tau_model = tau_model
            else:
                print('Tau Model from Scratch.')
                self.tau_model = models.tau_policy(self.args).to(device)

    def sample_user(self):
        user_set = torch.from_numpy(np.arange(self.iter * self.sample_batch, (self.iter + 1) * self.sample_batch, 1))
        self.iter += 1
        if (self.iter % (self.args.user_size // self.sample_batch)) == 0:
            self.iter = 0
        return user_set

    def sample_r(self):
        user_set = self.sample_user()
        # print('***'*10)
        # print('user_set:', user_set.size())
        # print('***'*10)

        user_embedding = self.r_model.user_embedding(user_set)  # sample_batch * emb_dim
        assert user_embedding.size()[1] == self.args.user_emb_dim_r

        # intervention
        if self.intervent_mode == 'RL':
            tau = self.tau_model.policy(user_embedding)
            impression_set = self.r_model.generate_rt(user_set, k=self.args.impression_length, tau=tau)
        elif self.intervent_mode == 'NO':
            tau = torch.zeros([self.args.user_batch, self.args.user_emb_dim_r], dtype=torch.float32).to(device)
            impression_set = self.r_model.generate_rt(user_set, k=self.args.impression_length, tau=tau)
        elif self.intervent_mode == 'RANDOM':
            impression_set = np.random.randint(0, self.args.item_size,
                                               size=(self.args.user_batch, self.args.impression_length))
            impression_set = torch.tensor(impression_set, dtype=torch.int64).to(device)
            tau = self.tau_model.policy(user_embedding)

        return user_set, impression_set, tau

    def sample_s(self):
        user_set, r_set, tau = self.sample_r()
        # print('***'*5 + '{}'.format(self.args.intervent_mode) + '***'*5)
        # print('user_set, r_set', user_set.shape, r_set.shape, tau.shape)
        # print(user_set[:100].cpu().numpy(), '\n', r_set[:100].cpu().numpy())
        # print('***'*10)

        # if self.args.pair_wise_selection:
        #    self.args.impression_length = 1

        # user_set: [batch_size], r_set: [batch_size, impression_length]
        s_prefer, s_not_prefer = self.s_model.generate_st(user_set, r_set, n=self.args.impression_length)

        # print('***'*10)
        # print('s_prefer, s_not_prefer', s_prefer.size(), s_not_prefer.size())
        # print('***'*10)

        return user_set, r_set, tau, s_prefer, s_not_prefer

    def get_samples(self):
        user_set, r_set, tau, s_prefer, s_not_prefer = self.sample_s()
        return user_set, r_set, tau, s_prefer, s_not_prefer

    def train_tau(self):
        user_set, r_set, tau, s_prefer, s_not_prefer = self.get_samples()

        user_set_k = user_set.repeat(self.topK, 1).t().reshape(-1)

        bpr_loss = self.bpr_model.forward(user_set_k, s_prefer, s_not_prefer)

        user_embedding_k = self.r_model.user_embedding(user_set_k)

        tau_loss = self.tau_model.loss(user_embedding_k, -bpr_loss)

        st_tau_loss = torch.norm(tau)
        return tau_loss, st_tau_loss

    def train_bpr(self, flag=False):
        user_set, r_set, tau, s_prefer, s_not_prefer = self.get_samples()
        user_set_k = user_set.repeat(self.topK, 1).t().reshape(-1)

        if flag:
            user_set_k = user_set_k.detach().cpu().numpy().reshape((-1, 1))
            s_prefer = s_prefer.detach().cpu().numpy().reshape((-1, 1))
            s_not_prefer = s_not_prefer.detach().cpu().numpy().reshape((-1, 1))

            users = np.concatenate([user_set_k, user_set_k], axis=0)
            positives = np.concatenate([s_prefer, s_not_prefer], axis=0)
            users, positives, negatives = self.dataset.sample_negatives(users, positives, num_negatives=2)

            user_set_k = torch.tensor(np.concatenate([user_set_k, users], axis=0), dtype=torch.int64)
            s_prefer = torch.tensor(np.concatenate([s_prefer, positives], axis=0), dtype=torch.int64)
            s_not_prefer = torch.tensor(np.concatenate([s_not_prefer, negatives], axis=0), dtype=torch.int64)

        bpr_loss = self.bpr_model.forward(user_set_k, s_prefer, s_not_prefer).mean()
        return bpr_loss

    def pretrain_bpr(self, x, a_prefer, a_not_prefer):
        bpr_loss = self.bpr_model.forward(x, a_prefer, a_not_prefer)
        return bpr_loss

    def pretrain_bpr_one_epoch(self, dataset, optimizer, batch_size, verbose=True):
        # user, item, rating pairs

        if args.model_name in ['BPR', 'bprBPR', 'gmfBPR', 'mlpBPR', 'NeuBPR']:
            user_ids, item_ids, neg_ids = dataset.generate_triple_data()

            num_training = len(user_ids)
            num_batches = int(np.ceil(num_training / batch_size))

            perm = np.random.permutation(num_training)

            loss = 0.0
            for b in range(num_batches):
                optimizer.zero_grad()

                if (b + 1) * batch_size >= num_training:
                    batch_idx = perm[b * batch_size:]
                else:
                    batch_idx = perm[b * batch_size: (b + 1) * batch_size]

                # batch_users = np.array(user_ids[batch_idx], dtype=np.int64)
                # batch_items = np.array(item_ids[batch_idx], dtype=np.int64)
                # batch_negs = np.array(neg_ids[batch_idx], dtype=np.int64)

                batch_users = torch.LongTensor(user_ids[batch_idx])
                batch_items = torch.LongTensor(item_ids[batch_idx])
                batch_negs = torch.LongTensor(neg_ids[batch_idx])
                batch_loss = self.bpr_model.forward(batch_users, batch_items, batch_negs).mean()
                batch_loss.backward()
                optimizer.step()

                loss += batch_loss

                if verbose and b % 50 == 0:
                    print('(%3d / %3d) loss = %.4f' % (b, num_batches, batch_loss))

        elif args.model_name in ['LightGCN']:
            loss = self.bpr_model.train_one_epoch(optimizer, batch_size, verbose)

        return loss


