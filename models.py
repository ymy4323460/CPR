import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import autograd, nn, optim
import numpy
import math
import time
import utils as ut

args, _ = ut.get_config()
device = torch.device("cuda:{}".format(args.gpu_id) if (torch.cuda.is_available()) else "cpu")
device_cpu = torch.device("cpu")


class tau_policy(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.name = 'tau_policy'
        self.x_dim = args.user_dim

        self.x_emb_dim = args.user_emb_dim_r
        # self.x_emb_dim = args.user_item_dim_r_s_model

        self.x_size = args.user_size
        self.layer_dim = args.tau_layer_dims
        self.policy_mode = args.policy_mode
        self.tau_lambda = args.tau_lambda_

        self.policy_logit_net = nn.Sequential(
            nn.Linear(self.x_emb_dim, self.layer_dim[0]),
            nn.ELU(),
            nn.Linear(self.layer_dim[0], self.x_emb_dim),
        )

        self.softmax = torch.nn.Softmax()
        self.tanh = torch.nn.Tanh()

    def policy(self, x):
        x_emb = torch.tensor(x, dtype=torch.float32).to(device).reshape(x.size()[0], self.x_emb_dim)
        if self.policy_mode == 'constraint':
            output = self.tanh(self.policy_logit_net(x_emb)) + 0.1 * torch.randn(x_emb.size()[0], x_emb.size()[1],
                                                                                 out=None).to(device)
        elif self.policy_mode == 'normalized':
            output = self.tanh(self.policy_logit_net(x_emb)) + 0.1 * torch.randn(x_emb.size()[0], x_emb.size()[1],
                                                                                 out=None).to(device)
            # output = self.tanh(self.policy_logit_net(x_emb))
            output = self.tau_lambda * F.normalize(output, p=2, dim=1)
        return output

    def loss(self, x, reward):
        policy = self.policy(x)
        return (reward * policy.mean(1)).mean()


class RecommendModel(nn.Module):
    '''
    R = f_2(user, epsilon_r)
    '''

    def __init__(self, args):
        super().__init__()
        self.name = 'Recommendation_Model'
        self.args = args
        self.x_dim = args.user_dim
        self.a_dim = args.item_dim
        self.x_size = args.user_size
        self.a_size = args.item_size

        self.x_emb_dim = args.user_emb_dim_r
        self.a_emb_dim = args.item_emb_dim_r
        # self.x_emb_dim = args.user_item_dim_r_s_model
        # self.a_emb_dim = args.user_item_dim_r_s_model

        self.layer_dim = args.recommend_layer_dims
        self.r_size = args.r_size

        self.user_embedding_lookup = nn.Embedding(self.x_size, self.x_emb_dim)
        self.item_embedding_lookup = nn.Embedding(self.a_size, self.a_emb_dim)
        self.set_epsilon()

        self.user_net = nn.Sequential(
            nn.Linear(self.x_dim, self.layer_dim[0]),
            nn.ReLU(),
            nn.Linear(self.layer_dim[0], self.x_emb_dim),
        )

        self.item_net = nn.Sequential(
            nn.Linear(self.a_dim, self.layer_dim[1]),
            nn.ReLU(),
            nn.Linear(self.layer_dim[1], self.a_emb_dim)
        )

        self.exogenous_net = nn.Linear(1, 1)

        self.recomend_logit_net = nn.Sequential(
            nn.Linear(self.x_emb_dim + self.a_emb_dim, self.layer_dim[0]),
            nn.ELU(),  # activation function
            nn.Linear(self.layer_dim[0], self.layer_dim[1]),
            nn.ELU(),
            nn.Linear(self.layer_dim[1], 2)
        )

        self.sigmoid = torch.nn.Sigmoid()
        self.bce = torch.nn.CrossEntropyLoss()

    # self.bce = torch.nn.BCELoss()

    def set_epsilon(self):
        '''
        ?????: why random embedding initialization when learning epsilon,
        while using zero mean and one variance when training recommendation model
        '''
        #		if self.args.learn_epsilon:
        #			self.er_embedding_lookup = nn.Embedding(self.a_size, 2).to(device)
        #		else:
        #			with torch.no_grad():
        er_init = torch.cat((torch.zeros(self.a_size, 1), torch.ones(self.a_size, 1)), dim=1).to(device)
        self.er_embedding_lookup = nn.Embedding.from_pretrained(er_init)

    def user_embedding(self, x):
        x = torch.tensor(x, dtype=torch.int64).to(device).reshape(x.size()[0], 1)
        user_emb = self.user_embedding_lookup(x).reshape(x.size()[0], self.x_emb_dim)
        return user_emb

    def embedding(self, x, a):
        x = torch.tensor(x, dtype=torch.int64).to(device).reshape(x.size()[0], 1)
        a = torch.tensor(a, dtype=torch.int64).to(device).reshape(a.size()[0], 1)
        user_emb = self.user_embedding_lookup(x).reshape(x.size()[0], self.x_emb_dim)
        item_emb = self.item_embedding_lookup(a).reshape(a.size()[0], self.a_emb_dim)

        er_m_v = self.er_embedding_lookup(a).reshape(a.size()[0], 2)
        # print(er_m_v)
        return user_emb, item_emb, er_m_v[:, 0], torch.abs(er_m_v[:, 1])

    def loss(self, x, a, y):
        assert not self.args.learn_epsilon
        y = torch.tensor(y, dtype=torch.int64).to(device)

        y_hat, _, _, _ = self.predict(x, a)
        return self.bce(y_hat,
                        y)  # +torch.mean(torch.norm(user_emb, p=2, dim=1))+torch.mean(torch.norm(item_emb, p=2, dim=1))

    def epsilon_loss(self, x, a, y):
        assert self.args.learn_epsilon
        y = torch.tensor(y, dtype=torch.int64).to(device)
        y_hat, er, er_m, er_v = self.predict(x, a)
        q_er = torch.mean(
            torch.log(1.0 / (er_v * torch.sqrt(2 * torch.tensor(math.pi, dtype=torch.float32)))) - (er - er_m).pow(
                2) / er_v.pow(2))
        # print(q_er, er, er.size())
        return -self.bce(y_hat, y) + torch.mean(
            torch.norm(er, p=2, dim=1)) - q_er  # +torch.mean(torch.norm(item_emb, p=2, dim=1))

    def predict(self, x, a, tau=None, mode='train'):
        if mode == 'train':
            user_emb, item_emb, er_m, er_v = self.embedding(x, a)
            if tau is not None:
                assert self.args.learning_mode == 'intervention'
                user_emb += tau
            e_r = ut.sample_gaussian(er_m, er_v).reshape(x.size()[0], 1)
            # print(user_emb.size(), item_emb.size())
            return self.recomend_logit_net(torch.cat((user_emb, item_emb), 1)) + self.exogenous_net(
                e_r), e_r, er_m, er_v
        elif mode == 'test':
            x = torch.tensor(x, dtype=torch.int64).to(device_cpu).reshape(x.shape[0], 1)
            a = torch.tensor(a, dtype=torch.int64).to(device_cpu).reshape(a.shape[0], 1)
            user_emb = self.user_embedding_lookup(x).reshape(x.size()[0], self.x_emb_dim)
            item_emb = self.item_embedding_lookup(a).reshape(a.size()[0], self.a_emb_dim)
            er_m_v = self.er_embedding_lookup(a).reshape(a.size()[0], 2)
            er_m = er_m_v[:, 0]
            er_v = torch.abs(er_m_v[:, 1])
            e_r = er_m + (er_v ** 0.5) * torch.randn(er_m.shape).to(device_cpu)
            e_r = e_r.reshape(x.size()[0], 1)
            predictions = F.softmax(
                self.recomend_logit_net(torch.cat((user_emb, item_emb), 1)) + self.exogenous_net(e_r), dim=1)
            # predictions = self.sigmoid(self.recomend_logit_net(torch.cat((user_emb, item_emb), 1)) + self.exogenous_net(e_r)[:, 1])
            # print('-------------------------------')
            # print(predictions.detach().cpu().numpy())
            # print('-------------------------------')
            return predictions.detach().cpu().numpy()[:, 1]

    def train_one_epoch(self, dataset, optimizer, batch_size, verbose=False):
        # user, item, rating pairs
        user_ids, item_ids, labels = dataset.generate_pairwise_data()

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

            batch_users = torch.LongTensor(user_ids[batch_idx]).to(device)
            batch_items = torch.LongTensor(item_ids[batch_idx]).to(device)
            batch_labels = torch.LongTensor(labels[batch_idx]).to(device)

            if not self.args.learn_epsilon:
                batch_loss = self.loss(batch_users, batch_items, batch_labels)
            else:
                batch_loss = self.epsilon_loss(batch_users, batch_items, batch_labels)

            batch_loss.backward()
            optimizer.step()

            loss += batch_loss

            if verbose and b % 50 == 0:
                print('(%3d / %3d) loss = %.4f' % (b, num_batches, batch_loss))
        return loss


    def generate_rt(self, x, k, tau=None):
        '''
         - x: user_embedding
         - k: impression length
        '''
        n = self.r_size
        x_size = x.size()[0]  # number of users
        if x_size == 1:
            pass
        # x = x.repeat(n)
        # all_a = torch.from_numpy(np.random.randint(0, self.args.item_size, size=n))
        # pred_rt = self.predict(x, all_a, tau).reshape(x_size, n)
        # topk_rt_value, topk_rt = pred_rt.topk(k, dim=1, largest=True, sorted=True) #pred 1 * a_size
        else:
            tau = tau.reshape(tau.shape[0], 1, tau.shape[1])
            tau = tau.repeat(1, n, 1).reshape(-1, tau.shape[-1])

            x = x.reshape(x.size()[0], 1).repeat(1, n).reshape(-1, 1)
            # print(x)
            all_a = torch.from_numpy(np.random.randint(0, self.args.item_size, size=n)).repeat(x_size).reshape(-1, 1)
            # all_a = torch.from_numpy(np.random.randint(0, self.args.item_size, size=(x_size, n))).reshape(-1, 1)
            # print(x.size(),all_a.size(), all_a)
            if self.args.learn_intervention:
                pred_rt = self.predict(x, all_a, tau)[0][:, 1].reshape(x_size, n)
            else:
                pred_rt = self.predict(x, all_a)[0][:, 1].reshape(x_size, n)
            # print(pred_rt, pred_rt.size())
            topk_rt_value, topk_rt = pred_rt.topk(k, dim=1, largest=True, sorted=True)
        return topk_rt


class SelectModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.name = 'Select_Model'
        self.args = args
        self.pair_wise_selection = args.pair_wise_selection
        self.x_dim = args.user_dim
        self.a_dim = args.item_dim
        self.x_size = args.user_size
        self.a_size = args.item_size

        self.x_emb_dim = args.user_emb_dim_s
        self.a_emb_dim = args.item_emb_dim_s
        # self.x_emb_dim = args.user_item_dim_r_s_model
        # self.a_emb_dim = args.user_item_dim_r_s_model

        self.layer_dim = args.select_layer_dims
        self.impression_length = self.args.impression_length
        self.topK = args.topK

        self.user_embedding_lookup = nn.Embedding(self.x_size, self.x_emb_dim)
        self.item_embedding_lookup = nn.Embedding(self.a_size, self.a_emb_dim)
        self.set_epsilon()
        self.flag = False

        self.user_net = nn.Sequential(
            nn.Linear(self.x_dim, self.layer_dim[0]),
            nn.ReLU(),
            nn.Linear(self.layer_dim[0], self.x_emb_dim),
        )

        self.item_net = nn.Sequential(
            nn.Linear(self.a_dim, self.layer_dim[1]),
            nn.ReLU(),
            nn.Linear(self.layer_dim[1], self.a_emb_dim)
        )

        # self.exogenous_net = nn.Linear(self.impression_length, self.impression_length)
        if self.flag:
            self.exogenous_net = torch.nn.Parameter(torch.FloatTensor(1))
        else:
            self.exogenous_net = torch.nn.Parameter(torch.FloatTensor(self.impression_length))

        self.pair_select_logits_net = nn.Sequential(
            nn.Linear(self.x_emb_dim + self.a_emb_dim, self.layer_dim[2]),
            nn.ReLU(),
            nn.Linear(self.layer_dim[2], 2)
        )
        if self.flag:  # a : batch_size
            self.select_logits_net = nn.Sequential(
                nn.Linear(self.x_emb_dim + self.a_emb_dim, self.layer_dim[2]),
                nn.ReLU(),
                nn.Linear(self.layer_dim[2], 1),
                nn.Sigmoid())
        elif self.pair_wise_selection:  # a : batch_size
            self.select_logits_net = nn.Sequential(
                nn.Linear(self.x_emb_dim + self.a_emb_dim, self.layer_dim[2]),
                nn.ReLU(),
                nn.Linear(self.layer_dim[2], 2))
        else:
            self.select_logits_net = nn.Sequential(
                nn.Linear(self.x_emb_dim + self.impression_length * self.a_emb_dim, self.layer_dim[2]),
                nn.ReLU(),
                nn.Linear(self.layer_dim[2], self.impression_length)
            )
        # self.user_embedding = nn.Sequential(
        # 	nn.Linear(x_dim, layer_dim[0])
        # )

        # self.item_embedding = nn.Sequential(
        # 	nn.Linear(a_dim*a_size, layer_dim[1]),
        # )
        self.sigmd = torch.nn.Sigmoid()
        if self.pair_wise_selection:
            self.sftcross = torch.nn.CrossEntropyLoss()
        else:
            self.sftcross = torch.nn.BCEWithLogitsLoss()
            self.bce = torch.nn.BCEWithLogitsLoss()

    def set_epsilon(self):
        #		if self.args.learn_epsilon:
        #			self.es_embedding_lookup = nn.Embedding(self.a_size, 2).to(device)
        #		else:
        # with torch.no_grad():
        es_init = torch.cat((torch.zeros(self.a_size, 1), torch.ones(self.a_size, 1)), dim=1).to(device)
        self.es_embedding_lookup = nn.Embedding.from_pretrained(es_init)
        assert es_init.size()[1] == 2

    def embedding(self, x, a):  #
        x = torch.tensor(x, dtype=torch.int64).to(device).reshape(x.size()[0], 1)
        if self.pair_wise_selection or self.flag:  # a : batch_size
            a = torch.tensor(a, dtype=torch.int64).to(device).reshape(a.size()[0], 1)
            item_emb = self.item_embedding_lookup(a).reshape(a.size()[0], self.a_emb_dim)
        else:  # a : batch_size * impression len
            # print(a)
            a = torch.tensor(a, dtype=torch.int64).to(device).reshape(a.size()[0], a.size()[1], 1)
            # print(a)
            item_emb = self.item_embedding_lookup(a).reshape(-1, a.size()[
                1] * self.a_emb_dim)  # batch , impression len * a embedding dim
        # print(item_emb)
        user_emb = self.user_embedding_lookup(x).reshape(x.size()[0], self.x_emb_dim)

        if self.args.pair_wise_selection:
            es_m_v = self.es_embedding_lookup(a).reshape(a.size()[0], 2)
            return user_emb, item_emb, es_m_v[:, 0], torch.abs(es_m_v[:, 1])
        else:
            es_m_v = self.es_embedding_lookup(a).reshape(a.size()[0], a.size()[1], 2)
            # print(es_m_v.size())
            return user_emb, item_emb, es_m_v[:, :, 0].reshape(-1), torch.abs(es_m_v[:, :, 1].reshape(-1))

    def predict(self, x, a, n=5):
        if self.flag:
            x_size = x.size()[0]
            # print(tools)
            # x: 1, a: impression size
            assert n == a.size()[1]
            x = x.reshape(x.size()[0], 1).repeat(1, n).reshape(-1, 1)
            # print(x)
            a = a.reshape(-1, 1)
        user_emb, item_emb, es_m, es_v = self.embedding(x, a)
        if self.flag:
            es = ut.sample_gaussian(es_m, es_v).reshape(a.size()[0], 1)
        else:
            es = ut.sample_gaussian(es_m, es_v).reshape(a.size()[0], self.impression_length)

        scores = self.select_logits_net(torch.cat((user_emb, item_emb), 1)) + self.exogenous_net * es

        # print('***'*10)
        # print('user_emb, item_emb, scores', user_emb.shape, item_emb.shape, scores.shape)
        # print('***'*10)
        return scores, es, es_m.reshape(a.size()[0], self.impression_length), es_v.reshape(a.size()[0],
                                                                                           self.impression_length)

    def evaluate(self, x, a, y, n=5):

        x = torch.tensor(x, dtype=torch.int64).to(device)
        a = torch.tensor(a, dtype=torch.int64).to(device)
        y = torch.tensor(y, dtype=torch.int32).to(device)

        y_score = self.predict(x, a)[0].reshape(x.size()[0], n)
        y_pred = torch.sigmoid(y_score)
        # print('Predictions during evaluation:')
        # print(y_pred[:100])
        y_pred[y_pred >= 0.5] = 1
        y_pred = y_pred.int()
        res = (y_pred == y)
        acc = (y_pred == y).sum().float() / (y.size()[0] * y.size()[1])

        return acc.detach().cpu().numpy(), y_score.detach().cpu().numpy(), y.detach().cpu().numpy()

    def loss(self, x, a, y):
        assert not self.args.learn_epsilon
        y = torch.tensor(y, dtype=torch.int64).to(device)
        if not self.pair_wise_selection:
            y = torch.tensor(y, dtype=torch.float32).to(device)

        y_hat, _, _, _ = self.predict(x, a)
        # print(y_hat, y)
        return self.sftcross(y_hat,
                             y)  # +torch.mean(torch.norm(user_emb, p=2, dim=1))+torch.mean(torch.norm(item_emb, p=2, dim=1))

    def bce_loss(self, x, a, y, n=5):
        assert not self.args.learn_epsilon
        y = torch.tensor(y, dtype=torch.int64).to(device)
        if not self.pair_wise_selection:
            y = torch.tensor(y, dtype=torch.float32).to(device)

        y_hat = self.predict(x, a)[0].reshape(x.size()[0], n)
        return self.bce(torch.sigmoid(y_hat), y)

    def epsilon_loss(self, x, a, y, n=5):  # ELBO
        assert self.args.learn_epsilon
        if self.pair_wise_selection:  # binary classification
            y = torch.tensor(y, dtype=torch.int64).to(device)
            y_hat, es, es_m, es_v = self.predict(x, a)
            q_es = torch.mean(torch.log(1.0 / torch.sqrt(es_v)))

            return -self.sftcross(y_hat, y) + torch.mean(torch.norm(es, p=2, dim=1)) - q_es + torch.mean(
                torch.norm(es, p=2, dim=1))
        else:  # multi-label classification
            y = torch.tensor(y, dtype=torch.float32).to(device)
            y_hat, es, es_m, es_v = self.predict(x, a)
            # q_es = torch.mean(torch.log(1.0/torch.sqrt(es_v)))
            # print(es.size(), es_m.size(), es_v.size())
            q_es = torch.mean(
                torch.log(1.0 / (es_v * torch.sqrt(2 * torch.tensor(math.pi, dtype=torch.float32)))) - (es - es_m).pow(
                    2) / es_v.pow(2))
            if self.flag:
                y_hat = y_hat.reshape(x.size()[0], n)
            # print(y_hat.size(), y.size(), y_hat, y)
            # --------------------------------------------
            # return -q_es#-self.sftcross(y_hat, y)+torch.mean(torch.norm(es, p=2, dim=1))
            return -self.sftcross(y_hat, y) + torch.mean(
                torch.norm(es, p=2, dim=1)) - q_es  # +torch.mean(torch.norm(item_emb, p=2, dim=1))

    def generate_st(self, x, a, n):
        '''
        x: [x_size]
        a: [x_size, n]
        '''
        # print(x.size())
        x_size = x.size()[0]
        tools = torch.from_numpy(numpy.arange(0, x_size, 1)).to(device)
        tools = tools.repeat(self.topK, 1).t().reshape(-1)
        # print(tools)
        if self.pair_wise_selection or self.flag:  # binary classification
            # x: 1, a: impression size
            # assert n == a.size()[1]
            # x = x.reshape(x.size()[0],1).repeat(1,n).reshape(-1,1)

            if self.flag:
                pred_st = self.predict(x, a)[0].reshape(x_size, n)
            else:
                pred_st = self.predict(x, a)[0][:, 1].reshape(x_size, n)
            _, p = pred_st.topk(self.topK, dim=1, largest=True, sorted=True)
            _, np = pred_st.topk(self.topK, dim=1, largest=False, sorted=True)
            # print(p,np)
            p, np = n * tools + p.reshape(-1), n * tools + np.reshape(-1)
            # print(n*tools)
            # print(p,np)
            a = a.reshape(-1, 1)
            prefer = torch.index_select(a, 0, p)
            # print(prefer.size(), k,'................................')
            not_prefer = torch.index_select(a, 0, np)

        else:  # multi-label classification
            assert n == a.size()[1]
            pred_st = self.predict(x, a)[0].reshape(x_size, n)
            _, p = pred_st.topk(self.topK, dim=1, largest=True, sorted=True)
            _, np = pred_st.topk(self.topK, dim=1, largest=False, sorted=True)
            # print(p, np)
            p, np = n * tools + p.reshape(-1), n * tools + np.reshape(-1)
            # print(n*tools)
            # print(a.reshape(-1,1).size())
            prefer = torch.index_select(a.reshape(-1, 1), 0, p)
            not_prefer = torch.index_select(a.reshape(-1, 1), 0, np)

        # print('prefer', prefer.size())
        # print('not_prefer', not_prefer.size())

        return prefer, not_prefer


class BPR(nn.Module):
    def __init__(self, args, weight_decay=0.00001):
        super().__init__()
        self.name = 'BPR'
        self.args = args
        self.W = nn.Parameter(torch.empty(self.args.user_size, self.args.user_emb_dim))
        self.H = nn.Parameter(torch.empty(self.args.item_size, self.args.item_emb_dim))
        nn.init.xavier_normal_(self.W.data)
        nn.init.xavier_normal_(self.H.data)
        self.weight_decay = self.args.weight_decay

    def forward(self, u, i, j):
        """Return loss value.

        Args:
            u(torch.LongTensor): tensor stored user indexes. [batch_size,]
            i(torch.LongTensor): tensor stored item indexes which is prefered by user. [batch_size,]
            j(torch.LongTensor): tensor stored item indexes which is not prefered by user. [batch_size,]

        Returns:
            torch.FloatTensor
        """
        # print(u.size(), i.size())
        u = torch.tensor(u, dtype=torch.int64).to(device).reshape(u.size()[0], 1)
        i = torch.tensor(i, dtype=torch.int64).to(device).reshape(i.size()[0], 1)
        j = torch.tensor(j, dtype=torch.int64).to(device).reshape(j.size()[0], 1)

        # print('u', u)
        # print('i', i)
        # print('j', j)

        u = self.W[u, :]
        i = self.H[i, :]
        j = self.H[j, :]

        # print('u_1', u)
        # print('i_1', i)
        # print('j_1', j)

        x_ui = torch.mul(u, i).sum(dim=1)
        x_uj = torch.mul(u, j).sum(dim=1)
        x_uij = x_ui - x_uj
        log_prob = F.logsigmoid(x_uij).mean(1)
        regularization = self.weight_decay * (
                    u.norm(dim=1).pow(2).mean(1) + i.norm(dim=1).pow(2).mean(1) + j.norm(dim=1).pow(2).mean(1))
        return -log_prob + regularization

    def predict(self, u, i, mode='test'):
        # print('W', self.W[:100, :])
        # print('H', self.H[:100, :])
        # print('u', u)
        # print('i', i)
        u = torch.tensor(u, dtype=torch.int64).to(device_cpu).reshape(u.shape[0], 1)
        i = torch.tensor(i, dtype=torch.int64).to(device_cpu).reshape(i.shape[0], 1)
        # print('u_2', u)
        # print('i_2', i)
        # print(u.size(), i.size())
        u = self.W[u, :]
        i = self.H[i, :]
        u = u.reshape(u.size()[0], u.size()[2])
        i = i.reshape(i.size()[0], i.size()[2])
        # print(u.size()[0], u.size()[1])
        # print(u.size(), i.size(), (u*i).sum(dim=1).size())
        pred_rating = (u * i).sum(dim=1).detach().cpu().numpy()

        # print('u_e', u)
        # print('i_e', i)
        # if mode == 'test':
        # 	print('pred', pred_rating)

        return pred_rating


class BPR_NEW(nn.Module):
    def __init__(self, args, weight_decay=0.00001):
        super().__init__()
        self.name = 'BPR'
        self.args = args
        self.W = nn.Parameter(torch.empty(self.args.user_size, self.args.user_emb_dim))
        self.H = nn.Parameter(torch.empty(self.args.item_size, self.args.item_emb_dim))
        nn.init.xavier_normal_(self.W.data)
        nn.init.xavier_normal_(self.H.data)
        self.weight_decay = self.args.weight_decay

    def forward(self, u, i, j, mode='others'):
        """Return loss value.

        Args:
            u(torch.LongTensor): tensor stored user indexes. [batch_size,]
            i(torch.LongTensor): tensor stored item indexes which is prefered by user. [batch_size,]
            j(torch.LongTensor): tensor stored item indexes which is not prefered by user. [batch_size,]

        Returns:
            torch.FloatTensor
        """
        u = torch.tensor(u, dtype=torch.int64).to(device).reshape(u.size()[0], 1)
        i = torch.tensor(i, dtype=torch.int64).to(device).reshape(i.size()[0], 1)
        j = torch.tensor(j, dtype=torch.int64).to(device).reshape(j.size()[0], 1)

        u = self.W[u, :].reshape(u.size()[0], self.args.user_emb_dim)
        i = self.H[i, :].reshape(i.size()[0], self.args.item_emb_dim)
        j = self.H[j, :].reshape(j.size()[0], self.args.item_emb_dim)


        x_ui = torch.mul(u, i).sum(dim=1)
        x_uj = torch.mul(u, j).sum(dim=1)
        x_uij = x_ui - x_uj

        log_prob = F.logsigmoid(x_uij)
        regularization = self.weight_decay * (u.norm(dim=1) + i.norm(dim=1) + j.norm(dim=1))

        return -log_prob + regularization

    def predict(self, u, i, mode='test'):
        u = torch.tensor(u, dtype=torch.int64).to(device_cpu).reshape(u.shape[0], 1)
        i = torch.tensor(i, dtype=torch.int64).to(device_cpu).reshape(i.shape[0], 1)
        u = self.W[u, :]
        i = self.H[i, :]
        u = u.reshape(u.size()[0], u.size()[2])
        i = i.reshape(i.size()[0], i.size()[2])
        pred_rating = (u * i).sum(dim=1).detach().cpu().numpy()



        return pred_rating


    def recommend(self, u):
        """Return recommended item list given users.
        Args:
            u(torch.LongTensor): tensor stored user indexes. [batch_size,]
        Returns:
            pred(torch.LongTensor): recommended item list sorted by preference. [batch_size, item_size]
        """
        u = self.W[u, :]
        x_ui = torch.mm(u, self.H.t())
        pred = torch.argsort(x_ui, dim=1)
        return pred


class NeuBPR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.name = args.model_name
        self.args = args
        self.layers = [int(l) for l in args.layers.split('|')]
        # self.layers = args.layers

        self.W_mlp = torch.nn.Embedding(num_embeddings=args.user_size, embedding_dim=args.user_emb_dim)
        self.H_mlp = torch.nn.Embedding(num_embeddings=args.item_size, embedding_dim=args.item_emb_dim)
        self.W_mf = torch.nn.Embedding(num_embeddings=args.user_size, embedding_dim=args.user_emb_dim)
        self.H_mf = torch.nn.Embedding(num_embeddings=args.item_size, embedding_dim=args.item_emb_dim)

        nn.init.xavier_normal_(self.W_mlp.weight.data)
        nn.init.xavier_normal_(self.H_mlp.weight.data)
        nn.init.xavier_normal_(self.W_mf.weight.data)
        nn.init.xavier_normal_(self.H_mf.weight.data)

        if self.name == 'NeuBPR':
            self.fc_layers = torch.nn.ModuleList()
            for idx, (in_size, out_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
                self.fc_layers.append(torch.nn.Linear(in_size, out_size))

            self.affine_output = torch.nn.Linear(in_features=self.layers[-1] + args.user_emb_dim, out_features=1)

        elif self.name == 'gmfBPR':
            self.affine_output = torch.nn.Linear(in_features=args.user_emb_dim, out_features=1)

        elif self.name == 'mlpBPR':
            self.fc_layers = torch.nn.ModuleList()
            for idx, (in_size, out_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
                self.fc_layers.append(torch.nn.Linear(in_size, out_size))

            self.affine_output = torch.nn.Linear(in_features=self.layers[-1], out_features=1)

        self.logistic = torch.nn.Sigmoid()
        self.weight_decay = args.weight_decay
        self.dropout = torch.nn.Dropout(p=args.dropout)

    def forward(self, u, i, j):
        """Return loss value.

        Args:
            u(torch.LongTensor): tensor stored user indexes. [batch_size,]
            i(torch.LongTensor): tensor stored item indexes which is prefered by user. [batch_size,]
            j(torch.LongTensor): tensor stored item indexes which is not prefered by user. [batch_size,]

        Returns:
            torch.FloatTensor
        """
        # print(u.size(), i.size())

        u = torch.tensor(u, dtype=torch.int64).to(device).reshape(u.size()[0], 1)
        i = torch.tensor(i, dtype=torch.int64).to(device).reshape(i.size()[0], 1)
        j = torch.tensor(j, dtype=torch.int64).to(device).reshape(j.size()[0], 1)

        x_ui = self.predict(u, i, mode='train')
        x_uj = self.predict(u, j, mode='train')
        x_uij = x_ui - x_uj
        # # log_prob = F.logsigmoid(x_uij).mean()
        # log_prob = F.logsigmoid(x_uij)

        Wu_mlp = self.W_mlp(u).reshape(u.size()[0], self.args.user_emb_dim)
        Wu_mf = self.W_mf(u).reshape(u.size()[0], self.args.user_emb_dim)

        Hi_mlp = self.H_mlp(i).reshape(i.size()[0], self.args.item_emb_dim)
        Hi_mf = self.H_mf(i).reshape(i.size()[0], self.args.item_emb_dim)

        Hj_mlp = self.H_mlp(j).reshape(j.size()[0], self.args.item_emb_dim)
        Hj_mf = self.H_mf(j).reshape(j.size()[0], self.args.item_emb_dim)



        # log_prob = F.logsigmoid(x_uij).mean()

        # if self.args.model_name == 'NeuBPR':
        # 	regularization = self.weight_decay * (Wu_mlp.norm(dim=1).pow(2).mean() + \
        # 		Wu_mf.norm(dim=1).pow(2).mean() + Hi_mlp.norm(dim=1).pow(2).mean() + \
        # 		Hi_mf.norm(dim=1).pow(2).mean() + Hj_mlp.norm(dim=1).pow(2).mean() + \
        # 		Hj_mf.norm(dim=1).pow(2).mean())
        # elif self.args.model_name in ['gmfBPR', 'bprBPR']:
        # 	regularization = self.weight_decay * (Wu_mf.norm(dim=1).pow(2).mean() + \
        # 		Hi_mf.norm(dim=1).pow(2).mean() + Hj_mf.norm(dim=1).pow(2).mean())
        # elif self.args.model_name == 'mlpBPR':
        # 	regularization = self.weight_decay * (Wu_mlp.norm(dim=1).pow(2).mean() + \
        # 		Hi_mlp.norm(dim=1).pow(2).mean() + Hj_mlp.norm(dim=1).pow(2).mean())

        log_prob = F.logsigmoid(x_uij)

        if self.args.model_name == 'NeuBPR':
            regularization = self.weight_decay * (Wu_mlp.norm(dim=1) + \
                                                  Wu_mf.norm(dim=1) + Hi_mlp.norm(dim=1) + \
                                                  Hi_mf.norm(dim=1) + Hj_mlp.norm(dim=1) + \
                                                  Hj_mf.norm(dim=1))
        elif self.args.model_name in ['gmfBPR', 'bprBPR']:
            regularization = self.weight_decay * (Wu_mf.norm(dim=1) + \
                                                  Hi_mf.norm(dim=1) + Hj_mf.norm(dim=1))
        elif self.args.model_name == 'mlpBPR':
            regularization = self.weight_decay * (Wu_mlp.norm(dim=1) + \
                                                  Hi_mlp.norm(dim=1) + Hj_mlp.norm(dim=1))
        # ------------------------------------------------------------------------
        return -log_prob + regularization

    def predict(self, u, i, mode='test'):

        if mode == 'test':
            u = torch.tensor(u, dtype=torch.int64).to(device_cpu).reshape(u.shape[0], 1)
            i = torch.tensor(i, dtype=torch.int64).to(device_cpu).reshape(i.shape[0], 1)

        if self.args.model_name == 'NeuBPR':
            user_embedding_mlp = self.W_mlp(u).reshape(u.size()[0], self.args.user_emb_dim)
            item_embedding_mlp = self.H_mlp(i).reshape(i.size()[0], self.args.item_emb_dim)
            user_embedding_mf = self.W_mf(u).reshape(u.size()[0], self.args.user_emb_dim)
            item_embedding_mf = self.H_mf(i).reshape(i.size()[0], self.args.item_emb_dim)

            mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # the concat latent vector
            mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)

            for idx, _ in enumerate(range(len(self.fc_layers))):
                mlp_vector = self.fc_layers[idx](mlp_vector)
                mlp_vector = torch.nn.ReLU()(mlp_vector)

            vector = torch.cat([mlp_vector, mf_vector], dim=-1)

        elif self.args.model_name == 'gmfBPR':
            user_embedding_mf = self.W_mf(u).reshape(u.size()[0], self.args.user_emb_dim)
            item_embedding_mf = self.H_mf(i).reshape(i.size()[0], self.args.item_emb_dim)

            vector = torch.mul(user_embedding_mf, item_embedding_mf)

        elif self.args.model_name == 'bprBPR':
            user_embedding_mf = self.W_mf(u).reshape(u.size()[0], self.args.user_emb_dim)
            item_embedding_mf = self.H_mf(i).reshape(i.size()[0], self.args.item_emb_dim)

            vector = torch.mul(user_embedding_mf, item_embedding_mf)

        elif self.args.model_name == 'mlpBPR':
            user_embedding_mlp = self.W_mlp(u).reshape(u.size()[0], self.args.user_emb_dim)
            item_embedding_mlp = self.H_mlp(i).reshape(i.size()[0], self.args.item_emb_dim)

            vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # the concat latent vector

            for idx, _ in enumerate(range(len(self.fc_layers))):
                vector = self.fc_layers[idx](vector)
                vector = torch.nn.ReLU()(vector)
                vector = self.dropout(vector)

        # print('###'*10)
        # print('user_emb, item_emb, vector', user_embedding_mf.size(), item_embedding_mf.size(), vector.size())
        # print('###'*10)

        if self.args.model_name in ['NeuBPR', 'gmfBPR', 'mlpBPR']:
            logits = self.affine_output(vector)
            rating = logits.reshape(logits.size()[0])
        elif self.args.model_name == 'bprBPR':
            rating = vector.sum(dim=1)
            rating = rating.reshape(rating.size()[0])

        if mode == 'test':
            # rating = self.logistic(rating)
            rating = rating.detach().cpu().numpy()

        # print('rating', rating.shape, rating)

        return rating


    def load_pretrain_weights(self, gmf_model, mlp_model):
        """Loading weights from trained MLP model & GMF model for NeuBPR"""

        self.W_mlp.weight.data = mlp_model.W_mlp.weight.data
        self.H_mlp.weight.data = mlp_model.H_mlp.weight.data
        for idx in range(len(self.fc_layers)):
            self.fc_layers[idx].weight.data = mlp_model.fc_layers[idx].weight.data

        self.W_mf.weight.data = gmf_model.W_mf.weight.data
        self.H_mf.weight.data = gmf_model.H_mf.weight.data

        self.affine_output.weight.data = 0.5 * torch.cat(
            [mlp_model.affine_output.weight.data, gmf_model.affine_output.weight.data], dim=-1)
        self.affine_output.bias.data = 0.5 * (mlp_model.affine_output.bias.data + gmf_model.affine_output.bias.data)


# -------------------------------------------------------------
# LightGCN
class LightGCN(nn.Module):
    def __init__(self, args, train_matrix, device, device_cpu):
        super().__init__()
        self.name = args.model_name
        self.datasets = args.datasets
        self.data_name = args.data_name
        self.num_users = args.user_size
        self.num_items = args.item_size
        self.train_matrix = train_matrix

        self.emb_dim = args.user_emb_dim
        self.num_layers = args.num_layers
        self.node_dropout = args.node_dropout

        # self.num_negatives = model_conf['num_negatives']
        self.split = args.split
        self.num_folds = args.num_folds

        self.reg = args.reg
        self.batch_size = args.batch_size

        self.Graph = None
        self.data_loader = None
        self.path = args.graph_dir
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        self.device = device
        self.device_cpu = device_cpu

        self.build_graph()

    def build_graph(self):
        # Variable
        # torch.ones(self.num_users, self.emb_dim)
        self.user_embedding = nn.Embedding(self.num_users, self.emb_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.emb_dim)
        nn.init.normal_(self.user_embedding.weight, 0, 0.01)
        nn.init.normal_(self.item_embedding.weight, 0, 0.01)

        self.user_embedding_pred = None
        self.item_embedding_pred = None

        if self.Graph == None:
            self.Graph = self.getSparseGraph(self.train_matrix)
        if self.data_loader == None:
            self.data_loader = PairwiseGenerator(self.train_matrix, num_negatives=5, batch_size=self.batch_size,
                                                 shuffle=True, device=self.device)

        self.to(self.device)

    def forward(self, user, pos, neg):

        user = torch.tensor(user, dtype=torch.long).reshape(user.shape[0]).to(device)
        pos = torch.tensor(pos, dtype=torch.long).reshape(pos.shape[0]).to(device)
        neg = torch.tensor(neg, dtype=torch.long).reshape(neg.shape[0]).to(device)

        u_embedding, i_embedding = self.lightgcn_embedding(self.Graph)

        user_latent = F.embedding(user, u_embedding)
        pos_latent = F.embedding(pos, i_embedding)
        neg_latent = F.embedding(neg, i_embedding)

        pos_score = torch.mul(user_latent, pos_latent).sum(1)
        neg_score = torch.mul(user_latent, neg_latent).sum(1)

        userEmb0 = self.user_embedding(user)
        posEmb0 = self.item_embedding(pos)
        negEmb0 = self.item_embedding(neg)

        batch_loss = F.softplus(neg_score - pos_score)
        reg_loss = self.reg * (userEmb0.norm(1) + posEmb0.norm(1) + negEmb0.norm(1))

        batch_loss = batch_loss + reg_loss

        return batch_loss

    def forward_inside(self, user, pos, neg=None):
        u_embedding, i_embedding = self.lightgcn_embedding(self.Graph)

        user_latent = F.embedding(user, u_embedding)
        pos_latent = F.embedding(pos, i_embedding)

        pos_score = torch.mul(user_latent, pos_latent).sum(1)
        if neg is not None:
            neg_latent = F.embedding(neg, i_embedding)
            neg_score = torch.mul(user_latent, neg_latent).sum(1)
            return pos_score, neg_score
        else:
            return pos_score

    def predict(self, u, i, mode='test'):
        u = torch.tensor(u, dtype=torch.long, device=self.device_cpu)
        i = torch.tensor(i, dtype=torch.long, device=self.device_cpu)
        scores = self.forward_inside(u, i)
        scores = scores.reshape(scores.size()[0]).detach().cpu().numpy()
        return scores

    def train_one_epoch(self, optimizer, batch_size, verbose):
        loss = 0.0
        for b, batch_data in enumerate(self.data_loader):
            optimizer.zero_grad()
            batch_user, batch_pos, batch_neg = batch_data

            pos_output, neg_output = self.forward_inside(batch_user, batch_pos, batch_neg)
            userEmb0 = self.user_embedding(batch_user)
            posEmb0 = self.item_embedding(batch_pos)
            negEmb0 = self.item_embedding(batch_neg)

            batch_loss = torch.mean(F.softplus(neg_output - pos_output))
            reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2)) / float(
                len(batch_user))

            batch_loss = batch_loss + self.reg * reg_loss

            batch_loss.backward()
            optimizer.step()

            loss += batch_loss

        return loss

    def predict_batch_users(self, user_ids):
        user_embeddings = F.embedding(user_ids, self.user_embedding_pred)
        item_embeddings = self.item_embedding_pred
        return user_embeddings @ item_embeddings.T

    def before_evaluate(self):
        self.user_embedding_pred, self.item_embedding_pred = self.lightgcn_embedding(self.Graph)

    ##################################### LightGCN Code
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def lightgcn_embedding(self, graph):
        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight
        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]
        if self.node_dropout > 0:
            if self.training:
                g_droped = self.__dropout(graph, self.node_dropout)
            else:
                g_droped = graph
        else:
            g_droped = graph

        for layer in range(self.num_layers):
            if self.split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def make_train_matrix(self):
        train_matrix_arr = self.dataset.train_matrix.toarray()
        self.train_matrix = sp.csr_matrix(train_matrix_arr)

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.num_users + self.num_items) // self.num_folds
        for i_fold in range(self.num_folds):
            start = i_fold * fold_len
            if i_fold == self.num_folds - 1:
                end = self.num_users + self.num_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(self.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self, rating_matrix):
        n_users, n_items = rating_matrix.shape
        print("loading adjacency matrix")

        filename = f'{self.data_name}_{self.datasets}_s_pre_adj_mat.npz'
        try:
            print(os.path.join(self.path, filename))
            pre_adj_mat = sp.load_npz(os.path.join(self.path, filename))
            print("successfully loaded...")
            norm_adj = pre_adj_mat
        except:
            print("generating adjacency matrix")
            s = time.time()
            adj_mat = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = rating_matrix.tolil()
            adj_mat[:n_users, n_users:] = R
            adj_mat[n_users:, :n_users] = R.T
            adj_mat = adj_mat.todok()

            # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)

            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()
            end = time.time()
            print(f"costing {end - s}s, saved norm_mat...")
            sp.save_npz(os.path.join(self.path, filename), norm_adj)

        if self.split == True:
            Graph = self._split_A_hat(norm_adj)
            print("done split matrix")
        else:
            Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            Graph = Graph.coalesce().to(self.device)
            print("don't split the matrix")
        return Graph


################ CUSTOM SAMPLER
class PairwiseGenerator:
    def __init__(self, input_matrix, num_negatives=1, batch_size=32, shuffle=True, device=None):
        super().__init__()
        self.input_matrix = input_matrix
        self.num_negatives = num_negatives
        self.num_users, self.num_items = input_matrix.shape

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

        self._construct()

    def _construct(self):
        self.pos_dict = {}
        for u in range(self.num_users):
            u_items = self.input_matrix[u].indices

            self.pos_dict[u] = u_items.tolist()

    def __len__(self):
        return int(np.ceil(self.num_users / self.batch_size))

    def __iter__(self):
        if self.shuffle:
            perm = np.random.permutation(self.num_users)
        else:
            perm = np.arange(self.num_users)

        for b, st in enumerate(range(0, len(perm), self.batch_size)):
            batch_users = []
            batch_pos = []
            batch_neg = []

            ed = min(st + self.batch_size, len(perm))
            users = perm[st:ed]
            for i, u in enumerate(users):

                posForUser = self.pos_dict[u]
                if len(posForUser) == 0:
                    continue
                posindex = np.random.randint(0, len(posForUser))
                positem = posForUser[posindex]

                for i in range(self.num_negatives):
                    while True:
                        negitem = np.random.randint(0, self.num_items)
                        if negitem in posForUser:
                            continue
                        else:
                            break
                    batch_users.append(u)
                    batch_pos.append(positem)
                    batch_neg.append(negitem)

            batch_users = torch.tensor(batch_users, dtype=torch.long, device=self.device)
            batch_pos = torch.tensor(batch_pos, dtype=torch.long, device=self.device)
            batch_neg = torch.tensor(batch_neg, dtype=torch.long, device=self.device)
            yield batch_users, batch_pos, batch_neg

# LightGCN
# -------------------------------------------------------------