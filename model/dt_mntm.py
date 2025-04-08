# python3.6                                
# encoding    : utf-8 -*-                            
# @author     : YingqiuXiong
# @e-mail     : 1916728303@qq.com                                    
# @file       : dt_mntm.py
# @Time       : 2023/11/14 21:22
# Decoupling topics for multimodal neural topic models
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F, init
import numpy as np



# 初始化topic_vec(K*L) 和 word_vec(V*L) 两个矩阵
def embedding_init(dim1, dim2, constant=1):
    low = -constant*np.sqrt(6.0/(dim1 + dim2))
    high = constant*np.sqrt(6.0/(dim1 + dim2))
    return torch.Tensor(size=(dim1, dim2)).uniform_(low, high)


class EncoderNetworkFromLamo(nn.Module):
    def __init__(self, input_size, hidden_sizes, zPrivate_dim=40, zShared_dim=10,
                 activation='softplus', dropout=0.2, seed=888):
        super(EncoderNetworkFromLamo, self).__init__()
        assert isinstance(input_size, int), "input_size must by type int."
        assert isinstance(hidden_sizes, tuple), "hidden_layers must be type tuple."
        assert isinstance(zShared_dim, int), "zShared_dim must be type int."
        assert isinstance(zPrivate_dim, int), "zPrivate_dim must be type int."
        assert activation in ['softplus', 'relu'], "activation must be 'softplus' or 'relu'."
        assert dropout >= 0, "dropout must be >= 0."

        self.seed = seed  # 随机数种子
        self.EPS = 1e-9
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.zShared_dim = zShared_dim
        self.zPrivate_dim = zPrivate_dim
        self.dropout = dropout

        if activation == 'softplus':
            self.activation = nn.Softplus()  # f(x)=log(1+exp(x))
        elif activation == 'relu':
            self.activation = nn.ReLU()

        self.input_layer = nn.Linear(input_size, hidden_sizes[0])  # 推理网络的输入层
        self.hidden_layers = nn.Sequential(OrderedDict([
            ('l_{}'.format(i), nn.Sequential(nn.Linear(h_in, h_out), self.activation))
            for i, (h_in, h_out) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:]))]))  # 隐藏层

        self.dropout_enc = nn.Dropout(p=self.dropout)

        self.fc_mu_private = nn.Linear(hidden_sizes[-1], zPrivate_dim)
        self.mu_private_batchnorm = nn.BatchNorm1d(zPrivate_dim, affine=False)
        self.fc_log_sigma_private = nn.Linear(hidden_sizes[-1], zPrivate_dim)
        self.log_sigma_private_batchnorm = nn.BatchNorm1d(zPrivate_dim, affine=False)
        self.fc_mu_shared = nn.Linear(hidden_sizes[-1], zShared_dim)
        self.mu_shared_batchnorm = nn.BatchNorm1d(zShared_dim, affine=False)
        self.fc_log_sigma_shared = nn.Linear(hidden_sizes[-1], zShared_dim)
        self.log_sigma_shared_batchnorm = nn.BatchNorm1d(zShared_dim, affine=False)

        self.weight_init()

    def weight_init(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        for module in self._modules:
            if isinstance(module, nn.Sequential):
                for one_module in module:
                    init.kaiming_uniform_(one_module.weight)
            else:
                try:
                    init.kaiming_uniform_(module.weight)
                except AttributeError as e:
                    pass

    def forward(self, bow):
        x = self.input_layer(bow)
        x = self.activation(x)
        x = self.hidden_layers(x)
        x = self.dropout_enc(x)
        # 私有分布的参数
        muPrivate = self.mu_private_batchnorm(self.fc_mu_private(x))
        logStdPrivate = self.log_sigma_private_batchnorm(self.fc_log_sigma_private(x))
        # muPrivate = self.fc_mu_private(x)
        # logStdPrivate = self.fc_log_sigma_private(x)
        stdPrivate = torch.exp(logStdPrivate)
        # 共有表征
        muShared = self.mu_shared_batchnorm(self.fc_mu_shared(x))
        logStdShared = self.log_sigma_shared_batchnorm(self.fc_log_sigma_shared(x))
        # muShared = self.fc_mu_shared(x)
        # logStdShared = self.fc_log_sigma_shared(x)
        stdShared = torch.exp(logStdShared)
        # 私有的和共有的正太分布的参数
        return muPrivate, stdPrivate, muShared, stdShared


class DecoderNetworkFromLamo(nn.Module):
    def __init__(self, input_size, zPrivate_dim=15, zShared_dim=5, dropout=0.2, learn_priors=False, tm_type="prodLDA"):
        super(DecoderNetworkFromLamo, self).__init__()
        assert isinstance(input_size, int), "input_size must by type int."
        assert isinstance(zPrivate_dim, int) and zPrivate_dim > 0, "n_components must be type int > 0."
        assert isinstance(zShared_dim, int) and zShared_dim > 0, "n_components must be type int > 0."
        assert dropout >= 0, "dropout must be >= 0."
        assert tm_type in ['prodLDA', 'LDA', "GSM"], "model type must be 'prodLDA' or 'LDA'"
        assert isinstance(learn_priors, bool), "learn_priors must be type bool."

        self.input_size = input_size
        self.zPrivate_dim = zPrivate_dim
        self.zShared_dim = zShared_dim
        self.dropout = dropout
        self.learn_priors = learn_priors
        self.tm_type = tm_type
        self.topic_word_matrix = None  # 主题-词分布

        # init prior parameters
        z_prior_mean = 0.0
        self.prior_mean_private = torch.tensor([z_prior_mean] * zPrivate_dim)
        self.prior_mean_shared = torch.tensor([z_prior_mean] * zShared_dim)
        if torch.cuda.is_available():
            self.prior_mean_private = self.prior_mean_private.cuda()
            self.prior_mean_shared = self.prior_mean_shared.cuda()
        if self.learn_priors:
            self.prior_mean_private = nn.Parameter(self.prior_mean_private)
            self.prior_mean_shared = nn.Parameter(self.prior_mean_shared)

        zPrivate_prior_variance = 1. - (1. / self.zPrivate_dim)
        zShared_prior_variance = 1. - (1. / self.zShared_dim)
        self.prior_variance_private = torch.tensor([zPrivate_prior_variance] * zPrivate_dim)
        self.prior_variance_shared = torch.tensor([zShared_prior_variance] * zShared_dim)
        if torch.cuda.is_available():
            self.prior_variance_private = self.prior_variance_private.cuda()
            self.prior_variance_shared = self.prior_variance_shared.cuda()
        if self.learn_priors:
            self.prior_variance_private = nn.Parameter(self.prior_variance_private)
            self.prior_variance_shared = nn.Parameter(self.prior_variance_shared)

        if tm_type == "GSM":
            self.topic_embedding = nn.Parameter(embedding_init(zPrivate_dim + zShared_dim, 100))
            self.word_embedding = nn.Parameter(embedding_init(self.input_size, 100))
        else:
            self.beta = nn.Parameter(torch.Tensor((self.zPrivate_dim+self.zShared_dim), input_size).cuda())  # 主题-词分布
            nn.init.xavier_uniform_(self.beta)

            self.beta_batchnorm = nn.BatchNorm1d(input_size, affine=False)
            self.drop_theta = nn.Dropout(p=self.dropout)

    def reparameterize(self, mu, std):
        """Reparameterize the theta distribution."""
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, zPrivate, zShared):
        # 输出重构的词分布
        theta = F.softmax(torch.cat([zPrivate, zShared], -1), dim=1)  # 主题分布

        if self.tm_type == 'prodLDA':
            theta = self.drop_theta(theta)
            reconstruct_word_dist = F.softmax(torch.matmul(theta, self.beta), dim=1)
            # reconstruct_word_dist = F.softmax(self.beta_batchnorm(torch.matmul(theta, self.beta)), dim=1)
            self.topic_word_matrix = self.beta
        elif self.tm_type == 'LDA':
            # theta = self.drop_theta(theta)
            beta = F.softmax(self.beta_batchnorm(self.beta), dim=1)
            # beta = F.softmax(self.beta, dim=1)
            self.topic_word_matrix = beta
            reconstruct_word_dist = torch.matmul(theta, beta)
        elif self.tm_type == 'GSM':
            self.beta = torch.softmax(torch.matmul(self.topic_embedding,
                                                   torch.transpose(self.word_embedding, 0, 1)), dim=1)
            self.topic_word_matrix = self.beta
            reconstruct_word_dist = torch.matmul(theta, self.beta)  # p_x_out的每一维就是log(p(wi|theta))
        else:
            raise NotImplementedError("Model Type Not Implemented")

        return (self.prior_mean_private, self.prior_variance_private, self.prior_mean_shared,
                self.prior_variance_shared, reconstruct_word_dist)

    def get_theta(self, mu_private, std_private, mu_shared, std_shared):
        with torch.no_grad():
            # 输出重构的词分布
            zPrivate = self.reparameterize(mu_private, std_private)
            zShared = self.reparameterize(mu_shared, std_shared)
            theta = F.softmax(torch.cat([zPrivate, zShared], -1), dim=1)  # 主题分布

            return theta


class EncoderNetworkFromGsm(nn.Module):
    def __init__(self, input_size, hidden_size, zPrivate_dim=40, zShared_dim=10,
                 activation='relu', seed=888):
        super(EncoderNetworkFromGsm, self).__init__()
        self.seed = seed  # 随机数种子
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.zShared_dim = zShared_dim
        self.zPrivate_dim = zPrivate_dim

        if activation == 'softplus':
            self.activation = nn.Softplus()  # f(x)=log(1+exp(x))
        elif activation == 'relu':
            self.activation = nn.ReLU()

        self.hidden_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            self.activation,
            nn.Linear(hidden_size, hidden_size),
            self.activation,
        )
        # neural networks with dropout to reduce overfitting.
        self.dropout_enc = nn.Dropout(0.2)

        self.fc_mu_private = nn.Linear(hidden_size, zPrivate_dim)
        self.fc_log_sigma_private = nn.Linear(hidden_size, zPrivate_dim)
        self.fc_mu_shared = nn.Linear(hidden_size, zShared_dim)
        self.fc_log_sigma_shared = nn.Linear(hidden_size, zShared_dim)

        self.weight_init()  # 初始化参数

    def weight_init(self):
        for module in self._modules:
            if isinstance(module, nn.Sequential):
                for one_module in module:
                    init.kaiming_uniform_(one_module.weight, nonlinearity='relu')
                    init.zeros_(one_module.bias)
            else:
                try:
                    init.kaiming_uniform_(module.weight, nonlinearity='relu')
                    init.zeros_(module.bias)
                except AttributeError as e:
                    pass

    def forward(self, bow):
        x = self.hidden_layers(bow)
        x = self.dropout_enc(x)
        # 私有分布的参数
        muPrivate = self.fc_mu_private(x)
        logStdPrivate = self.fc_log_sigma_private(x)
        stdPrivate = torch.exp(logStdPrivate)
        # 共有表征
        muShared = self.fc_mu_shared(x)
        logStdShared = self.fc_log_sigma_shared(x)
        stdShared = torch.exp(logStdShared)
        # 私有的和共有的正太分布的参数
        return muPrivate, stdPrivate, muShared, stdShared


class DecoderNetworkFromGsm(nn.Module):
    def __init__(self, input_size, zPrivate_dim=15, zShared_dim=5):
        super(DecoderNetworkFromGsm, self).__init__()

        self.input_size = input_size
        self.zPrivate_dim = zPrivate_dim
        self.zShared_dim = zShared_dim
        self.topic_word_matrix = None  # 主题-词分布

        # init prior parameters
        z_prior_mean = 0.0
        self.prior_mean_private = torch.tensor([z_prior_mean] * zPrivate_dim)
        self.prior_mean_shared = torch.tensor([z_prior_mean] * zShared_dim)
        if torch.cuda.is_available():
            self.prior_mean_private = self.prior_mean_private.cuda()
            self.prior_mean_shared = self.prior_mean_shared.cuda()

        # zPrivate_prior_variance = 1. - (1. / self.zPrivate_dim)
        # zShared_prior_variance = 1. - (1. / self.zShared_dim)
        zPrivate_prior_variance = 1.
        zShared_prior_variance = 1.
        self.prior_variance_private = torch.tensor([zPrivate_prior_variance] * zPrivate_dim)
        self.prior_variance_shared = torch.tensor([zShared_prior_variance] * zShared_dim)
        if torch.cuda.is_available():
            self.prior_variance_private = self.prior_variance_private.cuda()
            self.prior_variance_shared = self.prior_variance_shared.cuda()

        # 初始化两个矩阵topic and word embedding
        self.topic_embedding = nn.Parameter(embedding_init(zPrivate_dim+zShared_dim, 100))
        self.word_embedding = nn.Parameter(embedding_init(self.input_size, 100))
        self.transform_theta = nn.Linear(self.zPrivate_dim+self.zShared_dim, self.zPrivate_dim+self.zShared_dim)

    # def forward(self, mu_private, std_private, mu_shared, std_shared):
    def forward(self, zPrivate, zShared):
        theta = F.softmax(self.transform_theta(torch.cat([zPrivate, zShared], -1)), dim=1)  # 主题分布

        # theta = F.softmax(torch.cat([zPrivate, zShared], -1), dim=1)  # 主题分布

        self.beta = torch.softmax(torch.matmul(self.topic_embedding,
                                    torch.transpose(self.word_embedding, 0, 1)), dim=1)
        self.topic_word_matrix = self.beta
        reconstruct_word_dist = torch.matmul(theta, self.beta)  # p_x_out的每一维就是log(p(wi|theta))

        return (self.prior_mean_private, self.prior_variance_private, self.prior_mean_shared,
                self.prior_variance_shared, reconstruct_word_dist)

    # def get_theta(self, mu_private, std_private, mu_shared, std_shared):
    def get_theta(self, zPrivate, zShared):
        with torch.no_grad():
            # 输出重构的词分布
            # zPrivate = self.reparameterize(mu_private, std_private)
            # zShared = self.reparameterize(mu_shared, std_shared)
            theta = F.softmax(self.transform_theta(torch.cat([zPrivate, zShared], -1)), dim=1)  # 主题分布
            # theta = F.softmax(torch.cat([zPrivate, zShared], -1), dim=1)  # 主题分布

            return theta


class GsmEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, topic_dim=25,
                 activation='relu', seed=888):
        super(GsmEncoder, self).__init__()
        self.seed = seed  # 随机数种子
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.topic_dim = topic_dim

        if activation == 'softplus':
            self.activation = nn.Softplus()  # f(x)=log(1+exp(x))
        elif activation == 'relu':
            self.activation = nn.ReLU()

        self.hidden_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            self.activation,
            nn.Linear(hidden_size, hidden_size),
            self.activation,
        )
        # neural networks with dropout to reduce overfitting.
        self.bz = nn.BatchNorm1d(hidden_size)
        self.dropout_enc = nn.Dropout(p=dropout)

        self.fc_mu = nn.Linear(hidden_size, topic_dim)
        self.fc_log_sigma = nn.Linear(hidden_size, topic_dim)

        self.encoder_theta = nn.Linear(topic_dim, topic_dim)

    def reparameterize(self, mu, std):
        """Reparameterize the theta distribution."""
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, bow):
        x = self.hidden_layers(bow)
        # x = self.bz(x)
        x = self.dropout_enc(x)
        # 私有分布的参数
        mu = self.fc_mu(x)
        logStd = self.fc_log_sigma(x)
        std = torch.exp(logStd)
        # 重参数采样
        z = self.reparameterize(mu, std)
        theta = F.softmax(self.encoder_theta(z), dim=1)  # 主题分布

        return mu, std, theta


class GsmDecoder(nn.Module):
    def __init__(self, input_size, topic_dim=25, seed=888):
        super(GsmDecoder, self).__init__()
        self.seed = seed  # 随机数种子
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        self.input_size = input_size
        self.topic_dim = topic_dim
        self.topic_word_matrix = None  # 主题-词分布

        # init prior parameters
        z_prior_mean = 0.0
        self.prior_mean = torch.tensor([z_prior_mean] * topic_dim)
        if torch.cuda.is_available():
            self.prior_mean = self.prior_mean.cuda()

        z_prior_variance = 1. - (1. / self.topic_dim)
        self.prior_variance = torch.tensor([z_prior_variance] * topic_dim)
        if torch.cuda.is_available():
            self.prior_variance = self.prior_variance.cuda()

        # 初始化两个矩阵topic and word embedding
        self.topic_embedding = nn.Parameter(embedding_init(topic_dim, 100))
        self.word_embedding = nn.Parameter(embedding_init(self.input_size, 100))

    def forward(self, theta):
        self.beta = torch.softmax(torch.matmul(self.topic_embedding,
                                    torch.transpose(self.word_embedding, 0, 1)), dim=1)
        self.topic_word_matrix = self.beta
        reconstruct_word_dist = torch.matmul(theta, self.beta)  # p_x_out的每一维就是log(p(wi|theta))

        return (self.prior_mean, self.prior_variance, reconstruct_word_dist)


class ProdLDAEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, topic_dim=25, activation='relu', seed=888):
        super(ProdLDAEncoder, self).__init__()
        self.seed = seed  # 随机数种子
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.topic_dim = topic_dim

        if activation == 'softplus':
            self.activation = nn.Softplus()  # f(x)=log(1+exp(x))
        elif activation == 'relu':
            self.activation = nn.ReLU()

        self.hidden_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            self.activation,
            nn.Linear(hidden_size, hidden_size),
            self.activation,
        )
        # neural networks with dropout to reduce overfitting.
        self.bz = nn.BatchNorm1d(hidden_size)
        self.dropout_enc = nn.Dropout(p=dropout)

        self.fc_mu = nn.Linear(hidden_size, topic_dim)
        self.fc_log_sigma = nn.Linear(hidden_size, topic_dim)

        self.encoder_theta = nn.Linear(topic_dim, topic_dim)

    def reparameterize(self, mu, std):
        """Reparameterize the theta distribution."""
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, bow):
        x = self.hidden_layers(bow)
        # x = self.bz(x)
        x = self.dropout_enc(x)
        # 私有分布的参数
        mu = self.fc_mu(x)
        logStd = self.fc_log_sigma(x)
        std = torch.exp(logStd)
        # 重参数采样
        z = self.reparameterize(mu, std)
        theta = F.softmax(self.encoder_theta(z), dim=1)  # 主题分布

        return mu, std, theta


class ProdLDADecoder(nn.Module):
    def __init__(self, input_size, topic_dim=25, seed=888):
        super(ProdLDADecoder, self).__init__()
        self.seed = seed  # 随机数种子
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        self.input_size = input_size
        self.topic_dim = topic_dim
        self.topic_word_matrix = None  # 主题-词分布

        # init prior parameters
        z_prior_mean = 0.0
        self.prior_mean = torch.tensor([z_prior_mean] * topic_dim)
        if torch.cuda.is_available():
            self.prior_mean = self.prior_mean.cuda()

        z_prior_variance = 1. - (1. / self.topic_dim)
        self.prior_variance = torch.tensor([z_prior_variance] * topic_dim)
        if torch.cuda.is_available():
            self.prior_variance = self.prior_variance.cuda()

        # 初始化两个矩阵topic and word embedding
        self.beta = nn.Parameter(torch.Tensor((self.topic_dim), input_size).cuda())  # 主题-词分布
        nn.init.xavier_uniform_(self.beta)

        self.beta_batchnorm = nn.BatchNorm1d(input_size, affine=False)
        self.drop_theta = nn.Dropout(p=0.2)

    def forward(self, theta):
        theta = self.drop_theta(theta)
        reconstruct_word_dist = F.softmax(torch.matmul(theta, self.beta), dim=1)
        # reconstruct_word_dist = F.softmax(self.beta_batchnorm(torch.matmul(theta, self.beta)), dim=1)
        self.topic_word_matrix = self.beta

        return (self.prior_mean, self.prior_variance, reconstruct_word_dist)

