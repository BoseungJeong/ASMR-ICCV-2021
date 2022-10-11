import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
# from pytorch_metric_learning import miners, losses
import numpy as np
import pdb

def binarize(T, nb_classes):
    T = T.cpu().numpy()
    if nb_classes == 2:
        tmp = torch.zeros(len(T),nb_classes)
        for i in range(len(T)):
            tmp[i,int(T[i])] = 1
        T = tmp
    else:
        import sklearn.preprocessing
        T = sklearn.preprocessing.label_binarize(
            T, classes = range(0, nb_classes)
        )
    T = torch.FloatTensor(T).cuda()
    return T

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.s = s
        self.m = m

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, cos, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
#         cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cos, 2)).clamp(0, 1))
        phi = cos * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cos > 0, phi, cos)
        else:
            phi = torch.where(cos > self.th, phi, cos - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cos.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cos)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output
    



class MA_ArcFace(torch.nn.Module):
    def __init__(self, sigma = 16, margin = 0.3):
        torch.nn.Module.__init__(self)
        self.sigma = sigma
        self.margin = margin
        self.ArcMarginProduct = ArcMarginProduct(self.sigma, self.margin,easy_margin=False)


    def forward(self, X, X_att, T):
        loss = 0
        P = X_att
        norm_P = l2_norm(P)
        norm_X = l2_norm(X)
        cos = F.linear(norm_X, norm_P)
        logit = self.ArcMarginProduct(cos, T)
#         pdb.set_trace()
#         logit = self.sigma * (cos - self.margin)
        loss += F.cross_entropy(logit,T)

        return cos, loss

class ASMR(nn.Module):
    def __init__(self, nb_classes, nb_categories):
        super(ASMR, self).__init__()
        self.nb_classes = nb_classes
        self.nb_categories = nb_categories
        self.weight = nn.Parameter(((torch.rand(self.nb_categories,1)-0.5)/2.0).cuda())
#         self.Softplus = torch.nn.Softplus(1, 1)

    def forward(self, X_att, att_diff):
        mc = self.nb_classes * 1
        
        norm_att = l2_norm(X_att)
        att_cos = F.linear(norm_att, norm_att).triu(diagonal=1)
        diff_pid = torch.matmul(att_diff.float(), self.weight).squeeze(-1).triu(diagonal=1)
        
        sigmoid_diff = (torch.sigmoid(1-diff_pid/10)).triu(diagonal=1)
 
        cos_mu = 2.0 / (mc**2 - mc) * torch.sum(att_cos)
        cos_residual = torch.sum(((att_cos - cos_mu - sigmoid_diff)**2).triu(diagonal=1))
        cos_rw = 2.0 / (mc**2 - mc) * cos_residual
 
        return cos_mu, cos_rw, self.weight

#### Our ASMR ####
# class ASMR(nn.Module):
#     def __init__(self, nb_classes, nb_categories):
#         super(ASMR, self).__init__()
#         self.nb_classes = nb_classes
#         self.nb_categories = nb_categories
#         self.weight = nn.Parameter((torch.rand(self.nb_categories,1).cuda()-0.5)/2,requires_grad=True)
#         self.ones = torch.ones_like(self.weight)
#     def forward(self, att, diff):
#         mc = self.nb_classes * 1 # for one proxy in one class
#         W = self.weight
#         ones = self.ones
#         weights = ones + W
#         norm_att = l2_norm(att)
#         att_cos = F.linear(norm_att, norm_att).triu(diagonal=1)
#         diff_pid = torch.matmul(diff.float(), weights).squeeze(-1).triu(diagonal=1)
#         tanh_diff = nn.Sigmoid()(1.0-diff_pid/10.0).triu(diagonal=1)
#         w_decay = torch.norm(weights, p=2)
#         cos_mu = 2.0 / (mc**2 - mc) * torch.sum(att_cos)
#         cos_residual = torch.sum(((att_cos - cos_mu - tanh_diff)**2).triu(diagonal=1))
#         cos_rw = 2.0 / (mc**2 - mc) * cos_residual
#         return cos_mu, cos_rw, w_decay, weights

class ASMR_PETA(nn.Module):
    def __init__(self, nb_classes, nb_categories):
        super(ASMR_PETA, self).__init__()
        self.nb_classes = nb_classes
        self.nb_categories = nb_categories
        self.weight = nn.Parameter((torch.rand(self.nb_categories,1).cuda()-0.5)/2,requires_grad=True)
#         self.weight = nn.Parameter(torch.rand(self.nb_categories,1).cuda(),requires_grad=True)
#         nn.init.kaiming_normal_(self.weight, mode='fan_out')
        self.ones = torch.ones_like(self.weight)
#         self.w = self.ones + self.weight
    def forward(self, att, diff):
        mc = self.nb_classes * 1 # for one proxy in one class
        W = self.weight
        ones = self.ones
        weights = ones + W
#         W_sum = W.sum()
#         reg_W = (W_sum - torch.tensor(10.0))**2
#         att = l2_norm(att)
#         att_1 = att.reshape(1,att.shape[0],-1)
#         att_2 = att.reshape(att.shape[0],1,-1)
        norm_att = l2_norm(att)
        att_cos = F.linear(norm_att, norm_att).triu(diagonal=1)
        diff_pid = torch.matmul(diff.float(), weights).squeeze(-1).triu(diagonal=1)
#         diff_pid = diff.sum(-1).triu(diagonal=1)
        tanh_diff = nn.Sigmoid()(1.0-diff_pid/8.5).triu(diagonal=1)
        norm_w = 0.0
#         norm_w = torch.norm(W, p=2)**2
#         norm_w = nn.MSELoss()(W,target)


        cos_mu = 2.0 / (mc**2 - mc) * torch.sum(att_cos)
        cos_residual = torch.sum(((att_cos - cos_mu - tanh_diff)**2).triu(diagonal=1))
#         cos_residual = torch.sum(((att_cos - cos_mu + self.K - (1.0/num_inst).triu(diagonal=1))**2).triu(diagonal=1))
        cos_rw = 2.0 / (mc**2 - mc) * cos_residual
#         pdb.set_trace()
        return cos_mu, cos_rw, weights

