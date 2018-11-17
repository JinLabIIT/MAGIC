#!/usr/bin/python3.7
"""
Borrowed from and rewritten based on Muhan's pytorch_DGCNN repo at
https://github.com/muhanzhang/pytorch_DGCNN
"""
import os
import sys
import torch
import glog as log
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

sys.path.append('%s/pytorch_structure2vec-master/s2v_lib'
                % os.path.dirname(os.path.realpath(__file__)))
from pytorch_util import weights_init


class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_labels):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_labels)
        weights_init(self)

    def forward(self, x, y=None):
        predProb = F.softmax(self.linear(x), dim=1)
        logProb = F.log_softmax(self.linear(x), dim=1)
        if y is None:
            return predProb
        else:
            pred = predProb.data.max(1)[1]
            loss = F.nll_loss(logProb, y)
            correct = pred.eq(y.data.view_as(pred))
            accu = (correct.sum().item()) / float(correct.size(0))
            return loss, accu, pred

    def print_result_dict(self):
        pass


class MLPRegression(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLPRegression, self).__init__()

        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, 1)
        weights_init(self)

    def forward(self, x, y=None):
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)

        pred = self.h2_weights(h1)

        if y is not None:
            y = Variable(y)
            mse = F.mse_loss(pred, y)
            mae = F.l1_loss(pred, y)
            return pred, mae, mse
        else:
            log.debug('[MLPRegression] None label, return only predictions.')
            return pred

    def print_result_dict(self):
        pass


class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, dropout=0.0):
        super(MLPClassifier, self).__init__()

        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h1_bn = nn.BatchNorm1d(hidden_size)
        self.h2_weights = nn.Linear(hidden_size, num_class)
        self.dropout = dropout
        weights_init(self)

    def forward(self, x, y=None):
        h1 = self.h1_weights(x)
        h1 = self.h1_bn(h1)
        h1 = torch.tanh(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        h2 = self.h2_weights(h1)
        predProb = F.softmax(h2, dim=1)
        logits = F.log_softmax(h2, dim=1)
        pred = logits.data.max(1)[1]

        if y is not None:
            y = Variable(y)
            loss = F.nll_loss(logits, y)
            correct = pred.eq(y.data.view_as(pred))
            accu = (correct.sum().item()) / float(correct.size(0))
            return loss, accu, pred
        else:
            log.debug('[MLPClassifier] None label, return only predict prob.')
            return predProb

    def print_result_dict(self):
        pass


class RecallAtPrecision(nn.Module):
    def __init__(self, input_size, hidden_size, alpha, dropout=0.0):
        super(RecallAtPrecision, self).__init__()

        self.device = torch.device('cuda')
        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, 2)
        self.dropout = dropout

        self.alpha = alpha
        self.alpha_term = alpha / (1 - alpha)
        self.lam = Parameter(torch.tensor([2.0], device=self.device,
                                          requires_grad=True))
        self.result_dict = {}
        log.info('Optimize recall @ fixed precision=%.2f' % self.alpha)

        weights_init(self)

    def print_result_dict(self):
        TP = self.result_dict['true_pos']
        FP = self.result_dict['false_pos']
        NYP = self.result_dict['num_Y_pos']
        TPL = self.result_dict['tp_lower']
        FPU = self.result_dict['fp_upper']
        precision, recall = TP / (TP + FP + 1e-10), TP / (NYP + 1e-10)

        if self.training:
            # log.info(self.h1_weights.weight)
            log.info('lambda = %.5f' % self.lam.item())

        log.info('TP = %.1f(>=%.1f), FP = %.1f(<=%.1f), |Y+| = %.1f' %
                 (TP, TPL, FP, FPU, NYP))
        log.info('precision = %.5f, recall = %.5f' % (precision, recall))
        log.info('inequality = %.5f(<=0)' % self.result_dict['inequality'])
        # recall_lb, precision_lb = TPL / (NP + 1e-5), TPL / (TPL + FPU + 1e-5)
        # log.info('R LB = %.5f, P LB = %.5f' % (recall_lb, precision_lb))

    def forward(self, X, target=None):
        """
        logits = f(X), target = Y in {0, 1}
        """
        h1 = self.h1_weights(X)
        h1 = F.sigmoid(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        logits = self.h2_weights(h1)
        logits = F.softmax(logits, dim=1)
        pred_cls = (logits[:, 1] > logits[:, 0]).to(torch.int32)

        if target is None:
            return pred_cls

        target = target.to(torch.float32)
        y = 2 * target - 1  # y must in {-1, 1}
        L = 0.0

        # pred belong to {-1, 1}
        # pred = (logits[:, 1] - self.bias) * 2 - 1
        pred = (logits[:, 1] - logits[:, 0]) * 2 - 1
        hinge_loss = torch.max(1 - y * pred, torch.tensor(0.0).to(y.device))
        Lp = (hinge_loss * target).sum()
        Ln = (hinge_loss * (1 - target)).sum()
        # L = (1 + lam) * Lp + lam * alpha_term * Ln - lam * target.sum()
        L = Lp + self.lam * (self.alpha_term * Ln + Lp - target.sum())

        # # pred_cls and pred_cls_float belong to {0.0, 1.0}
        pred_cls_float = (logits[:, 1] > logits[:, 0]).to(torch.float32)
        true_pos = (target * pred_cls_float).sum().item()
        false_pos = ((1 - target) * pred_cls_float).sum().item()
        num_Y_pos = target.sum().item()  # NOT positive of predicition
        tp_lower = (num_Y_pos - Lp).item()
        fp_upper = Ln.item()
        inequality = self.alpha_term * Ln + Lp - self.lam * num_Y_pos

        keys = ['true_pos', 'false_pos', 'num_Y_pos', 'tp_lower', 'fp_upper',
                'inequality', ]
        values = [true_pos, false_pos, num_Y_pos, tp_lower, fp_upper,
                  inequality, ]
        for key, value in zip(keys, values):
            self.result_dict[key] = value

        correct = pred_cls.eq(target.to(torch.int32).data.view_as(pred_cls))
        accu = (correct.sum().item()) / float(correct.size(0))

        if self.lam.requires_grad is True and self.training is True:
            L *= -1

        return L, accu, pred_cls
