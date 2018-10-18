#!/usr/bin/python3.7
import sys
import os
import torch
import math
import random
import time
import glog as log
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from dgcnn_embedding import DGCNN
from mlp_dropout import MLPClassifier, RecallAtPrecision
from embedding import EmbedMeanField, EmbedLoopyBP
from ml_utils import cmd_args, storeConfusionMatrix
from ml_utils import computePrScores, loadGraphsMayCache, kFoldSplit
from e2e_model import Classifier


def loopDataset(g_list, classifier, sampleIndices,
                 optimizer=None, bsize=cmd_args.batch_size):
    total_score = []
    numGiven = len(sampleIndices)
    total_iters = math.ceil(numGiven / bsize)
    pbar = tqdm(range(total_iters), unit='batch')

    numUsed = 0
    all_pred = []
    all_label = []
    for pos in pbar:
        end = min((pos + 1) * bsize, numGiven)
        batch_indices = sampleIndices[pos * bsize: end]
        batch_graph = [g_list[idx] for idx in batch_indices]
        if classifier.training:
            classifier.sgdModel(optimizer, batch_graph, pos)

        loss, acc, pred = classifier(batch_graph)
        all_pred.extend(pred.data.cpu().numpy().tolist())
        all_label.extend([g.label for g in batch_graph])
        loss = loss.data.cpu().numpy()
        pbar.set_description('loss: %0.5f acc: %0.5f' % (loss, acc))
        total_score.append(np.array([loss, acc]))
        numUsed += len(batch_indices)

    if numUsed != numGiven:
        log.warning(f"{numUsed} of {numGiven} cases used trainning/validating.")

    classifier.mlp.print_result_dict()
    total_score = np.array(total_score)
    avg_score = np.mean(np.array(total_score), 0)
    return avg_score, all_pred, all_label


def trainThenValid(trainGraphs, validGraphs):
    classifier = Classifier()
    if cmd_args.mode == 'gpu':
        classifier = classifier.cuda()

    optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)

    trainIndices = list(range(len(trainGraphs)))
    validIndices = list(range(len(validGraphs)))
    trainLossHist, validLossHist = [], []
    trainAccuHist, validAccuHist = [], []
    trainPrecHist, validPrecHist = [], []
    trainRecallHist, validRecallHist = [], []
    startTime = time.process_time()

    for epoch in range(cmd_args.num_epochs):
        random.shuffle(trainIndices)
        classifier.train()
        avgScore, trainPred, trainLabels = loopDataset(
            trainGraphs, classifier, trainIndices, optimizer=optimizer)
        prScore = computePrScores(trainPred, trainLabels, 'train')
        print('\033[92mTrain epoch %d: l %.5f a %.5f p %.5f r %.5f\033[0m' %
              (epoch, avgScore[0], avgScore[1], prScore['precisions'][1],
               prScore['recalls'][1]))
        trainLossHist.append(avgScore[0])
        trainAccuHist.append(avgScore[1])
        trainPrecHist.append(prScore['precisions'][1])
        trainRecallHist.append(prScore['recalls'][1])

        classifier.eval()
        validScore, validPred, validLabels = loopDataset(
            validGraphs, classifier, validIndices)
        prScore = computePrScores(validPred, validLabels, 'valid')
        print('\033[93mValid epoch %d: l %.5f a %.5f p %.5f r %.5f\033[0m' %
              (epoch, validScore[0], validScore[1],
               prScore['precisions'][1], prScore['recalls'][1]))
        validLossHist.append(validScore[0])
        validAccuHist.append(validScore[1])
        validPrecHist.append(prScore['precisions'][1])
        validRecallHist.append(prScore['recalls'][1])

    log.info(f'Net training time = {time.process_time() - startTime} seconds')
    hist = {}
    hist['TrainLoss'] = trainLossHist
    hist['TrainAccu'] = trainAccuHist
    hist['TrainPrec'] = trainPrecHist
    hist['TrainRecl'] = trainRecallHist
    hist['ValidLoss'] = validLossHist
    hist['ValidAccu'] = validAccuHist
    hist['ValidPrec'] = validPrecHist
    hist['ValidRecl'] = validRecallHist
    return hist


def averageMetrics(metrics):
    result = {}
    for history in metrics:
        for (name, value) in history.items():
            if name in result:
                result[name] = result[name] + np.array(value)
            else:
                result[name] = np.array(value)

    avgResult = {}
    for (name, seq) in result.items():
        avgName = 'Avg' + name
        avgResult[avgName] = result[name] / float(len(metrics))
        log.debug(f'{avgName} = {avgResult[avgName]}')

    return avgResult


def crossValidate(graphFolds):
    cvMetrics = []
    for f in range(len(graphFolds)):
        log.info(f'Start {f + 1}th cross validation tranning')
        trainGraphs, validGraphs = [], []
        for i in range(len(graphFolds)):
            if i == f:
                validGraphs = graphFolds[i]
            else:
                trainGraphs.extend(graphFolds[i])

        hist = trainThenValid(trainGraphs, validGraphs)
        cvMetrics.append(hist)

    avgMetrics = averageMetrics(cvMetrics)
    df = pd.DataFrame.from_dict(avgMetrics)
    df.to_csv('%s.hist' % cmd_args.data, index_label='Epoch', float_format='%.6f')


if __name__ == '__main__':
    log.setLevel("INFO")

    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    start_time = time.process_time()
    graphs = loadGraphsMayCache()
    data_ready_time = time.process_time() - start_time
    log.info('Dataset ready takes %.2fs' % data_ready_time)

    kFoldGraphs = kFoldSplit(5, graphs)
    crossValidate(kFoldGraphs)

    # train_graphs, valid_graphs = kFoldGraphs[0], kFoldGraphs[1]
    #
    # classifier = Classifier()
    # if cmd_args.mode == 'gpu':
    #     classifier = classifier.cuda()
    #
    # optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)
    #
    # train_indices = list(range(len(train_graphs)))
    # valid_indices = list(range(len(valid_graphs)))
    # train_loss_hist = []
    # train_accu_hist = []
    # train_prec_hist = []
    # train_recall_hist = []
    # valid_loss_hist = []
    # valid_accu_hist = []
    # valid_prec_hist = []
    # valid_recall_hist = []
    # for epoch in range(cmd_args.num_epochs):
    #     random.shuffle(train_indices)
    #     classifier.train()
    #     avg_score, train_pred, train_labels = loopDataset(
    #         train_graphs, classifier, train_indices, optimizer=optimizer)
    #     pr_score = computePrScores(train_pred, train_labels, 'train')
    #     print('\033[92mTrain epoch %d: l %.5f a %.5f p %.5f r %.5f\033[0m' %
    #           (epoch, avg_score[0], avg_score[1], pr_score['precisions'][1],
    #            pr_score['recalls'][1]))
    #     train_loss_hist.append(avg_score[0])
    #     train_accu_hist.append(avg_score[1])
    #     train_prec_hist.append(pr_score['precisions'][1])
    #     train_recall_hist.append(pr_score['recalls'][1])
    #
    #     classifier.eval()
    #     valid_score, valid_pred, valid_labels = loopDataset(
    #         valid_graphs, classifier, valid_indices)
    #     pr_score = computePrScores(valid_pred, valid_labels, 'valid')
    #     print('\033[93mValid epoch %d: l %.5f a %.5f p %.5f r %.5f\033[0m' %
    #           (epoch, valid_score[0], valid_score[1],
    #            pr_score['precisions'][1], pr_score['recalls'][1]))
    #     valid_loss_hist.append(valid_score[0])
    #     valid_accu_hist.append(valid_score[1])
    #     valid_prec_hist.append(pr_score['precisions'][1])
    #     valid_recall_hist.append(pr_score['recalls'][1])
    #
    #     if epoch + 1 == cmd_args.num_epochs:
    #         df = pd.DataFrame.from_dict(pr_score)
    #         df.to_csv(
    #             '%s_valid_pr_scores.txt' % cmd_args.data, float_format='%.4f')
    #         storeConfusionMatrix(train_pred, train_labels, 'train')
    #         storeConfusionMatrix(valid_pred, valid_labels, 'valid')
    #         # store_embedding(classifier, train_graphs, 'train')
    #         # store_embedding(classifier, valid_graphs, 'valid')
    #
    # duration = time.process_time() - start_time
    # log.info('Net training time = %.2f - %.2f = %.2fs' %
    #          (duration, data_ready_time, duration - data_ready_time))
    # torch.save(classifier.state_dict(),
    #            '%s_%s.model' % (cmd_args.data, cmd_args.mlp_type))
    # hist = {}
    # hist['train_loss'] = train_loss_hist
    # hist['train_accu'] = train_accu_hist
    # hist['train_prec'] = train_prec_hist
    # hist['train_recall'] = train_recall_hist
    # hist['valid_loss'] = valid_loss_hist
    # hist['valid_accu'] = valid_accu_hist
    # hist['valid_prec'] = valid_prec_hist
    # hist['valid_recall'] = valid_recall_hist
    # df = pd.DataFrame.from_dict(hist)
    # df.to_csv('%s_hist.txt' % cmd_args.data, float_format='%.6f')
