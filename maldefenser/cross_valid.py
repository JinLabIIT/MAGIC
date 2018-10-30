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
from typing import Dict, List
from ml_utils import cmd_args, gHP, S2VGraph
from ml_utils import computePrScores, loadGraphsMayCache, kFoldSplit
from e2e_model import Classifier, loopDataset
from hyperparameters import HyperParameterIterator, parseHpTuning


def trainThenValid(trainGraphs, validGraphs):
    classifier = Classifier()
    if cmd_args.mode == 'gpu':
        classifier = classifier.cuda()

    optimizer = optim.Adam(classifier.parameters(), lr=gHP['lr'])

    trainIndices = list(range(len(trainGraphs)))
    validIndices = list(range(len(validGraphs)))
    trainLossHist, validLossHist = [], []
    trainAccuHist, validAccuHist = [], []
    trainPrecHist, validPrecHist = [], []
    trainRecallHist, validRecallHist = [], []
    trainF1Hist, validF1Hist = [], []

    startTime = time.process_time()

    for epoch in range(gHP['numEpochs']):
        random.shuffle(trainIndices)
        classifier.train()
        avgScore, trainPred, trainLabels = loopDataset(
            trainGraphs, classifier, trainIndices, optimizer=optimizer)
        prScore = computePrScores(trainPred, trainLabels, 'train')
        print('\033[92mTrain epoch %d: l %.5f a %.5f p %.5f r %.5f\033[0m' %
              (epoch, avgScore[0], avgScore[1], prScore['precisions'],
               prScore['recalls']))
        trainLossHist.append(avgScore[0])
        trainAccuHist.append(avgScore[1])
        trainPrecHist.append(prScore['precisions'])
        trainRecallHist.append(prScore['recalls'])
        trainF1Hist.append(prScore['weightedF1'])

        classifier.eval()
        validScore, validPred, validLabels = loopDataset(
            validGraphs, classifier, validIndices)
        prScore = computePrScores(validPred, validLabels, 'valid')
        print('\033[93mValid epoch %d: l %.5f a %.5f p %.5f r %.5f\033[0m' %
              (epoch, validScore[0], validScore[1],
               prScore['precisions'], prScore['recalls']))
        validLossHist.append(validScore[0])
        validAccuHist.append(validScore[1])
        validPrecHist.append(prScore['precisions'])
        validRecallHist.append(prScore['recalls'])
        validF1Hist.append(prScore['weightedF1'])

    log.info(f'Net training time = {time.process_time() - startTime} seconds')
    hist = {}
    hist['TrainLoss'] = trainLossHist
    hist['TrainAccu'] = trainAccuHist
    hist['TrainPrec'] = trainPrecHist
    hist['TrainRecl'] = trainRecallHist
    hist['TrainF1'] = trainF1Hist
    hist['ValidLoss'] = validLossHist
    hist['ValidAccu'] = validAccuHist
    hist['ValidPrec'] = validPrecHist
    hist['ValidRecl'] = validRecallHist
    hist['ValidF1'] = validF1Hist
    return hist


def averageMetrics(kFoldHis: List[Dict[str, List[float]]]) -> Dict[str, float]:
    """Avg over k-fold training history"""
    result = {}
    for history in kFoldHis:
        for (name, value) in history.items():
            if name in result:
                result[name] = result[name] + np.array(value)
            else:
                result[name] = np.array(value)

    avgResult = {}
    for (name, seq) in result.items():
        avgName = 'Avg' + name
        avgResult[avgName] = result[name] / float(len(kFoldHis))
        log.info(f'{avgName} = {avgResult[avgName]}')

    return avgResult


def crossValidate(graphFolds: List[List[S2VGraph]], runId: int) -> None:
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
    histFile = open('%sGpu%sRun%s.csv' %
                    (cmd_args.data, cmd_args.gpu_id, runId), 'w')
    histFile.write("# %s\n" % str(gHP))
    df.to_csv(histFile, index_label='Epoch', float_format='%.6f')
    histFile.close()


if __name__ == '__main__':
    log.setLevel("INFO")

    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    startTime = time.process_time()
    graphs = loadGraphsMayCache(cmd_args.train_dir)
    dataReadyTime = time.process_time() - startTime
    log.info('Dataset ready takes %.2fs' % dataReadyTime)

    for (id, hp) in enumerate(HyperParameterIterator(cmd_args.hp_path)):
        for (key, val) in hp.items():
            gHP[key] = val

        numNodesList = sorted([g.num_nodes for g in graphs])
        idx = int(math.ceil(hp['sortPoolingRatio'] * len(graphs))) - 1
        gHP['sortPoolingK'] = numNodesList[idx]
        log.info(f"Hyperparameter setting: {gHP}")

        kFoldGraphs = kFoldSplit(gHP['cvFold'], graphs)
        crossValidate(kFoldGraphs, id)

    optHp = parseHpTuning(cmd_args.data)
    log.info(f'Optimal hyperparameter setting: {optHp}')
