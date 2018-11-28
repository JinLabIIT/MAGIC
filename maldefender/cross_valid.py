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
from ml_utils import cmd_args, gHP, S2VGraph, normalizeFeatures
from ml_utils import loadGraphsMayCache, kFoldSplit
from e2e_model import Classifier, loopDataset
from hyperparameters import HyperParameterIterator, parseHpTuning
from torch.optim.lr_scheduler import ReduceLROnPlateau


def trainThenValid(trainGraphs: List[S2VGraph], validGraphs: List[S2VGraph]):
    classifier = Classifier()
    log.info(f"Hyperparameter setting: {gHP}")

    if cmd_args.mode == 'gpu':
        classifier = classifier.cuda()

    optimizer = optim.SGD(classifier.parameters(),
                          lr=gHP['lr'], momentum=0.9,
                          weight_decay=gHP['l2RegFactor'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5,
                                  patience=4, verbose=True)
    trainIndices = list(range(len(trainGraphs)))
    validIndices = list(range(len(validGraphs)))
    trainLossHist, validLossHist = [], []

    startTime = time.process_time()
    for i in range(gHP['numEpochs']):
        random.shuffle(trainIndices)
        log.debug(f'First 10 train indices: {trainIndices[:10]}')
        classifier.train()
        avgScore, trainPred, trainLabels = loopDataset(
            trainGraphs, classifier, trainIndices, optimizer=optimizer)
        print('\033[92mTrain epoch %d: loss %.6f\033[0m' % (i, avgScore[0]))
        trainLossHist.append(avgScore[0])

        classifier.eval()
        validScore, validPred, validLabels = loopDataset(
            validGraphs, classifier, validIndices)
        print('\033[93mValid epoch %d: loss %.6f\033[0m' % (i, validScore[0]))
        validLossHist.append(validScore[0])
        scheduler.step(validScore[0])

    log.info(f'Net training time = {time.process_time() - startTime} seconds')
    hist = {}
    hist['TrainLoss'] = trainLossHist
    hist['ValidLoss'] = validLossHist
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
    normalizeFeatures(graphs, isTestSet=False, operation='min_max')
    dataReadyTime = time.process_time() - startTime
    log.info('Dataset ready takes %.2fs' % dataReadyTime)

    for (id, hp) in enumerate(HyperParameterIterator(cmd_args.hp_path)):
        for (key, val) in hp.items():
            gHP[key] = val

        numNodesList = sorted([g.num_nodes for g in graphs])
        idx = int(math.ceil(hp['poolingRatio'] * len(graphs))) - 1
        gHP['poolingK'] = numNodesList[idx]

        kFoldGraphs = kFoldSplit(gHP['cvFold'], graphs)
        crossValidate(kFoldGraphs, id)

    optHp = parseHpTuning(cmd_args.data)
    log.info(f'Optimal hyperparameter setting: {optHp}')
