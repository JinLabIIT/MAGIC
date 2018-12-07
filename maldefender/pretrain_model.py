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
from ml_utils import cmd_args, gHP, kFoldSplit, S2VGraph
from ml_utils import computePrScores, loadGraphsMayCache
from ml_utils import storeConfusionMatrix, normalizeFeatures
from ml_utils import adjustBatchSize, filterOutNoEdgeGraphs
from ml_utils import loadModel, saveModel, decideHyperparameters
from e2e_model import Classifier, loopDataset, predictDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau


def preTrain(trainSet: List[S2VGraph], numEpochs: int) -> Dict[str, float]:
    classifier = Classifier()
    loadModel(classifier)
    log.info(f'Global hyperparameter setting: {gHP}')
    if cmd_args.mode == 'gpu':
        classifier = classifier.cuda()

    optimizer = optim.Adam(classifier.parameters(), lr=gHP['lr'],
                           weight_decay=gHP['l2RegFactor'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=1,
                                  verbose=True)
    kFoldGraphs = kFoldSplit(max(gHP['cvFold'], 5), trainSet)
    trainGraphs = []
    for foldGraphs in kFoldGraphs[:-1]:
        trainGraphs.extend(foldGraphs)

    validGraphs = kFoldGraphs[-1]
    trainIndices = list(range(len(trainGraphs)))
    validIndices = list(range(len(validGraphs)))
    trainLossHist, trainAccuHist = [], []
    trainPrecHist, trainRecallHist, trainF1Hist = [], [], []
    validLossHist, validAccuHist = [], []
    validPrecHist, validRecallHist, validF1Hist = [], [], []

    startTime = time.process_time()
    for e in range(numEpochs):
        random.shuffle(trainIndices)
        log.debug(f'First 10 train indices: {trainIndices[:10]}')
        classifier.train()
        avgScore, trainPred, trainLabels = loopDataset(
            trainGraphs, classifier, trainIndices, optimizer=optimizer)
        prScore = computePrScores(trainPred, trainLabels, 'train')
        line = '\033[92mTrain epoch %d: l %.5f, a %.5f\033[0m'
        print(line % (e, avgScore[0], avgScore[1]))
        trainLossHist.append(avgScore[0])
        trainAccuHist.append(avgScore[1])
        trainPrecHist.append(prScore['Precision'])
        trainRecallHist.append(prScore['Recall'])
        trainF1Hist.append(prScore['F1'])

        classifier.eval()
        validScore, validPred, validLabels = loopDataset(
            validGraphs, classifier, validIndices)
        scheduler.step(validScore[0])

        prScore = computePrScores(validPred, validLabels, 'valid')
        line = '\033[93mValid epoch %d: l %.5f, a %.5f, p %.5f, r %.5f, f1 %.5f\033[0m'
        print(line % (e, validScore[0], validScore[1], prScore['Precision'],
                      prScore['Recall'], prScore['F1']))
        validLossHist.append(validScore[0])
        validAccuHist.append(validScore[1])
        validPrecHist.append(prScore['Precision'])
        validRecallHist.append(prScore['Recall'])
        validF1Hist.append(prScore['F1'])
        # adjustBatchSize(optimizer, validLossHist)

        if validScore[0] < 0.04 or e % 10 == 0:
            log.info(f'Save model with {validScore[0]} validation loss.')
            saveModel(classifier, msg='_vl%.6f' % validScore[0])
            storeConfusionMatrix(trainPred, trainLabels, 'train_e%d' % e)
            storeConfusionMatrix(validPred, validLabels, 'valid_e%d' % e)
            computePrScores(trainPred, trainLabels, 'train_e%d' % e, None, store=True)
            computePrScores(validPred, validLabels, 'valid_e%d' % e, None, store=True)

    log.info(f'Net training time = {time.process_time() - startTime} seconds')
    storeConfusionMatrix(trainPred, trainLabels, 'train')
    storeConfusionMatrix(validPred, validLabels, 'valid')
    computePrScores(trainPred, trainLabels, 'train', None, store=True)
    computePrScores(validPred, validLabels, 'valid', None, store=True)
    result = {
        'TrainLoss': trainLossHist,
        'TrainAccu': trainAccuHist,
        'TrainPrec': trainPrecHist,
        'TrainRecl': trainRecallHist,
        'TrainF1': trainF1Hist,
        'ValidLoss': validLossHist,
        'ValidAccu': validAccuHist,
        'ValidPrec': validPrecHist,
        'ValidRecl': validRecallHist,
        'ValidF1': validF1Hist,
    }
    log.info(f'Model validset loss:\n{validLossHist}')
    df = pd.DataFrame.from_dict(result)
    histFile = open(cmd_args.train_dir + '/%sPredHist.csv' % cmd_args.data, 'w')
    histFile.write("# %s\n" % str(gHP))
    df.to_csv(histFile, index_label='Epoch', float_format='%.6f')
    histFile.close()


if __name__ == '__main__':
    log.setLevel("INFO")
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    startTime = time.process_time()
    trainGraphs = loadGraphsMayCache(cmd_args.train_dir, False)
    normalizeFeatures(trainGraphs, isTestSet=False, operation=cmd_args.norm_op)
    trainGraphs = filterOutNoEdgeGraphs(trainGraphs)
    dataReadyTime = time.process_time() - startTime
    log.info('Trainset ready takes %.2fs' % dataReadyTime)

    numEpochs = decideHyperparameters(trainGraphs)
    preTrain(trainGraphs, numEpochs)
