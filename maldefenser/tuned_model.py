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
from ml_utils import cmd_args, gHP, storeConfusionMatrix
from ml_utils import computePrScores, loadGraphsMayCache, loadData
from e2e_model import Classifier, loopDataset
from hyperparameters import parseHpTuning


def exportPredictions(graphs, preds):
    log.info(f'{preds.shape}')
    output = open(cmd_args.test_dir + '/submission.csv', 'w')
    output.write('"Id","Prediction1","Prediction2","Prediction3","Prediction4","Prediction5","Prediction6","Prediction7","Prediction8","Prediction9"\n')
    for (i, g) in enumerate(graphs):
        elems = ['"' + str(g.bId) + '"']
        for k in range(max(gHP['numClasses'], 9)):
            if int(preds[i]) == k:
                elems.append('1.0')
            else:
                elems.append('0.0')

        output.write('%s\n' % ",".join(elems))

    output.close()


def trainThenPredict(trainGraphs, testGraphs) -> Dict[str, float]:
    classifier = Classifier()
    if cmd_args.mode == 'gpu':
        classifier = classifier.cuda()

    optimizer = optim.Adam(classifier.parameters(), lr=gHP['lr'])

    trainIndices = list(range(len(trainGraphs)))
    trainLossHist, trainAccuHist = [], []
    trainPrecHist, trainRecallHist, trainF1Hist = [], [], []
    startTime = time.process_time()
    for epoch in range(int(gHP['optNumEpochs'])):
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

    storeConfusionMatrix(trainPred, trainLabels, 'train')

    log.info(f'Net training time = {time.process_time() - startTime} seconds')
    classifier.eval()
    testPred = classifier.predict(testGraphs)
    result = {
        'TrainLoss': trainLossHist,
        'TrainAccu': trainAccuHist,
        'TrainPrec': trainPrecHist,
        'TrainRecl': trainRecallHist,
        'TrainF1': trainF1Hist,
    }
    log.info(f'Model trainset performance: {result}')
    exportPredictions(testGraphs, testPred)
    return result


if __name__ == '__main__':
    log.setLevel("INFO")
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    startTime = time.process_time()
    trainGraphs = loadGraphsMayCache(cmd_args.train_dir, False)
    dataReadyTime = time.process_time() - startTime
    log.info('Trainset ready takes %.2fs' % dataReadyTime)

    startTime = time.process_time()
    testGraphs = loadData(cmd_args.test_dir, True)
    dataReadyTime = time.process_time() - startTime
    log.info('Trainset ready takes %.2fs' % dataReadyTime)

    optHp = parseHpTuning(cmd_args.data)
    log.info(f'Optimal hyperparameter setting: {optHp}')
    for (key, val) in optHp.items():
        if key not in gHP:
            log.info(f'Add {key} = {val} to global HP')
        elif gHP[key] != val:
            log.info(f'Replace {key} from {gHP[key]} to {val}')

        gHP[key] = val

    log.info(f'Merged with global hyperparameter setting: {gHP}')
    trainThenPredict(trainGraphs, testGraphs)
