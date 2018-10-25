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
from e2e_model import Classifier, loopDataset, predictDataset
from hyperparameters import parseHpTuning


def exportPredictions(graphs, predProb):
    log.debug(f'Export {len(predProb)} predictions for {len(graphs)} graphs')
    assert len(graphs) == len(predProb)
    output = open(cmd_args.test_dir + '/submission.csv', 'w')
    output.write('"Id","Prediction1","Prediction2","Prediction3","Prediction4","Prediction5","Prediction6","Prediction7","Prediction8","Prediction9"\n')
    for (i, g) in enumerate(graphs):
        prob = ["%.8f" % p for p in predProb[i]]
        elems = ['"' + str(g.bId) + '"'] + prob
        output.write('%s\n' % ",".join(elems))

    emptyIds = ["ZOtweKduNMynmpiG4brh",
                "y5l1PF7qGvsQSDgmRkKn",
                "TroLhDaQ2qkKe4XmtPEd",
                "spRNUv6MFb8ihB9JXk5r",
                "VZ2rzALmJS38uIG5wR1X",
                "N2TJvMjcebxGKq1YDC9k",
                "xYr76sCtHa2dD48FiGkK",
                "YvpzOeBSu7Tmia3wKlLf",
                "W8VtX0E95TSzxJuGqiI4",
                "uzRUIAil6dVwWsCvhbKD",
                "W8aI0V7G5lFTpOgSvjf6",
                "pLY05AFladXWQ9fDZnhb",
                "QpHV1IWD72EnAyB3FowM",
    ]
    guessProb = [1.0 / 3, 0.0, 0.0, 0.0, 1.0 / 3, 1.0 / 3, 0.0, 0.0, 0.0]
    for (i, bId) in enumerate(emptyIds):
        prob = ["%.8f" % p for p in guessProb]
        elems = ['"' + bId + '"'] + prob
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

    log.info(f'Net training time = {time.process_time() - startTime} seconds')
    storeConfusionMatrix(trainPred, trainLabels, 'train')
    result = {
        'TrainLoss': trainLossHist,
        'TrainAccu': trainAccuHist,
        'TrainPrec': trainPrecHist,
        'TrainRecl': trainRecallHist,
        'TrainF1': trainF1Hist,
    }
    log.info(f'Model trainset performance: {result}')

    classifier.eval()
    startTime = time.process_time()
    testPredProb = predictDataset(testGraphs, classifier)
    log.info(f'Net testing time = {time.process_time() - startTime} seconds')
    exportPredictions(testGraphs, testPredProb)

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
            log.debug(f'Add {key} = {val} to global HP')
        elif gHP[key] != val:
            log.debug(f'Replace {key} from {gHP[key]} to {val}')

        gHP[key] = val

    log.info(f'Merged with global hyperparameter setting: {gHP}')
    trainThenPredict(trainGraphs, testGraphs)
