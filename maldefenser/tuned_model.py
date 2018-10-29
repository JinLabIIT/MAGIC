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
from ml_utils import cmd_args, gHP, storeConfusionMatrix, kFoldSplit
from ml_utils import computePrScores, loadGraphsMayCache, loadData
from e2e_model import Classifier, loopDataset, predictDataset
from hyperparameters import parseHpTuning


def exportRandomPredictions(graphs):
    output = open(cmd_args.test_dir + '/randomSubmission.csv', 'w')
    output.write('"Id","Prediction1","Prediction2","Prediction3","Prediction4","Prediction5","Prediction6","Prediction7","Prediction8","Prediction9"\n')
    guessProb = ["%.8f" % (1.0 / 9) for _ in range(9)]
    for (i, g) in enumerate(graphs):
        elems = ['"' + str(g.bId) + '"'] + guessProb
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
    for (i, bId) in enumerate(emptyIds):
        elems = ['"' + str(bId) + '"'] + guessProb
        output.write('%s\n' % ",".join(elems))

    output.close()


def exportPredictions(graphs, predProb, epoch=None):
    log.debug(f'Export {len(predProb)} predictions for {len(graphs)} graphs')
    assert len(graphs) == len(predProb)
    if epoch is None:
        output = open(cmd_args.test_dir + '/submission.csv', 'w')
    else:
        output = open(cmd_args.test_dir + '/submissionE%d.csv' % epoch, 'w')

    output.write('"Id","Prediction1","Prediction2","Prediction3","Prediction4","Prediction5","Prediction6","Prediction7","Prediction8","Prediction9"\n')
    for (i, g) in enumerate(graphs):
        assert len(predProb[i]) == gHP['numClasses']
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


def trainThenPredict(trainSet, testGraphs) -> Dict[str, float]:
    classifier = Classifier()
    if cmd_args.mode == 'gpu':
        classifier = classifier.cuda()

    optimizer = optim.Adam(classifier.parameters(), lr=gHP['lr'])

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

        if epoch % 10 == 0:
            classifier.eval()
            testPredProb = predictDataset(testGraphs, classifier)
            exportPredictions(testGraphs, testPredProb, epoch)

    log.info(f'Net training time = {time.process_time() - startTime} seconds')
    storeConfusionMatrix(trainPred, trainLabels, 'train')
    storeConfusionMatrix(validPred, validLabels, 'valid')
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
    log.info(f'Model trainset + validset performance:\n{result}')
    df = pd.DataFrame.from_dict(result)
    histFile = open('%sPred.hist' % cmd_args.data, 'w')
    histFile.write("# %s\n" % str(gHP))
    df.to_csv(histFile, index_label='Epoch', float_format='%.6f')
    histFile.close()

    classifier.eval()
    startTime = time.process_time()
    testPredProb = predictDataset(testGraphs, classifier)
    log.info(f'Net testing time = {time.process_time() - startTime} seconds')
    exportPredictions(testGraphs, testPredProb)


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
    testGraphs = loadGraphsMayCache(cmd_args.test_dir, True)
    dataReadyTime = time.process_time() - startTime
    log.info('Testset ready takes %.2fs' % dataReadyTime)

    optHp = parseHpTuning(cmd_args.data)
    log.info(f'Optimal hyperparameter setting: {optHp}')
    for (key, val) in optHp.items():
        if key not in gHP:
            log.debug(f'Add {key} = {val} to global HP')
        elif gHP[key] != val:
            log.debug(f'Replace {key} from {gHP[key]} to {val}')

        gHP[key] = val

    log.info(f'Global hyperparameter setting: {gHP}')
    trainThenPredict(trainGraphs, testGraphs)
