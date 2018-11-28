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
from ml_utils import computePrScores, loadGraphsMayCache, filterOutNoEdgeGraphs
from ml_utils import storeConfusionMatrix, normalizeFeatures
from e2e_model import Classifier, loopDataset, predictDataset
from hyperparameters import parseHpTuning, HyperParameterIterator
from torch.optim.lr_scheduler import ReduceLROnPlateau


def exportRandomPredictions(graphs: List[S2VGraph]):
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


def exportPredictions(graphs: List[S2VGraph], predProb, epoch: int = None):
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


def testWithModel(classifier, testGraphs: List[S2VGraph]) -> None:
    if testGraphs is None:
        return

    classifier.eval()
    startTime = time.process_time()
    testPredProb = predictDataset(testGraphs, classifier)
    log.info(f'Net testing time = {time.process_time() - startTime} seconds')
    exportPredictions(testGraphs, testPredProb)


def trainThenPredict(trainSet: List[S2VGraph],
                     testGraphs: List[S2VGraph],
                     numEpochs: int) -> Dict[str, float]:
    classifier = Classifier()
    log.info(f'Global hyperparameter setting: {gHP}')
    if cmd_args.mode == 'gpu':
        classifier = classifier.cuda()

    optimizer = optim.SGD(classifier.parameters(), momentum=0.9, lr=gHP['lr'],
                          weight_decay=gHP['l2RegFactor'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5,
                                  patience=2, verbose=True)
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
        classifier.train()
        avgScore, trainPred, trainLabels = loopDataset(
            trainGraphs, classifier, trainIndices, optimizer=optimizer)
        prScore = computePrScores(trainPred, trainLabels, 'train')
        print('\033[92mTrain epoch %d: l %.5f\033[0m' % (e, avgScore[0]))
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
        if e % 10 == 0:
            testWithModel(classifier, testGraphs)

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
    histFile = open('%sPredHist.csv' % cmd_args.data, 'w')
    histFile.write("# %s\n" % str(gHP))
    df.to_csv(histFile, index_label='Epoch', float_format='%.6f')
    histFile.close()
    testWithModel(classifier, testGraphs)


def decideHyperparameters(graphs: List[S2VGraph]) -> int:
    if cmd_args.hp_path == 'none':
        log.info(f'Using previous CV results to decide hyperparameters')
        optHp = parseHpTuning(cmd_args.data)
        log.info(f'Optimal hyperparameter setting: {optHp}')
        for (key, val) in optHp.items():
            if key not in gHP:
                log.debug(f'Add {key} = {val} to global HP')
            elif gHP[key] != val:
                log.debug(f'Replace {key} from {gHP[key]} to {val}')

            gHP[key] = val

        return gHP['optNumEpochs']
    else:
        log.info(f'Using 1st hyperparameter setting from {cmd_args.hp_path}')
        hpIter = HyperParameterIterator(cmd_args.hp_path)
        hp = next(hpIter)
        for (key, val) in hp.items():
            gHP[key] = val

        numNodesList = sorted([g.num_nodes for g in graphs])
        idx = int(math.ceil(hp['poolingRatio'] * len(graphs))) - 1
        gHP['poolingK'] = numNodesList[idx]
        return gHP['numEpochs']


if __name__ == '__main__':
    log.setLevel("INFO")
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    if cmd_args.data == 'YANACFG':
        log.warning(f'No testset for YANACFG data')
        testGraphs = None
    else:
        startTime = time.process_time()
        testGraphs = loadGraphsMayCache(cmd_args.test_dir, True)
        normalizeFeatures(testGraphs, isTestSet=True, operation='min_max')
        dataReadyTime = time.process_time() - startTime
        log.info('Testset ready takes %.2fs' % dataReadyTime)

    startTime = time.process_time()
    trainGraphs = loadGraphsMayCache(cmd_args.train_dir, False)
    normalizeFeatures(trainGraphs, isTestSet=False, operation='min_max')
    trainGraphs = filterOutNoEdgeGraphs(trainGraphs)
    dataReadyTime = time.process_time() - startTime
    log.info('Trainset ready takes %.2fs' % dataReadyTime)

    numEpochs = decideHyperparameters(trainGraphs)
    trainThenPredict(trainGraphs, testGraphs, numEpochs)
