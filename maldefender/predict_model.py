#!/usr/bin/python3.7
import sys
import os
import torch
import math
import random
import time
import glog as log
import numpy as np
from typing import Dict, List
from ml_utils import cmd_args, gHP, S2VGraph
from ml_utils import loadGraphsMayCache, normalizeFeatures
from ml_utils import filterOutNoEdgeGraphs
from ml_utils import loadModel, saveModel, decideHyperparameters
from e2e_model import Classifier, predictDataset


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


def exportPredictions(graphs: List[S2VGraph], predProb):
    log.debug(f'Export {len(predProb)} predictions for {len(graphs)} graphs')
    assert len(graphs) == len(predProb)
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

    output = open(cmd_args.test_dir + '/submission.csv', 'w')
    output.write('"Id","Prediction1","Prediction2","Prediction3","Prediction4","Prediction5","Prediction6","Prediction7","Prediction8","Prediction9"\n')
    for (i, g) in enumerate(graphs):
        assert len(predProb[i]) == gHP['numClasses']
        assert g.bId not in emptyIds

        prob = ["%.8f" % p for p in predProb[i]]
        elems = ['"' + str(g.bId) + '"'] + prob
        output.write('%s\n' % ",".join(elems))

    guessProb = [1.0 / 3, 0.0, 0.0, 0.0, 1.0 / 3, 1.0 / 3, 0.0, 0.0, 0.0]
    for (i, bId) in enumerate(emptyIds):
        prob = ["%.8f" % p for p in guessProb]
        elems = ['"' + bId + '"'] + prob
        output.write('%s\n' % ",".join(elems))

    output.close()


def testWithModel(testGraphs: List[S2VGraph]) -> None:
    classifier = Classifier()
    loadModel(classifier)
    if cmd_args.mode == 'gpu':
        classifier = classifier.cuda()

    classifier.eval()
    startTime = time.process_time()
    testPredProb = predictDataset(testGraphs, classifier)
    log.info(f'Net test time = {time.process_time() - startTime} seconds')
    exportPredictions(testGraphs, testPredProb)


if __name__ == '__main__':
    log.setLevel("INFO")
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    if cmd_args.data == 'YANACFG':
        log.warning(f'No testset for YANACFG data')
    else:
        startTime = time.process_time()
        testGraphs = loadGraphsMayCache(cmd_args.test_dir, isTestSet=True)
        normalizeFeatures(testGraphs, isTestSet=True,
                          operation=cmd_args.norm_op)
        dataReadyTime = time.process_time() - startTime
        log.info('Testset ready takes %.2fs' % dataReadyTime)

        trainGraphs = loadGraphsMayCache(cmd_args.train_dir, isTestSet=False)
        trainGraphs = filterOutNoEdgeGraphs(trainGraphs)
        decideHyperparameters(trainGraphs)
        testWithModel(testGraphs)
