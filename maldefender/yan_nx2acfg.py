#!/usr/bin/python2.7
import glob
import glog as log
import os
import numpy as np
import pandas as pd
import networkx as nx
from networkx import number_of_nodes, adjacency_matrix
from yan_attributes import nodeFeatures
from matplotlib import pyplot as plt
from python23_common import list2Str
from python23_common import neighborsFromAdjacentMatrix


malwareNames = ['Bagle', 'Benign', 'Bifrose', 'Hupigon', 'Koobface',
                'Ldpinch', 'Lmir', 'Rbot', 'Sdbot', 'Swizzor',
                'Vundo', 'Zbot', 'Zlob']
# Map malware names to 1-based integer label
malwareName2Label = {k: v + 1 for (v, k) in enumerate(malwareNames)}
log.info('%s types of CFGs:  %s' % (len(malwareNames), malwareNames))


def plotHistgramInRange(data, left=1, right=200):
    for column in data:
        df = data[column].dropna()
        plt.hist(df, bins=np.arange(left, right, 1), density=False,
                 histtype='step', label=column)

    plt.legend()
    plt.title('Histogram of graph size')
    plt.grid(True)
    plt.show()


def summaryGraphSizes(dirname, malwareNames):
    data = pd.read_csv(dirname + 'graphSizes.csv', header=0)
    cntsByName = {key: data[key].count() for key in malwareNames}
    log.info("Graph size distribution by class:")
    log.info(cntsByName)
    log.info("Total #graphs: %s" % sum(cntsByName.values()))

    maxGraphSizes = data.max()
    log.info("Max graph size for each class:")
    log.info(maxGraphSizes)
    maxGraphSizes = max(data.max())
    log.info("maximum graph size = %s" % maxGraphSizes)
    plotHistgramInRange(data, right=600)


def nxCfg2Acfg(outputDir, malwareDirPrefix):
    """Path name format: class/graphId/pkl_name"""
    logGraphPaths = open(outputDir + '/graph_pathnames.csv', 'w')
    graphSizes = {x: [] for x in malwareNames}
    realPklPath = []
    for malwareDirname in glob.glob(malwareDirPrefix + '/*'):
        log.debug('Enter malware directory %s' % malwareDirname)
        malwareName = malwareDirname.split('/')[-1]
        if malwareName not in malwareNames:
            log.warning('%s not in known malware types' % malwareName)
            continue

        log.info("Search %s gpickles in %s" % (malwareName, malwareDirname))
        pklPaths = glob.glob(malwareDirname + '/*')
        if len(pklPaths) == 0:
            log.warning('Pickle path %s is empty' % pklPaths)

        for pklPath in pklPaths:
            if pklPath[-8:] != '.gpickle':
                log.warning('Ignore %s since it\'s not gpickle file' % pklPath)
                continue

            log.debug('Loading nx.Graph from %s' % pklPath)
            logGraphPaths.write(pklPath + '\n')
            try:
                G = nx.read_gpickle(pklPath)
            except UnicodeDecodeError as e:
                log.error('Decode failed for %s: %s' % (pklPath, e))

            realPklPath.append(pklPath)
            graphSizes[malwareName].append(number_of_nodes(G))

    log.info("Found %s gpickles" % len(realPklPath))
    graphSizesDf = pd.DataFrame.from_dict(graphSizes, orient='index').T
    graphSizesDf.to_csv(outputDir + '/graph_sizes.csv', index=False)
    return realPklPath


def acfg2DgcnnFormat(pklPaths, outputPrefix, outputTxtName='YANACFG'):
    output = open(outputPrefix + '.txt', 'w')
    output.write("%d\n" % len(pklPaths))
    for pklPath in pklPaths:
        G = nx.read_gpickle(pklPath)
        graphId = pklPath.split('/')[-1][:-8]  # ignore '.gpickle'
        malwareName = pklPath.split('/')[-2]
        label = malwareName2Label[malwareName]
        log.info("Process %s as %s(label=%d)" % (pklPath, malwareName, label))

        features, orderedNodes = nodeFeatures(G)
        spAdjacentMat = adjacency_matrix(G, nodelist=orderedNodes)
        output.write("%d %s %s\n" % (features.shape[0], label, graphId))
        indices = neighborsFromAdjacentMatrix(spAdjacentMat)
        for (i, feature) in enumerate(features):
            neighbors = indices[i] if i in indices else []
            nPlusF = list2Str(neighbors, feature)
            output.write("1 %d %s\n" % (len(neighbors), nPlusF))

    output.close()
    log.info('Finish processing %d gpickles' % len(pklPaths))


def iterAllDirectories(cfgDirPrefix, outputDir):
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
        log.info('Make new output dir: %s' % outputDir)

    pklPaths = nxCfg2Acfg(outputDir, cfgDirPrefix)
    acfg2DgcnnFormat(pklPaths, outputDir + '/YANACFG')


if __name__ == '__main__':
    log.setLevel("INFO")
    cfgDirPrefix='../../YANACFG/AllCfg'
    outputDir='../../YANACFG/TrainSet'
    iterAllDirectories(cfgDirPrefix, outputDir)
