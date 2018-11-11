#!/usr/bin/python2.7

import glob
import glog as log
import os
import numpy as np
import scipy as sp
import pandas as pd
import networkx as nx
from networkx import number_of_nodes, adjacency_matrix
from yan_attributes import nodeFeatures
from matplotlib import pyplot as plt


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
    total = 0
    for malwareDirname in glob.glob(malwareDirPrefix + '/*'):
        log.debug('Enter malware directory %s' % malwareDirname)
        malwareName = malwareDirname.split('/')[-1]
        if malwareName not in malwareNames:
            log.warning('%s not in known malware types' % malwareName)
            continue

        log.info("Processing %s CFGs" % malwareName)
        pklPaths = glob.glob(malwareDirname + '/*')

        if len(pklPaths) == 0:
            log.warning('%s is empty' % pklPaths)

        for pklPath in pklPaths[:4]:
            if pklPath[-8:] != '.gpickle':
                log.warning('%s is not gpickle file' % pklPath)
                continue

            log.debug('Loading nx.Graph from %s' % pklPath)
            total += 1
            logGraphPaths.write(pklPath + '\n')

            try:
                G = nx.read_gpickle(pklPath)
            except UnicodeDecodeError as e:
                log.error('Decode failed for %s : %s' % (pklPath, e))
                continue

            graphSizes[malwareName].append(number_of_nodes(G))
            graphId = pklPath.split('/')[-1][:-8]  # ignore '.gpickle'
            prefix = outputDir + '/' + graphId

            log.debug('Writing feature/label/adj for %s' % graphId)
            features = nodeFeatures(G)
            np.savetxt(prefix + '.features.txt', features, fmt="%d")
            np.savetxt(prefix + '.label.txt', [malwareName], fmt="%s")
            sp.sparse.save_npz(prefix + '.adjacent', adjacency_matrix(G))

    log.info("Processed %s CFGs" % total)
    graphSizesDf = pd.DataFrame.from_dict(graphSizes, orient='index').T
    graphSizesDf.to_csv(outputDir + '/graph_sizes.csv', index=False)


def iterAllDirectories(cfgDirPrefix='../../IdaProCfg/AllCfg',
                       outputDir='../../IdaProCfg/AllAcfg'):
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
        log.info('Make new output dir: ' % outputDir)

    nxCfg2Acfg(outputDir, cfgDirPrefix)


if __name__ == '__main__':
    malwareNames = ['Bagle', 'Benign', 'Bifrose', 'Hupigon', 'Koobface',
                    'Ldpinch', 'Lmir', 'Rbot', 'Sdbot', 'Swizzor',
                    'Vundo', 'Zbot', 'Zlob']
    log.info('%s types of CFGs:  %s' % (len(malwareNames), malwareNames))
    iterAllDirectories()
