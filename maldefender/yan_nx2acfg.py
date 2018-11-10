#/usr/bin/python3.7
import glob
import glog as log
import os
import pickle as pkl
import numpy as np
import scipy as sp
import pandas as pd
import networkx as nx
from typing import List, Dict
from collections import Counter
from networkx import number_of_nodes, adjacency_matrix
from yan_attributes import nodeFeatures
from matplotlib import pyplot as plt


def plotHistgramInRange(data, left=1, right=200):
    for column in data:
        df = data[column].dropna()
        maxNumNodes = df.max()
        plt.hist(df, bins=np.arange(left, right, 1), density=False,
                 histtype='step', label=column)

    plt.legend()
    plt.title('Histogram of graph size')
    plt.grid(True)
    plt.show()


def summaryGraphSizes(dirname: str, malwareNames: List[str]):
    data = pd.read_csv(dirname + 'graphSizes.csv', header=0)
    cntsByName = {key: data[key].count() for key in malwareNames}
    log.info("Graph size distribution by class:")
    log.info(cntsByName)
    log.info(f"Total #graphs: {sum(cntsByName.values())}")

    maxGraphSizes = data.max()
    log.info("Max graph size for each class:")
    log.info(maxGraphSizes)
    maxGraphSizes = max(data.max())
    log.info(f"maximum graph size = {maxGraphSizes}")
    plotHistgramInRange(data, right=600)


def nxCfg2Acfg(outputDir: str, malwareDirPrefix: str) -> None:
    """Path name format: class/graphId/pkl_name"""
    logGraphPaths = open(outputDir + '/graph_pathnames.csv', 'w')
    graphSizes = {x: [] for x in malwareNames}
    total = 0
    for malwareDirname in glob.glob(malwareDirPrefix + '/*'):
        log.debug(f'Enter malware directory {malwareDirname}')
        malwareName = malwareDirname.split('/')[-1]
        if malwareName not in malwareNames:
            log.warning(f'{malwareName} not in known malware types')
            continue

        log.info(f"Processing {malwareName} CFGs")
        pklPaths = glob.glob(malwareDirname + '/*')

        if len(pklPaths) == 0:
            log.warning(f'{pklPaths} is empty')

        for pklPath in pklPaths[:4]:
            if pklPath[-8:] != '.gpickle':
                log.warning(f'{pklPath} is not gpickle file')
                continue

            log.debug(f'Loading nx.Graph from {pklPath}')
            total += 1
            logGraphPaths.write(pklPath + '\n')

            try:
                G = nx.read_gpickle(pklPath)
            except UnicodeDecodeError as e:
                log.error(f'Decode failed for gpickle file {pklPath}: {e}')
                continue

            graphSizes[malwareName].append(number_of_nodes(G))
            graphId = pklPath.split('/')[-1][:-8] # ignore '.gpickle'
            prefix = outputDir + '/' + graphId

            log.debug(f'Writing feature/label/adj for {graphId}')
            features = nodeFeatures(G)
            np.savetxt(prefix + '.features.txt', features, fmt="%d")
            np.savetxt(prefix + '.label.txt', [malwareName], fmt="%s")
            sp.sparse.save_npz(prefix + '.adjacent', adjacency_matrix(G))

    log.info(f"Processed {total} CFGs")
    graphSizesDf = pd.DataFrame.from_dict(graphSizes, orient='index').T
    graphSizesDf.to_csv(outputDir + '/graph_sizes.csv', index=False)


def iterAllDirectories(cfgDirPrefix: str = '../../IdaProCfg/AllCfg',
                       outputDir: str = '../../IdaProCfg/AllAcfg'):
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
        log.info(f'Make new output dir {outputDir}')

    nxCfg2Acfg(outputDir, cfgDirPrefix)


if __name__ == '__main__':
    malwareNames = ['Bagle', 'Benign', 'Bifrose', 'Hupigon', 'Koobface',
                    'Ldpinch', 'Lmir', 'Rbot', 'Sdbot', 'Swizzor',
                    'Vundo', 'Zbot', 'Zlob']
    log.info(f'{len(malwareNames)} types of CFGs:  + {malwareNames}')
    iterAllDirectories()
