#!/usr/bin/python3.7
import time
import glog as log
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
from typing import Dict, List
from ml_utils import cmd_args, S2VGraph
from ml_utils import loadGraphsMayCache, filterOutNoEdgeGraphs


font = {'family': 'normal', 'size': 28}
matplotlib.rc('font', **font)
matplotlib.rcParams["figure.figsize"] = [14, 10]


def familyDistribution(data: str, familyLabel: Dict[int, str]):
    startTime = time.process_time()
    cmd_args.use_cached_data = True
    cmd_args.cache_path = 'cached_%s_graphs' % data.lower()
    trainGraphs = loadGraphsMayCache(cmd_args.train_dir, False)
    trainGraphs = filterOutNoEdgeGraphs(trainGraphs)
    dataReadyTime = time.process_time() - startTime
    log.info('Trainset ready takes %.2fs' % dataReadyTime)

    familyDist = {name: 0 for name in familyLabel.values()}
    log.info(f'Initial family dist: {familyDist}')
    for graph in trainGraphs:
        name = familyLabel[graph.label]
        familyDist[name] += 1

    df = pd.DataFrame.from_dict(familyDist, orient='index', columns=['Cnt'])
    file = open('%s_train_label_distribution.csv' % data, 'w')
    df.to_csv(file, float_format='%.1f')
    file.close()

    log.info(f'Family dist for {data}: {familyDist}')
    return familyDist


def plotFamilyDist(dist: Dict[str, float], data: str):
    indices = range(0, len(dist))
    fig, ax = plt.subplots()
    barWidth = 0.65

    rects = ax.bar(indices, dist.values(), barWidth,
                   color='white', hatch='/', edgecolor='black')
    ax.set_ylabel('#Instances in Dataset')
    ax.set_xticks(indices)
    ax.set_xticklabels(dist.keys(), rotation=45)
    plt.subplots_adjust(left=0.14, bottom=0.17, right=0.98, top=0.99)
    plt.grid(ls='-.')
    plt.savefig('%sLabelDist.pdf' % data, format='pdf')
    log.info(f'Figure for {data} saved to {data}LabelDist.pdf')
    # plt.show()


if __name__ == '__main__':
    log.setLevel("INFO")
    msNames = [
        'Ramnit', 'Lollipop', 'KeliVer3', 'Vundo', 'Simda', 'Tracur',
        'KeliVer1', 'Obf.ACY', 'Gatak'
    ]
    yanNames = [
        'Bagle', 'Benign', 'Bifrose', 'Hupigon', 'Koobface', 'Ldpinch', 'Lmir',
        'Rbot', 'Sdbot', 'Swizzor', 'Vundo', 'Zbot', 'Zlob'
    ]
    msLabel = {label: name for (label, name) in enumerate(msNames)}
    yanLabel = {label: name for (label, name) in enumerate(yanNames)}

    msDist = familyDistribution('MSACFG', msLabel)
    plotFamilyDist(msDist, 'MsAcfg')
    yanDist = familyDistribution('YANACFG', yanLabel)
    plotFamilyDist(yanDist, 'YanAcfg')
