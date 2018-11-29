#!/usr/bin/python3.7
"""
Borrowed from and rewritten based on Muhan's pytorch_DGCNN repo at
https://github.com/muhanzhang/pytorch_DGCNN
"""
import argparse
import random
import torch
import math
import numpy as np
import pandas as pd
import glog as log
import networkx as nx
import pickle as pkl
from typing import List, Dict, Set, Tuple
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

cmd_opt = argparse.ArgumentParser(
    description='Argparser for graph_classification')
# Execution options
cmd_opt.add_argument('-mode', default='cpu', help='cpu/gpu')
cmd_opt.add_argument('-gpu_id', default=1,
                     help='on which gpu the model is trained, in {0, 1, 2, 3}')
cmd_opt.add_argument('-gm', default='dgcnn',
                     help='Type of graph model: dgcnn|mean_field|loopy_bp')
cmd_opt.add_argument('-data', default=None, help='txt data name')
cmd_opt.add_argument('-train_dir', default='../TrainSet',
                     help='folder for trainset')
cmd_opt.add_argument('-test_dir', default='../TestSet',
                     help='folder for testset')
cmd_opt.add_argument('-seed', type=int, default=1, help='seed')
cmd_opt.add_argument('-mlp_type', type=str, default='vanilla',
                     help='Type of regression MLP: vanilla|rap|vgg')
cmd_opt.add_argument('-use_cached_data', type=str, default='False',
                     help='if use previously cached dataset')
cmd_opt.add_argument('-cache_path', type=str, default='cached_graphs',
                     help='which cached data to use or write new data to')
cmd_opt.add_argument('-norm_op', type=str, default='min_max',
                     help='scaling operation: [min_max, zero_mean, none]')
cmd_opt.add_argument('-norm_path', type=str, default='norm',
                     help='which cached data to use or write new data to')
cmd_opt.add_argument('-hp_path', type=str, default='none',
                     help='raw hyperparameter values')
gHP = dict()
cmd_args, _ = cmd_opt.parse_known_args()
cmd_args.use_cached_data = (cmd_args.use_cached_data == "True")
log.info("Parsed cmdline arguments: %s" % cmd_args)
testBinaryIds = {
    'cqdUoQDaZfGkt5ilBe7n': 0,
    'jgOs7KiB0aTEzvSUJVPp': 1,
    '6RQtx0X42zOelTDaZnvi': 2,
    'HaTioeY3kbvJW2LXtOwF': 3,
    'Fnda3PuqJT6Ep5vjOWCk': 4,
    'exGy3iaKJmRprdHcB0NO': 5,
    'bZz2OoQmqx0PdGBhaHKk': 6,
    '0Q4ALVSRnlHUBjyOb1sw': 7,
    'hIkK1vBdj9fDJPcUWzA8': 8,
}


class S2VGraph(object):
    def __init__(self, binaryId: str, g: nx.Graph, label: int,
                 node_tags: int = None, node_features=None):
        """
        g: a networkx graph
        label: an integer graph label
        node_tags: a list of integer node tags
        node_features: a numpy array of continuous node features
        """
        self.bId = binaryId
        self.num_nodes = len(node_tags)
        self.node_tags = node_tags
        self.label = None if label == '?' else label
        self.node_features = node_features  # nparray (node_num * feature_dim)
        self.degs = list(dict(g.degree).values())

        if g.number_of_edges() != 0:
            x, y = zip(*g.edges())
            self.num_edges = len(x)
            self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
            self.edge_pairs[:, 0] = x
            self.edge_pairs[:, 1] = y
            self.edge_pairs = self.edge_pairs.flatten()
        else:
            self.num_edges = 0
            self.edge_pairs = np.array([])
            log.warning(f'{binaryId} has no edge')
            log.debug(f'{binaryId} #nodes: {self.num_nodes}, label: {label}')


def filterOutNoEdgeGraphs(graphs: List[S2VGraph]) -> List[S2VGraph]:
    result = list(filter(lambda x: x.num_edges > 0, graphs))
    numFiltered = len(result) - len(graphs)
    log.info(f'Skip {numFiltered} graphs that have no edge')
    return result


def loadData(dataDir: str, isTestSet: bool = False) -> List[S2VGraph]:
    log.info('Loading data as list of S2VGraph(s)')
    gList: List[S2VGraph] = []
    tagDict: Dict[str, int] = {}   # mapping node tag to 0-based int
    numClasses = 0
    f = open('%s/%s.txt' % (dataDir, cmd_args.data), 'r')
    numGraphs = int(f.readline().strip())
    maxVector, minVector, avgVector, stdVector = None, None, None, None

    for i in range(numGraphs):
        row = f.readline().strip().split()
        numNodes, label, bId = int(row[0]), row[1], row[2]
        if not isTestSet:
            label = int(row[1]) - 1
            numClasses = max(numClasses, label + 1)

        if bId in testBinaryIds:
            log.debug(f'[Test {bId}]{label} =? {testBinaryIds[bId]}')
            assert label == testBinaryIds[bId]

        g = nx.Graph()
        nodeTags = []
        nodeFeatures = []
        featDim = None
        for j in range(numNodes):
            row = f.readline().strip().split()
            g.add_node(j)
            nodeTag = row[0]
            featInitIdx = int(row[1]) + 2
            if featInitIdx == len(row):
                # no node attributes
                neighbors = [int(w) for w in row[2:]]
                features = None
            else:
                neighbors = [int(w) for w in row[2: featInitIdx]]
                features = np.array([float(w) for w in row[featInitIdx:]])

            if not nodeTag in tagDict:
                mapped = len(tagDict)
                tagDict[nodeTag] = mapped

            nodeTags.append(tagDict[nodeTag])

            if features is not None:
                nodeFeatures.append(features)
                if maxVector is not None:
                    maxVector = np.maximum(features, maxVector)
                    minVector = np.minimum(features, minVector)
                    avgVector = np.add(features, avgVector)
                    stdVector = np.add(np.square(features), stdVector)
                else:
                    maxVector = features
                    minVector = features
                    avgVector = features
                    stdVector = np.square(features)

            for k in neighbors:
                g.add_edge(j, k)

        # Convert nodeFeatures to np matrix
        if len(nodeFeatures) > 0:
            nodeFeatures = np.stack(nodeFeatures)
        else:
            nodeFeatures = None

        assert g.number_of_nodes() == numNodes
        gList.append(S2VGraph(bId, g, label, nodeTags, nodeFeatures))

    totalNumNodes = np.sum([g.num_nodes for g in gList])
    avgVector = avgVector / totalNumNodes
    stdVector = np.sqrt(stdVector / totalNumNodes - np.square(avgVector))
    cachePath = cmd_args.norm_path + '_test' if isTestSet else cmd_args.norm_path
    log.info(f'Dumping min/max/avg/std vectors to {cachePath}')
    log.debug(f'Max feature vector: {list(maxVector)}')
    log.debug(f'Min feature vector: {list(minVector)}')
    log.debug(f'Avg feature vector: {list(avgVector)}')
    log.debug(f'Std feature vector: {list(stdVector)}')
    cacheFile = open(cachePath + '.pkl', 'wb')
    norm = {'minVector': minVector, 'maxVector': maxVector,
            'avgVector': avgVector, 'stdVector': stdVector}
    pkl.dump(norm, cacheFile)
    cacheFile.close()

    gHP['numClasses'] = numClasses
    gHP['nodeTagDim'] = len(tagDict)
    gHP['featureDim'] = 0
    if nodeFeatures is not None:
        gHP['featureDim'] = nodeFeatures.shape[1]

    log.info(f'# graphs: {len(gList)}')
    log.info(f'# classes: {gHP["numClasses"]}')
    log.info(f'node tag dimension: {gHP["nodeTagDim"]}')
    log.info(f'node feature dimension: {gHP["featureDim"]}')
    return gList


def loadGraphsMayCache(dataDir: str, isTestSet: bool = False) -> List[S2VGraph]:
    """ Enhance loadData() with caching. """
    cachePath = cmd_args.cache_path
    if isTestSet == True:
        cachePath += '_test'

    if cmd_args.use_cached_data:
        log.info(f"Loading cached dataset from {cachePath}")
        cacheFile = open(cachePath + '.pkl', 'rb')
        dataset = pkl.load(cacheFile)
        gHP['numClasses'] = dataset['numClasses']
        gHP['featureDim'] = dataset['featureDim']
        gHP['nodeTagDim'] = dataset['nodeTagDim']
        graphs = dataset['graphs']
        cacheFile.close()
    else:
        graphs = loadData(dataDir, isTestSet)
        log.info(f"Dumping cached dataset to {cachePath}")
        cacheFile = open(cachePath + '.pkl', 'wb')
        dataset = {}
        dataset['numClasses'] = gHP['numClasses']
        dataset['featureDim'] = gHP['featureDim']
        dataset['nodeTagDim'] = gHP['nodeTagDim']
        dataset['graphs'] = list(graphs)
        pkl.dump(dataset, cacheFile)
        cacheFile.close()

    return graphs


def kFoldSplit(k: int, graphs: List[S2VGraph]) -> List[List[S2VGraph]]:
    random.shuffle(graphs)
    results = []
    share = math.ceil(len(graphs) / k)
    for i in range(k):
        start = i * share
        end = min((i + 1) * share, len(graphs))
        results.append(graphs[start: end])
        log.info(f'Fold {i + 1} range from {start} to {end - 1}')

    return results


def loadNormVectors(graphs, isTestSet: bool = False) -> Tuple[List[float]]:
    cachePath = cmd_args.norm_path + '_test' if isTestSet else cmd_args.norm_path
    log.info(f'Loading pre-computed min/max/avg/std vectors from {cachePath}')
    cacheFile = open(cachePath + '.pkl', 'rb')
    norm = pkl.load(cacheFile)
    maxVector = norm['maxVector']
    minVector = norm['minVector']
    avgVector = norm['avgVector']
    stdVector = norm['stdVector']
    cacheFile.close()
    return (maxVector, minVector, avgVector, stdVector)


def normalizeFeatures(graphs: List[S2VGraph],
                      isTestSet: bool = False,
                      operation: str = 'min_max') -> List[List[float]]:
    normVectors = loadNormVectors(graphs, isTestSet)
    maxVector, minVector, avgVector, stdVector = normVectors
    log.info(f'Max feature vector: {list(maxVector)}')
    log.info(f'Min feature vector: {list(minVector)}')
    log.info(f'Avg feature vector: {list(avgVector)}')
    log.info(f'Std feature vector: {list(stdVector)}')
    diff = [x - y for (x, y) in zip(maxVector, minVector)]
    diffVector = [1 if math.isclose(x, 0.0) else x for x in diff]
    stdVector = [1 if math.isclose(x, 0.0) else x for x in stdVector]
    reduceDims = []
    for (i, pair) in enumerate(zip(maxVector, minVector)):
        if math.isclose(pair[0], pair[1]):
            reduceDims.append(i)

    log.info(f'Delete constant features: {reduceDims}')
    for g in graphs:
        if operation == 'min_max':
            g.node_features = (g.node_features - minVector) / diffVector
        elif operation == 'zero_mean':
            g.node_features = (g.node_features - avgVector) / stdVector
        else:
            log.debug(f'Unknown operation: {operation}')

        for i in reduceDims:
            g.node_features = np.delete(g.node_features, i, axis=1)

    gHP['featureDim'] -= len(reduceDims)
    return [maxVector, minVector, avgVector, stdVector]


def computePrScores(pred, labels, prefix: str = 'train',
                    avgMethod: str ='weighted',
                    store=False) -> Dict[str, float]:
    scores = {}
    if cmd_args.data == 'MSACFG':
        scores['family'] = [
            'Ramnit', 'Lollipop', 'KeliVer3', 'Vundo', 'Simda', 'Tracur',
            'KeliVer1', 'Obf.ACY', 'Gatak'
        ]
    elif cmd_args.data == 'YANACFG':
        scores['family'] = [
            'Bagle',  'Benign', 'Bifrose', 'Hupigon', 'Koobface', 'Ldpinch',
            'Lmir','Rbot', 'Sdbot', 'Swizzor', 'Vundo', 'Zbot', 'Zlob'
        ]

    scores['Precision'] = precision_score(labels, pred, average=avgMethod)
    scores['Recall'] = recall_score(labels, pred, average=avgMethod)
    scores['F1'] = f1_score(labels, pred, average=avgMethod)
    if store:
        df = pd.DataFrame.from_dict(scores)
        file = open('%s_%s_pr_scores.csv' % (cmd_args.data, prefix), 'w')
        df.to_csv(file, index=('family' in scores), float_format='%.6f')
        file.close()

    return scores


def storeConfusionMatrix(pred, labels, prefix) -> None:
    cm = confusion_matrix(labels, pred)
    np.savetxt('%s_%s_confusion_matrix.txt' % (cmd_args.data, prefix), cm,
               fmt='%4d', delimiter=' ')


def storeEmbedding(classifier, graphs, prefix, sample_size=100) -> None:
    if len(graphs) > sample_size:
        sample_idx = np.random.randint(0, len(graphs), sample_size)
        graphs = [graphs[i] for i in sample_idx]

    emb = classifier.embedding(graphs)
    emb = emb.data.cpu().numpy()
    labels = [g.label for g in graphs]
    np.savetxt('%s_%s_embedding.txt' % (cmd_args.data, prefix),
               emb, fmt='%8.8f')
    np.savetxt('%s_%s_embedding_label.txt' % (cmd_args.data, prefix),
               labels, fmt='%d')


def balancedSampling(graphs, neg_ratio=3):
    graph_labels = np.array([g.label for g in graphs])
    pos_indices = np.where(graph_labels == 1)[0]
    neg_indices = np.where(graph_labels == 0)[0]
    log.info('In given dataset #pos = %s, #neg = %s' %
             (pos_indices.size, neg_indices.size))
    upper_bound = min(pos_indices.size * neg_ratio, neg_indices.size)
    sampled_pos = [graphs[i] for i in pos_indices]
    sampled_neg = [graphs[i] for i in
                   np.random.choice(neg_indices, upper_bound, replace=False)]
    sampled = sampled_pos + sampled_neg
    np.random.shuffle(sampled)
    log.info("#Balance sampled graphs = %d" % len(sampled))
    return sampled


def toOnehot(indices, num_classes):
    onehot = torch.zeros(indices.size(0), num_classes, device=indices.device)
    return onehot.scatter_(1, indices.unsqueeze(1), 1)


def getLearningRate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
