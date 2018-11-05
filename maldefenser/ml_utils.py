#!/usr/bin/python3.7
import argparse
import random
import torch
import math
import numpy as np
import pandas as pd
import glog as log
import networkx as nx
import pickle as pkl
from typing import List, Dict, Set
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

cmd_opt = argparse.ArgumentParser(
    description='Argparser for graph_classification')
# Execution options
cmd_opt.add_argument('-mode', default='cpu', help='cpu/gpu')
cmd_opt.add_argument('-gpu_id', default=1,
                     help='on which gpu the model is trained, in {0, 1, 2, 3}')
cmd_opt.add_argument('-gm', default='mean_field', help='mean_field/loopy_bp')
cmd_opt.add_argument('-data', default=None, help='txt data name')
cmd_opt.add_argument('-train_dir', default='../TrainSet',
                     help='folder for trainset')
cmd_opt.add_argument('-test_dir', default='../TestSet',
                     help='folder for testset')
cmd_opt.add_argument('-seed', type=int, default=1, help='seed')
cmd_opt.add_argument('-mlp_type', type=str, default='vanilla',
                     help='Type of regression MLP: RAP or vanilla')
cmd_opt.add_argument('-use_cached_data', type=str, default='False',
                     help='whether to use previously cached dataset')
cmd_opt.add_argument('-cache_path', type=str, default='cached_graphs.pkl',
                     help='which cached data to use')
cmd_opt.add_argument('-hp_path', type=str, default='hp.txt',
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
            log.warning(f'ACFG-{binaryId} has no edge')
            log.debug(f'{binaryId} #nodes: {self.num_nodes}, label: {label}')


def loadData(dataDir: str, isTestSet: bool = False) -> List[S2VGraph]:
    log.info('Loading data as list of S2VGraph(s)')
    gList: List[S2VGraph] = []
    tagDict: Dict[str, int] = {}   # mapping node tag to 0-based int
    numClasses = 0
    f = open('%s/%s.txt' % (dataDir, cmd_args.data), 'r')
    numGraphs = int(f.readline().strip())

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

            for k in neighbors:
                g.add_edge(j, k)

        # Convert nodeFeatures to np matrix
        if len(nodeFeatures) > 0:
            nodeFeatures = np.stack(nodeFeatures)
        else:
            nodeFeatures = None

        assert g.number_of_nodes() == numNodes
        gList.append(S2VGraph(bId, g, label, nodeTags, nodeFeatures))

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
    results = []
    share = math.ceil(len(graphs) / k)
    for i in range(k):
        start = i * share
        end = min((i + 1) * share, len(graphs))
        results.append(graphs[start: end])
        log.info(f'Fold {i + 1} range from {start} to {end - 1}')

    return results


def normalizeFeatures(graphs: List[S2VGraph],
                      useCachedTrain: bool = False,
                      useCachedTest: bool = False,
                      operation: str = 'min_max') -> List[List[float]]:
    if useCachedTrain:
        log.debug(f'Using previous calculated train min/max vector')
        maxVector = [
            2.0, 1367.0, 5411.0, 525.0, 0.0, 4245.0, 1.0, 2516938.0, 8875.0,
            2489922.0, 7916.0, 25527.0, 2516954.0
        ]
        minVector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        avgVector = [0.5962032061725268, 0.4180271544042307,
                     1.6158560684958867, 0.24502194444083092,
                     0.0, 2.175661078432198, 0.13442428699219405,
                     63.45916465925378, 2.262125637901996,
                     78.41444585456485, 0.018334971530191924,
                     1.9924283931335767, 70.89752509521182]
        stdVector = [0.49097240095840666, 1.3898312633734076,
                     27.59374888091898, 0.514249527342708,
                     0.000000000000001, 17.252417053163388,
                     0.3411076048066078, 8141.555766865466,
                     15.87218165891597, 7755.615488089883,
                     2.9998780886433027, 28.02481653517624,
                     8141.84431147715]
    elif useCachedTest:
        log.debug(f'Using previous calculated test min/max vector')
        maxVector = [
            2.0, 1883.0, 6999.0, 1036.0, 0.0, 4122.0, 1.0, 2514158.0, 5407.0,
            2486757.0, 4728.0, 25527.0, 2514174.0
        ]
        minVector = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
        ]
        avgVector = [0.5970183273238024, 0.414436492695235,
                     1.6023817480537264, 0.24687678246451827,
                     0.0, 2.1342394812947156, 0.1330682587555883,
                     69.69281705685793, 2.258427280249422,
                     83.6536707518127, 0.018659769558083145,
                     1.987237055993612, 77.07021552681057]
        stdVector = [0.4908864860606825, 1.5937404761957945,
                     27.58260356845197, 0.6617551383910477,
                     0.000000000000001, 17.397127884916774,
                     0.33964849069835007, 8474.05518627391,
                     15.633733132574285, 7959.109393152846,
                     2.8443524115866694, 24.230326277528274,
                     8474.337473921465]
    else:
        nodeFeatures = None
        for g in graphs:
            if nodeFeatures is None:
                nodeFeatures = np.array(g.node_features)
            else:
                nodeFeatures = np.concatenate((nodeFeatures, g.node_features),
                                              axis=0)

        log.debug(f'Dim of features of all nodes: {nodeFeatures.shape}')
        maxVector = np.amax(nodeFeatures, axis=0)
        minVector = np.amin(nodeFeatures, axis=0)
        avgVector = np.mean(nodeFeatures, axis=0)
        stdVector = np.std(nodeFeatures, axis=0)

    log.info(f'Max feature vector: {list(maxVector)}')
    log.info(f'Min feature vector: {list(minVector)}')
    log.info(f'Avg feature vector: {list(avgVector)}')
    log.info(f'Std feature vector: {list(stdVector)}')

    diff = [x - y for (x, y) in zip(maxVector, minVector)]
    diffVector = [x if x > 0 else 1 for x in diff]
    for g in graphs:
        if operation == 'min_max':
            g.node_features = (g.node_features - minVector) / diffVector
        elif operation == 'zero_mean':
            g.node_features = (g.node_features - avgVector) / stdVector
        else:
            log.debug(f'Unknown operation: {operation}')

        # delete column with only zeros
        g.node_features = np.delete(g.node_features, 4, 1)

    gHP['featureDim'] -= 1
    return [maxVector, minVector, avgVector, stdVector]


def computePrScores(pred, labels, prefix) -> Dict[str, float]:
    scores = {}
    scores['precisions'] = precision_score(labels, pred, average='weighted')
    scores['recalls'] = recall_score(labels, pred, average='weighted')
    scores['weightedF1'] = f1_score(labels, pred, average='weighted')
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
