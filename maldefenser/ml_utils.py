#!/usr/bin/python3.7
import argparse
import random
import torch
import math
import numpy as np
import glog as log
import networkx as nx
import pickle as pkl
from typing import List, Dict, Set
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

cmd_opt = argparse.ArgumentParser(
    description='Argparser for graph_classification')
# Execution options
cmd_opt.add_argument('-mode', default='cpu', help='cpu/gpu')
cmd_opt.add_argument('-gm', default='mean_field', help='mean_field/loopy_bp')
cmd_opt.add_argument('-data', default=None, help='data folder name')
cmd_opt.add_argument('-seed', type=int, default=1, help='seed')
cmd_opt.add_argument('-mlp_type', type=str, default='vanilla',
                     help='Type of regression MLP: RAP or vanilla')
cmd_opt.add_argument('-use_cached_data', type=str, default='False',
                     help='whether to use previously cached dataset')
cmd_opt.add_argument('-cache_file', type=str, default='cached_graphs.pkl',
                     help='which cached data to use')
gHP = dict()
cmd_args, _ = cmd_opt.parse_known_args()
cmd_args.use_cached_data = (cmd_args.use_cached_data == "True")
log.info("Parsed cmdline arguments: %s" % cmd_args)


class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        """
        g: a networkx graph
        label: an integer graph label
        node_tags: a list of integer node tags
        node_features: a numpy array of continuous node features
        """
        self.num_nodes = len(node_tags)
        self.node_tags = node_tags
        self.label = label
        self.node_features = node_features  # nparray (node_num * feature_dim)
        self.degs = list(dict(g.degree).values())

        x, y = zip(*g.edges())
        self.num_edges = len(x)
        self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
        self.edge_pairs[:, 0] = x
        self.edge_pairs[:, 1] = y
        self.edge_pairs = self.edge_pairs.flatten()


def loadData() -> List[S2VGraph]:
    log.info('Loading data as list of S2VGraph(s)')
    gList: List[S2VGraph] = []
    labelDict: Dict[str, int] = {} # mapping label to 0-based int
    tagDict: Dict[str, int] = {}   # mapping node tag to 0-based int

    f = open('data/%s/%s.txt' % (cmd_args.data, cmd_args.data), 'r')
    numGraphs = int(f.readline().strip())
    for i in range(numGraphs):
        row = f.readline().strip().split()
        numNodes, label = [int(w) for w in row]
        if label not in labelDict:
            mapped = len(labelDict)
            labelDict[label] = mapped

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
        if g.number_of_edges() > 0:
            gList.append(S2VGraph(g, label, nodeTags, nodeFeatures))
        else:
            log.warning('[LoadData] Ignore graph having no edge')

    random.shuffle(gList)
    for g in gList:
        g.label = labelDict[g.label]

    gHP['numClasses'] = len(labelDict)
    gHP['nodeTagDim'] = len(tagDict)
    gHP['featureDim'] = 0
    if nodeFeatures is True:
        gHP['featureDim'] = nodeFeatures.shape[1]

    log.info(f'# graphs: {len(gList)}')
    log.info(f'# classes: {gHP["numClasses"]}')
    log.info(f'node tag dimension: {gHP["nodeTagDim"]}')
    log.info(f'node attributes dimension: {gHP["featureDim"]}')
    return gList


def loadGraphsMayCache() -> List[S2VGraph]:
    """ Enhance loadData() with caching. """
    cached_filename = cmd_args.cache_file
    if cmd_args.use_cached_data:
        log.info(f"Loading cached dataset from {cached_filename}")
        cache_file = open(cached_filename, 'rb')
        dataset = pkl.load(cache_file)
        gHP['numClasses'] = dataset['numClasses']
        gHP['featureDim'] = dataset['featureDim']
        gHP['nodeTagDim'] = dataset['nodeTagDim']
        graphs = dataset['graphs']
        cache_file.close()
    else:
        graphs = loadData()
        log.info(f"Dumping cached dataset to {cached_filename}")
        cache_file = open(cached_filename, 'wb')
        dataset = {}
        dataset['numClasses'] = gHP['numClasses']
        dataset['featureDim'] = gHP['featureDim']
        dataset['nodeTagDim'] = gHP['nodeTagDim']
        dataset['graphs'] = list(graphs)
        pkl.dump(dataset, cache_file)
        cache_file.close()

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

def computePrScores(pred, labels, prefix) -> Dict[str, float]:
    scores = {}
    scores['precisions'] = precision_score(labels, pred, average=None)
    scores['recalls'] = recall_score(labels, pred, average=None)
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
