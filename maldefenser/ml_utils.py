from __future__ import print_function

import argparse
import random
import torch
import numpy as np
import glog as log
import networkx as nx
import cPickle as cp
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
# import os
# import _pickle as cp  # python3 compatability

cmd_opt = argparse.ArgumentParser(
    description='Argparser for graph_classification')
cmd_opt.add_argument('-mode', default='cpu', help='cpu/gpu')
cmd_opt.add_argument('-gm', default='mean_field', help='mean_field/loopy_bp')
cmd_opt.add_argument('-data', default=None, help='data folder name')
cmd_opt.add_argument(
    '-batch_size', type=int, default=50, help='minibatch size')
cmd_opt.add_argument('-seed', type=int, default=1, help='seed')
cmd_opt.add_argument(
    '-feat_dim', type=int, default=0,
    help='dimension of discrete node feature (maximum node tag)')
cmd_opt.add_argument('-num_class', type=int, default=0, help='#classes')
cmd_opt.add_argument('-fold', type=int, default=1, help='fold (1..10)')
cmd_opt.add_argument(
    '-test_number', type=int, default=0,
    help='if specified, will overwrite -fold and \
    use the last -test_number graphs as testing data')
cmd_opt.add_argument('-num_epochs', type=int, default=1000,
                     help='number of epochs')
cmd_opt.add_argument('-latent_dim', type=str, default='64',
                     help='dimension(s) of latent layers')
cmd_opt.add_argument('-sortpooling_k', type=float, default=30,
                     help='number of nodes kept after SortPooling')
cmd_opt.add_argument('-out_dim', type=int, default=1024,
                     help='s2v output size')
cmd_opt.add_argument('-hidden', type=int, default=100,
                     help='dimension of regression')
cmd_opt.add_argument('-max_lv', type=int, default=4,
                     help='max rounds of message passing')
cmd_opt.add_argument('-learning_rate', type=float, default=0.0001,
                     help='init learning_rate')
cmd_opt.add_argument('-mlp_type', type=str, default='vanilla',
                     help='init learning_rate')
cmd_opt.add_argument('-dropout', type=str, default='False',
                     help='whether add dropout after dense layer')
cmd_opt.add_argument('-use_cached_data', type=str, default='False',
                     help='whether to use previously cached dataset')
cmd_opt.add_argument('-cache_file', type=str, default='cached_graphs.pkl',
                     help='which cached data to use')

cmd_args, _ = cmd_opt.parse_known_args()

cmd_args.latent_dim = [int(x) for x in cmd_args.latent_dim.split('-')]
if len(cmd_args.latent_dim) == 1:
    cmd_args.latent_dim = cmd_args.latent_dim[0]

cmd_args.dropout = (cmd_args.dropout == "True")
cmd_args.use_cached_data = (cmd_args.use_cached_data == "True")
log.info(cmd_args)


class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a numpy array of continuous node features
        '''
        self.num_nodes = len(node_tags)
        self.node_tags = node_tags
        self.label = label
        self.node_features = node_features  # np array (node_num * feature_dim)
        self.degs = dict(g.degree).values()

        x, y = zip(*g.edges())
        self.num_edges = len(x)
        self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
        self.edge_pairs[:, 0] = x
        self.edge_pairs[:, 1] = y
        self.edge_pairs = self.edge_pairs.flatten()


def load_data():
    log.info('Loading data as graphs')
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('data/%s/%s.txt' % (cmd_args.data, cmd_args.data), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if l not in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row = [int(w) for w in row[:tmp]]
                    attr = np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            # (some graphs in COLLAB have self-loops, ignored here)
            # assert len(g.edges()) * 2 == n_edges
            assert len(g) == n
            if len(g.edges()) > 0:
                g_list.append(S2VGraph(g, l, node_tags, node_features))

    random.shuffle(g_list)
    for g in g_list:
        g.label = label_dict[g.label]

    log.info('# graphs: %d' % len(g_list))

    cmd_args.num_class = len(label_dict)
    # maximum node label (tag)
    cmd_args.feat_dim = len(feat_dict)
    if node_feature_flag is True:
        # dim of node features (attributes)
        cmd_args.attr_dim = node_features.shape[1]
    else:
        cmd_args.attr_dim = 0

    log.info('# classes: %d' % cmd_args.num_class)
    log.info('# maximum node tag: %d' % cmd_args.feat_dim)

    if cmd_args.test_number == 0:
        train_idxes = np.loadtxt('data/%s/10fold_idx/train_idx-%d.txt' %
                                 (cmd_args.data, cmd_args.fold),
                                 dtype=np.int32).tolist()
        test_idxes = np.loadtxt('data/%s/10fold_idx/test_idx-%d.txt' %
                                (cmd_args.data, cmd_args.fold),
                                dtype=np.int32).tolist()
        return [g_list[i] for i in train_idxes], \
            [g_list[i] for i in test_idxes]
    else:
        return g_list[: len(g_list) - cmd_args.test_number], \
            g_list[len(g_list) - cmd_args.test_number:]


def compute_pr_scores(pred, labels, prefix):
    scores = {}
    scores['precisions'] = precision_score(labels, pred, average=None)
    scores['recalls'] = recall_score(labels, pred, average=None)
    return scores


def store_confusion_matrix(pred, labels, prefix):
    cm = confusion_matrix(labels, pred)
    np.savetxt('%s_%s_confusion_matrix.txt' % (cmd_args.data, prefix), cm,
               fmt='%4d', delimiter=' ')


def store_embedding(classifier, graphs, prefix, sample_size=100):
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


def balanced_sampling(graphs, neg_ratio=3):
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


def load_graphs_may_cache():
    cached_filename = cmd_args.cache_file
    if cmd_args.use_cached_data:
        log.info("Loading cached dataset from %s" % cached_filename)
        cache_file = open(cached_filename, 'rb')
        dataset = cp.load(cache_file)
        cmd_args.num_class = dataset['num_class']
        cmd_args.feat_dim = dataset['feat_dim']
        cmd_args.attr_dim = dataset['attr_dim']
        train_graphs = dataset['train_graphs']
        test_graphs = dataset['test_graphs']
        cache_file.close()
    else:
        train_graphs, test_graphs = load_data()
        log.info("Dumping cached dataset from %s" % cached_filename)
        cache_file = open(cached_filename, 'wb')
        dataset = {}
        dataset['num_class'] = cmd_args.num_class
        dataset['feat_dim'] = cmd_args.feat_dim
        dataset['attr_dim'] = cmd_args.attr_dim
        dataset['train_graphs'] = train_graphs
        dataset['test_graphs'] = test_graphs
        cp.dump(dataset, cache_file)
        cache_file.close()

    return train_graphs, test_graphs


def fair_sample_prob(indices, labels):
    """
    sample_prob[ith elem] is proportional to 1 / prob[ith elem's label]
    """
    hist, _ = np.histogram(labels, bins=np.arange(max(labels) + 2),
                           density=False)
    dist, _ = np.histogram(labels, bins=np.arange(max(labels) + 2),
                           density=True)
    sample_prob = [1 / x for x in dist]
    sample_prob = [x / sum(sample_prob) for x in sample_prob]  # normalize
    ret = [sample_prob[x] / hist[x] for x in labels]

    return ret


def to_onehot(indices, num_classes):
    onehot = torch.zeros(indices.size(0), num_classes, device=indices.device)
    return onehot.scatter_(1, indices.unsqueeze(1), 1)
