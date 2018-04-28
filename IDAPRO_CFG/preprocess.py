from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from networkx import to_numpy_matrix, number_of_nodes
from sets import Set
# from itertools import izip_longest
import pandas as pd
import pickle as pkl
# import csv
import glob


def extract_node_hist():
    pkl_filenames = []
    max_num_nodes = 0
    num_nodes_dist = []
    max_size_filename = ''
    for class_dirname in class_dirnames:
        data_dirnames = glob.glob(class_dirname + '/*')
        for data_dirname in data_dirnames:
            data_paths = glob.glob(data_dirname + '/*')
            if len(data_paths) == 0:
                print('[Warning] %s is empty' % data_dirname)
            else:
                pkl_filenames.append(data_paths[0])
                G = pkl.load(open(data_paths[0], 'rb'))
                num_nodes = number_of_nodes(G)
                if num_nodes > max_num_nodes:
                    max_size_filename = data_paths[0]
                num_nodes_dist.append(num_nodes)
                max_num_nodes = max(max_num_nodes, num_nodes)

    print('Total number of graphs:', len(pkl_filenames))
    print('Maximum number of nodes: %d in %s' %
          (max_num_nodes, max_size_filename))
    print('Number of nodes:', num_nodes_dist)

    return pkl_filenames


def extract_op_words():
    operators = Set()
    operands = Set()
    for class_dirname in class_dirnames:
        data_dirnames = glob.glob(class_dirname + '/*')
        for data_dirname in data_dirnames:
            data_paths = glob.glob(data_dirname + '/*')
            if len(data_paths) > 0:
                G = pkl.load(open(data_paths[0], 'rb'))
                for (node, attributes) in G.nodes(data=True):
                    instructions = attributes['Ins']
                    for (addr, inst) in instructions:
                        operators.add(inst[0])
                        if len(inst) > 1:
                            for op in inst[1].split(','):
                                comment_idx = op.find(';')
                                operands.add(op if comment_idx == -1
                                             else op[:comment_idx])

    df = pd.DataFrame({'operator': list(operators)})
    df.to_csv('operator.csv', index=False)
    df = pd.DataFrame({'operand': list(operands)})
    df.to_csv('operand.csv', index=False)


class_dirnames = glob.glob('./*')
extract_op_words()
# extract_node_hist()
# print(to_numpy_matrix(G).shape)
# for (n, nbrs) in G.adjacency_iter():
# for nbr, eattr in nbrs.items():
