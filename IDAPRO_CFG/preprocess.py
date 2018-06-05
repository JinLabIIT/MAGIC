from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from networkx import number_of_nodes
from sets import Set
# from itertools import izip_longest
import pandas as pd
import pickle as pkl
# import csv
import glob
import re


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
                # Assume only 1 file under data dir
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


def extract_operator_words():
    operators = Set()
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

    df = pd.DataFrame({'operator': list(operators)})
    df = df.sort_values(by='operator')
    df.to_csv('operator.csv', index=False, header=False)


def extract_operand_words():
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
                        if len(inst) > 1:
                            for op in inst[1].split(','):
                                comment_idx = op.find(';')
                                operands.add(op if comment_idx == -1
                                             else op[:comment_idx])

    df = pd.DataFrame({'operand': list(operands)})
    df = df.sort_values(by='operand')
    df.to_csv('operand.csv', index=False, header=False)


def match_constants():
    f = open('test_operand.csv', 'rb')
    numerics = []
    strings = []
    for line in f.readlines():
        operand = line.strip('\n\r\t ')
        """Whole operand is a num OR leading num in expression.
        E.g. "0ABh", "589h", "0ABh" in "0ABh*589h"
        """
        whole_num = r'^([1-9][0-9A-F]*|0[A-F][0-9A-F]*)h?.*'
        pattern = re.compile(whole_num)
        if pattern.match(operand):
            numerics.append('%s:WHOLE/LEAD' % operand)

        """Number inside expression, exclude the leading one."""
        num_in_expr = r'([+*/:]|-)([1-9][0-9A-F]*|0[A-F][0-9A-F]*)h?'
        pattern = re.compile(num_in_expr)
        match = pattern.findall(operand)
        if len(match) > 0:
            numerics.append('%s:%d' % (operand, len(match)))

        """Const string inside double/single quote"""
        str_re = r'["\'][^"]+["\']'
        pattern = re.compile(str_re)
        match = pattern.findall(operand)
        if len(match) > 0:
            strings.append('%s:%d' % (operand, len(match)))

    f.close()

    df = pd.DataFrame(numerics)
    df.to_csv('parsed_num.csv', index=False, header=False)
    df = pd.DataFrame(strings)
    df.to_csv('parsed_str.csv', index=False, header=False)


def log_graph(pkl_dir, log_path):
    data_paths = glob.glob(pkl_dir + '/*')
    f = open(log_path, 'wb')
    if len(data_paths) != 1:
        print('Multiple or zero graph pickles in %s' % pkl_dir)
    else:
        G = pkl.load(open(data_paths[0], 'rb'))
        for (node, attributes) in G.nodes(data=True):
            f.write(str(node) + '\n')
            for (key, val) in attributes.items():
                f.write('\t%s: %s\n' % (key, str(val)))

    f.close()


pkl_dir = 'Bifrose/4058ca0cc761f2b81f11986dedb494be'
log_path = '4058ca0cc761f2b81f11986dedb494be.log'
log_graph(pkl_dir, log_path)

class_dirnames = glob.glob('./*')
# extract_operator_words()
# extract_node_hist()
# match_constants()
