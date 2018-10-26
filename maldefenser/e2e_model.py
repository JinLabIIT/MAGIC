#!/usr/bin/python3.7
import sys
import torch
import math
import glog as log
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from dgcnn_embedding import DGCNN
from typing import Dict, List
from mlp_dropout import MLPClassifier, RecallAtPrecision
from embedding import EmbedMeanField, EmbedLoopyBP
from ml_utils import cmd_args, gHP, S2VGraph


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        if cmd_args.gm == 'mean_field':
            model = EmbedMeanField
        elif cmd_args.gm == 'loopy_bp':
            model = EmbedLoopyBP
        elif cmd_args.gm == 'DGCNN':
            model = DGCNN
        else:
            log.fatal('Unknown graph embedding model: %s' % cmd_args.gm)
            sys.exit()

        if cmd_args.gm == 'DGCNN':
            self.s2v = model(latent_dim=gHP['convSize'],
                             output_dim=gHP['s2vOutDim'],
                             num_node_feats=(gHP['featureDim'] +
                                             gHP['nodeTagDim']),
                             num_edge_feats=0,
                             k=gHP['sortPoolingK'])
        else:
            self.s2v = model(latent_dim=gHP['convSize'],
                             output_dim=gHP['s2vOutDim'],
                             num_node_feats=gHP['featureDim'],
                             num_edge_feats=0,
                             max_lv=gHP['msgPassLv'])

        if cmd_args.mlp_type == 'rap':
            self.mlp = RecallAtPrecision(input_size=gHP['s2vOutDim'],
                                         hidden_size=gHP['regHidden'],
                                         alpha=0.6,
                                         with_dropout=gHP['dropOutRate'])
        else:
            self.mlp = MLPClassifier(input_size=gHP['s2vOutDim'],
                                     hidden_size=gHP['regHidden'],
                                     num_class=gHP['numClasses'],
                                     with_dropout=gHP['dropOutRate'])

    def _prepareFeatureLabel(self, batch_graph):
        labels = torch.LongTensor(len(batch_graph))
        n_nodes = 0

        if batch_graph[0].node_tags is not None:
            node_tag_flag = True
            concat_tag = []
        else:
            node_tag_flag = False

        if batch_graph[0].node_features is not None:
            node_feat_flag = True
            concat_feat = []
        else:
            node_feat_flag = False

        for i in range(len(batch_graph)):
            if batch_graph[i].label is not None:
                labels[i] = batch_graph[i].label

            n_nodes += batch_graph[i].num_nodes
            if node_tag_flag is True:
                concat_tag += batch_graph[i].node_tags
            if node_feat_flag is True:
                tmp = torch.from_numpy(
                    batch_graph[i].node_features).type('torch.FloatTensor')
                concat_feat.append(tmp)

        if node_tag_flag is True:
            concat_tag = torch.LongTensor(concat_tag).view(-1, 1)
            node_tag = torch.zeros(n_nodes, gHP['nodeTagDim'])
            node_tag.scatter_(1, concat_tag, 1)

        if node_feat_flag is True:
            node_feat = torch.cat(concat_feat, 0)

        if node_feat_flag and node_tag_flag:
            """
            Concatenate one-hot embedding of node tags (node labels)
            with continuous node features
            """
            node_feat = torch.cat([node_tag.type_as(node_feat), node_feat], 1)
        elif node_feat_flag is False and node_tag_flag is True:
            node_feat = node_tag
        elif node_feat_flag is True and node_tag_flag is False:
            pass
        else:
            # use all-one vector as node features
            node_feat = torch.ones(n_nodes, 1)

        if cmd_args.mode == 'gpu':
            node_feat = node_feat.cuda()
            labels = labels.cuda()

        return node_feat, labels

    def forward(self, batch_graph):
        node_feat, labels = self._prepareFeatureLabel(batch_graph)
        embed = self.s2v(batch_graph, node_feat, edge_feat=None)
        return self.mlp(embed, labels)

    def embedding(self, graphs):
        node_feat, _ = self._prepareFeatureLabel(graphs)
        return self.s2v(graphs, node_feat, edge_feat=None)

    def predict(self, testGraphs):
        nodeFeature, _ = self._prepareFeatureLabel(testGraphs)
        embed = self.s2v(testGraphs, nodeFeature, edge_feat=None)
        return self.mlp(embed)

    def sgdModel(self, optimizer, batch_graph, pos):
        if cmd_args.mlp_type == 'rap':
            for p in self.parameters():
                p.requires_grad_(True)
            self.mlp.lam.requires_grad_(False)
            optimizer.zero_grad()
            loss, acc, pred = self.forward(batch_graph)
            loss.backward()
            optimizer.step()

            if pos != 0 and pos % 5 == 0:
                for p in self.parameters():
                    p.requires_grad_(False)
                self.mlp.lam.requires_grad_(True)
                optimizer.zero_grad()
                loss, acc, pred = self.forward(batch_graph)
                loss.backward()
                optimizer.step()
        else:
            optimizer.zero_grad()
            loss, acc, pred = self.forward(batch_graph)
            loss.backward()
            optimizer.step()


def loopDataset(gList: List[S2VGraph], classifier: Classifier,
                sampleIndices: List[int], optimizer=None):
    """Train e2e model by looping over dataset"""
    bsize = gHP['batchSize']
    totalScore = []
    numGiven = len(sampleIndices)
    totalIters = math.ceil(numGiven / bsize)
    pbar = tqdm(range(totalIters), unit='batch')
    numUsed, allPred, allLabel = 0, [], []

    for pos in pbar:
        end = min((pos + 1) * bsize, numGiven)
        batchIndices = sampleIndices[pos * bsize: end]
        batchGraphs = [gList[idx] for idx in batchIndices]
        if classifier.training:
            classifier.sgdModel(optimizer, batchGraphs, pos)

        loss, acc, pred = classifier(batchGraphs)
        allPred.extend(pred.data.cpu().numpy().tolist())
        allLabel.extend([g.label for g in batchGraphs])
        loss = loss.data.cpu().numpy()
        pbar.set_description('loss: %.5f acc: %.5f' % (loss, acc))
        totalScore.append(np.array([loss, acc]))
        numUsed += len(batchIndices)

    if numUsed != numGiven:
        log.warning(f"{numUsed} of {numGiven} cases used trainning/validating.")

    classifier.mlp.print_result_dict()
    avgScore = np.mean(np.array(totalScore), 0)
    return avgScore, allPred, allLabel


def predictDataset(gList: List[S2VGraph], classifier: Classifier):
    """Inference batch by batch on large dataset"""
    indices = list(range(len(gList)))
    allPred = []
    bsize = gHP['batchSize']
    totalIters = math.ceil(len(gList) / bsize)
    pbar = tqdm(range(totalIters), unit='batch')
    pbar.set_description('predicting')

    for pos in pbar:
        end = min((pos + 1) * bsize, len(gList))
        batchGraphs = [gList[idx] for idx in indices[pos * bsize: end]]
        batchPred = classifier.predict(batchGraphs)
        allPred.extend(batchPred.data.cpu().numpy().tolist())

    return allPred
