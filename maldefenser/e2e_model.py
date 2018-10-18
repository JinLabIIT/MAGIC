#!/usr/bin/python3.7
import sys
import torch
import glog as log
import torch.optim as optim
import torch.nn as nn
from dgcnn_embedding import DGCNN
from mlp_dropout import MLPClassifier, RecallAtPrecision
from embedding import EmbedMeanField, EmbedLoopyBP
from ml_utils import cmd_args


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
            self.s2v = model(latent_dim=cmd_args.latent_dim,
                             output_dim=cmd_args.out_dim,
                             num_node_feats=(cmd_args.feat_dim +
                                             cmd_args.attr_dim),
                             num_edge_feats=0,
                             k=cmd_args.sortpooling_k)
        else:
            self.s2v = model(latent_dim=cmd_args.latent_dim,
                             output_dim=cmd_args.out_dim,
                             num_node_feats=cmd_args.feat_dim,
                             num_edge_feats=0,
                             max_lv=cmd_args.max_lv)

        out_dim = cmd_args.out_dim
        if out_dim == 0:
            if cmd_args.gm == 'DGCNN':
                out_dim = self.s2v.dense_dim
            else:
                out_dim = cmd_args.latent_dim

        if cmd_args.mlp_type == 'rap':
            self.mlp = RecallAtPrecision(input_size=out_dim,
                                         hidden_size=cmd_args.hidden,
                                         alpha=0.6,
                                         with_dropout=cmd_args.dropout)
        else:
            self.mlp = MLPClassifier(input_size=out_dim,
                                     hidden_size=cmd_args.hidden,
                                     num_class=cmd_args.num_class,
                                     with_dropout=cmd_args.dropout)

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
            node_tag = torch.zeros(n_nodes, cmd_args.feat_dim)
            node_tag.scatter_(1, concat_tag, 1)

        if node_feat_flag is True:
            node_feat = torch.cat(concat_feat, 0)

        if node_feat_flag and node_tag_flag:
            # concatenate one-hot embedding of node tags (node labels)
            # with continuous node features
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
        embed = self.s2v(batch_graph, node_feat, None)
        return self.mlp(embed, labels)

    def embedding(self, graphs):
        node_feat, _ = self._prepareFeatureLabel(graphs)
        return self.s2v(graphs, node_feat, None)

    def sgdModel(self, optimizer, batch_graph, pos):
        if cmd_args.mlp_type == 'rap':
            for p in self.parameters():
                p.requires_grad_(True)
            self.mlp.lam.requires_grad_(False)
            loss, acc, pred = self.forward(batch_graph)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if pos != 0 and pos % 5 == 0:
                for p in self.parameters():
                    p.requires_grad_(False)
                self.mlp.lam.requires_grad_(True)
                loss, acc, pred = self.forward(batch_graph)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        else:
            loss, acc, pred = self.forward(batch_graph)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
