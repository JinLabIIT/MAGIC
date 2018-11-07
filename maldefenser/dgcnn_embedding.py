#!/usr/bin/python3.7
import os
import sys
import glog as log
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

sys.path.append('%s/pytorch_structure2vec-master/s2v_lib' %
                os.path.dirname(os.path.realpath(__file__)))
from s2v_lib import S2VLIB
from pytorch_util import weights_init, gnn_spmm


class DGCNN(nn.Module):
    def __init__(self, output_dim, num_node_feats, num_edge_feats=0,
                 latent_dim=[32, 32, 32, 1], k=30,
                 pooling_layer='sort', remaining_layers='conv1d',
                 conv2d_changel=16,
                 conv1d_channels=[16, 32],
                 conv1d_kws=[0, 5], conv1d_maxpl=[2, 2]):
        """
        Args
            output_dim: dimension of the DGCNN. If equals zero, it will be
                        computed as the output of the final 1d conv layer;
                        Otherwise, an extra dense layer will be appended after
                        the final 1d conv layer to produce exact output size.
            num_nodes_feats, num_edge_feats: dim of the node/edge attributes.
            latend_dim: sizes of graph convolution layers.
            pooling_layer: type of pooling graph vertices, 'sort' or 'adaptive'.
            remaining_layers: 'conv1d' or 'weight_vertices'.
                              NOT used if pooling layer is 'adaptive'.
            conv1d_channels: channel dimension of the 2 1d conv layers
            conv1d_kws: kernel size of the 2 1d conv layers.
                        conv1d_kws[0] is manually set to sum(latent_dim).
        """
        log.info('Initializing DGCNN')
        super(DGCNN, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats
        self.k = k
        self.total_latent_dim = sum(latent_dim)
        conv1d_kws[0] = self.total_latent_dim

        self.conv_params = nn.ModuleList()
        self.conv_params.append(nn.Linear(num_node_feats, latent_dim[0]))
        for i in range(1, len(latent_dim)):
            self.conv_params.append(
                nn.Linear(latent_dim[i - 1], latent_dim[i]))

        self.pooling_layer = pooling_layer
        if pooling_layer == 'adaptive':
            log.info(f'Unify graph sizes with ADAPTIVE pooling')
            self.conv2d_param = nn.Conv2d(in_channels=1,
                                          out_channels=conv2d_changel,
                                          kernel_size=3, padding=1)
            self.adp_pool = nn.AdaptiveMaxPool2d((self.k,
                                                  self.total_latent_dim))
        else:
            log.info(f'Unify graph sizes with SORT pooling')
            self.remaining_layers = remaining_layers
            if remaining_layers == 'weight_vertices':
                log.info(f'Ending with weight vertices layers')
                self.node_weights1 = nn.Conv1d(in_channels=1,
                                               out_channels=1,
                                               kernel_size=self.k,
                                               stride=self.k)
                self.dense_dim = self.total_latent_dim
            else:  # conv1d if not specified
                log.info(f'Ending with conv1d since remLayers not specified')
                self.conv1d_params1 = nn.Conv1d(in_channels=1,
                                                out_channels=conv1d_channels[0],
                                                kernel_size=conv1d_kws[0],
                                                stride=conv1d_kws[0])
                self.maxpool1d = nn.MaxPool1d(kernel_size=conv1d_maxpl[0],
                                              stride=conv1d_maxpl[1])
                self.conv1d_params2 = nn.Conv1d(in_channels=conv1d_channels[0],
                                                out_channels=conv1d_channels[1],
                                                kernel_size=conv1d_kws[1])
                tmp = int((k - conv1d_maxpl[0]) / conv1d_maxpl[1] + 1)
                self.dense_dim = (tmp - conv1d_kws[1] + 1) * conv1d_channels[1]

            if num_edge_feats > 0:
                self.w_e2l = nn.Linear(num_edge_feats, latent_dim)

            if output_dim > 0:
                self.out_params = nn.Linear(self.dense_dim, output_dim)

        weights_init(self)

    def forward(self, graph_list, node_feat, edge_feat):
        graph_sizes = [graph_list[i].num_nodes for i in range(len(graph_list))]
        node_degs = [
            torch.Tensor(graph_list[i].degs) + 1
            for i in range(len(graph_list))
        ]
        node_degs = torch.cat(node_degs).unsqueeze(1)

        n2n_sp, e2n_sp, subg_sp = S2VLIB.PrepareMeanField(graph_list)

        if isinstance(node_feat, torch.cuda.FloatTensor):
            n2n_sp = n2n_sp.cuda()
            e2n_sp = e2n_sp.cuda()
            subg_sp = subg_sp.cuda()
            node_degs = node_degs.cuda()

        node_feat = Variable(node_feat)
        if edge_feat is not None:
            edge_feat = Variable(edge_feat)

        n2n_sp = Variable(n2n_sp)
        e2n_sp = Variable(e2n_sp)
        subg_sp = Variable(subg_sp)
        node_degs = Variable(node_degs)

        conv_graphs = self.graph_convolution_layers(node_feat, edge_feat,
                                                    n2n_sp, e2n_sp, subg_sp,
                                                    graph_sizes, node_degs)
        if self.pooling_layer == 'adaptive':
            return self.adapooling_layer(conv_graphs, node_feat, graph_sizes)
        else:
            sp_graphs = self.sortpooling_layer(conv_graphs, node_feat,
                                               subg_sp, graph_sizes)
            if self.remaining_layers == 'weight_vertices':
                return self.weight_vertices_layers(sp_graphs, len(graph_sizes))
            else:
                return self.conv1d_layers(sp_graphs, len(graph_sizes))

    def graph_convolution_layers(self, node_feat, edge_feat, n2n_sp, e2n_sp,
                                 subg_sp, graph_sizes, node_degs):
        """graph convolution layers"""
        # if exists edge feature, concatenate to node feature vector
        if edge_feat is not None:
            input_edge_linear = self.w_e2l(edge_feat)
            e2npool_input = gnn_spmm(e2n_sp, input_edge_linear)
            node_feat = torch.cat([node_feat, e2npool_input], 1)

        lv = 0
        cur_message_layer = node_feat
        cat_message_layers = []
        while lv < len(self.latent_dim):
            # Y = (A + I) * X
            n2npool = gnn_spmm(n2n_sp, cur_message_layer) + cur_message_layer
            node_linear = self.conv_params[lv](n2npool)  # Y = Y * W
            normalized_linear = node_linear.div(node_degs)  # Y = D^-1 * Y
            cur_message_layer = F.tanh(normalized_linear)
            cat_message_layers.append(cur_message_layer)
            lv += 1

        message_layers = torch.cat(cat_message_layers, 1)
        return message_layers

    def adapooling_layer(self, message_layers, node_feat, graph_sizes):
        ap_graphs = torch.zeros(len(graph_sizes),
                                self.conv2d_param.out_channels,
                                self.k,
                                self.total_latent_dim)
        if isinstance(node_feat.data, torch.cuda.FloatTensor):
            ap_graphs = ap_graphs.cuda()

        accum_count = 0
        for i in range(len(graph_sizes)):
            graph = message_layers[accum_count: accum_count + graph_sizes[i]]
            to_conv = graph.unsqueeze(0)
            to_conv = to_conv.unsqueeze(0)
            conved = self.conv2d_param(to_conv)
            ap_graphs[i] = self.adp_pool(conved)
            accum_count += graph_sizes[i]

        return ap_graphs

    def sortpooling_layer(self, message_layers, node_feat,
                          subg_sp, graph_sizes):
        """sortpooling layer"""
        sort_channel = message_layers[:, -1]
        sp_graphs = torch.zeros(len(graph_sizes), self.k, self.total_latent_dim)
        if isinstance(node_feat.data, torch.cuda.FloatTensor):
            sp_graphs = sp_graphs.cuda()

        sp_graphs = Variable(sp_graphs)
        accum_count = 0
        for i in range(len(graph_sizes)):
            to_sort = sort_channel[accum_count:accum_count + graph_sizes[i]]
            k = self.k if self.k <= graph_sizes[i] else graph_sizes[i]
            _, topk_indices = to_sort.topk(k)
            topk_indices += accum_count
            sortpooling_graph = message_layers.index_select(0, topk_indices)
            if k < self.k:
                to_pad = torch.zeros(self.k - k, self.total_latent_dim)
                if isinstance(node_feat.data, torch.cuda.FloatTensor):
                    to_pad = to_pad.cuda()

                to_pad = Variable(to_pad)
                sortpooling_graph = torch.cat((sortpooling_graph, to_pad), 0)

            sp_graphs[i] = sortpooling_graph
            accum_count += graph_sizes[i]

        return sp_graphs

    def conv1d_layers(self, sp_graphs, num_graphs):
        """traditional 1d convlution and dense layers"""
        to_conv1d = sp_graphs.view((-1, 1, self.k * self.total_latent_dim))
        conv1d_res = self.conv1d_params1(to_conv1d)
        conv1d_res = F.relu(conv1d_res)
        conv1d_res = self.maxpool1d(conv1d_res)
        conv1d_res = self.conv1d_params2(conv1d_res)
        conv1d_res = F.relu(conv1d_res)

        to_dense = conv1d_res.view(num_graphs, -1)
        if self.output_dim == 0:
            return to_dense
        else:
            out_linear = self.out_params(to_dense)
            return F.relu(out_linear)

    def weight_vertices_layers(self, sp_graphs, num_graphs):
        tp_graphs = torch.zeros(num_graphs, self.total_latent_dim, self.k)
        if isinstance(sp_graphs.data, torch.cuda.FloatTensor):
            tp_graphs = tp_graphs.cuda()

        for i in range(sp_graphs.size()[0]):
            tp_graphs[i] = torch.t(sp_graphs[i])

        to_conv1d = tp_graphs.view((-1, 1, self.k * self.total_latent_dim))
        embeded_graphs = self.node_weights1(to_conv1d)

        to_dense = embeded_graphs.view(num_graphs, -1)
        if self.output_dim == 0:
            return to_dense
        else:
            to_dense = F.relu(to_dense)
            out_linear = self.out_params(to_dense)
            return F.relu(out_linear)
