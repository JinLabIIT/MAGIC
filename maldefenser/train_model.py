import sys
import os
import torch
import math
import random
import time
import glog as log
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from dgcnn_embedding import DGCNN
from mlp_dropout import MLPClassifier, RecallAtPrecision
from embedding import EmbedMeanField, EmbedLoopyBP
from ml_utils import cmd_args, store_confusion_matrix
from ml_utils import compute_pr_scores
from ml_utils import balanced_sampling, load_graphs_may_cache


sys.path.append('%s/pytorch_structure2vec-master/s2v_lib' %
                os.path.dirname(os.path.realpath(__file__)))


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
            log.fatal('unknown gm %s' % cmd_args.gm)
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

    def PrepareFeatureLabel(self, batch_graph):
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
        node_feat, labels = self.PrepareFeatureLabel(batch_graph)
        embed = self.s2v(batch_graph, node_feat, None)
        return self.mlp(embed, labels)

    def embedding(self, graphs):
        node_feat, _ = self.PrepareFeatureLabel(graphs)
        return self.s2v(graphs, node_feat, None)


def sgd_model(classifier, optimizer, batch_graph, pos):
    if cmd_args.mlp_type == 'rap':
        for p in classifier.parameters():
            p.requires_grad_(True)
        classifier.mlp.lam.requires_grad_(False)
        loss, acc, pred = classifier(batch_graph)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if pos != 0 and pos % 5 == 0:
            for p in classifier.parameters():
                p.requires_grad_(False)
            classifier.mlp.lam.requires_grad_(True)
            loss, acc, pred = classifier(batch_graph)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    else:
        loss, acc, pred = classifier(batch_graph)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def loop_dataset(g_list, classifier, sample_indices,
                 optimizer=None, bsize=cmd_args.batch_size):
    total_score = []
    n_given = len(sample_indices)
    if optimizer is None:
        total_iters = n_given // bsize
    else:
        total_iters = (n_given + bsize - 1) // bsize
    pbar = tqdm(range(total_iters), unit='batch')

    n_tested = 0
    all_pred = []
    all_label = []
    for pos in pbar:
        batch_indices = sample_indices[pos * bsize: (pos + 1) * bsize]
        batch_graph = [g_list[idx] for idx in batch_indices]
        if classifier.training:
            sgd_model(classifier, optimizer, batch_graph, pos)

        loss, acc, pred = classifier(batch_graph)
        all_pred.extend(pred.data.cpu().numpy().tolist())
        all_label.extend([g.label for g in batch_graph])
        loss = loss.data.cpu().numpy()
        pbar.set_description('loss: %0.5f acc: %0.5f' % (loss, acc))
        total_score.append(np.array([loss, acc]))
        n_tested += len(batch_indices)

    if optimizer is None and n_tested != n_given:
        log.warning("%d of %d cased used in testing." % (n_tested, n_given))

    classifier.mlp.print_result_dict()
    total_score = np.array(total_score)
    avg_score = np.mean(np.array(total_score), 0)
    return avg_score, all_pred, all_label


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    log.setLevel("INFO")
    start_time = time.time()

    train_graphs, test_graphs = load_graphs_may_cache()
    # train_graphs = balanced_sampling(train_graphs)
    # test_graphs = balanced_sampling(test_graphs)
    log.info('#train: %d, #test: %d' % (len(train_graphs), len(test_graphs)))
    data_ready_time = time.time() - start_time
    log.info('Dataset ready takes %.2fs' % data_ready_time)
    if cmd_args.sortpooling_k <= 1:
        num_nodes_list = sorted([g.num_nodes for g in train_graphs +
                                 test_graphs])
        cmd_args.sortpooling_k = num_nodes_list[
            int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1
        ]
        log.info('k used in SortPooling is: ' + str(cmd_args.sortpooling_k))

    classifier = Classifier()
    if cmd_args.mode == 'gpu':
        classifier = classifier.cuda()

    optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)

    train_indices = list(range(len(train_graphs)))
    test_indices = list(range(len(test_graphs)))
    train_loss_hist = []
    train_accu_hist = []
    train_prec_hist = []
    train_recall_hist = []
    test_loss_hist = []
    test_accu_hist = []
    test_prec_hist = []
    test_recall_hist = []
    for epoch in range(cmd_args.num_epochs):
        random.shuffle(train_indices)
        classifier.train()
        avg_score, train_pred, train_labels = loop_dataset(train_graphs,
                                                           classifier,
                                                           train_indices,
                                                           optimizer=optimizer)
        pr_score = compute_pr_scores(train_pred, train_labels, 'train')
        print('\033[92mTrain epoch %d: l %.5f a %.5f p %.5f r %.5f\033[0m' %
              (epoch, avg_score[0], avg_score[1],
               pr_score['precisions'][1], pr_score['recalls'][1]))
        train_loss_hist.append(avg_score[0])
        train_accu_hist.append(avg_score[1])
        train_prec_hist.append(pr_score['precisions'][1])
        train_recall_hist.append(pr_score['recalls'][1])

        classifier.eval()
        test_score, test_pred, test_labels = loop_dataset(test_graphs,
                                                          classifier,
                                                          test_indices)
        pr_score = compute_pr_scores(test_pred, test_labels, 'test')
        print('\033[93mTest epoch %d: l %.5f a %.5f p %.5f r %.5f\033[0m' %
              (epoch, test_score[0], test_score[1],
               pr_score['precisions'][1], pr_score['recalls'][1]))
        test_loss_hist.append(test_score[0])
        test_accu_hist.append(test_score[1])
        test_prec_hist.append(pr_score['precisions'][1])
        test_recall_hist.append(pr_score['recalls'][1])

        if epoch + 1 == cmd_args.num_epochs:
            df = pd.DataFrame.from_dict(pr_score)
            df.to_csv('%s_test_pr_scores.txt' % cmd_args.data,
                      float_format='%.4f')
            store_confusion_matrix(train_pred, train_labels, 'train')
            store_confusion_matrix(test_pred, test_labels, 'test')
            # store_embedding(classifier, train_graphs, 'train')
            # store_embedding(classifier, test_graphs, 'test')

    duration = time.time() - start_time
    log.info('Net training time = %.2f - %.2f = %.2fs' %
             (duration, data_ready_time, duration - data_ready_time))
    torch.save(classifier.state_dict(),
               '%s_%s.model' % (cmd_args.data, cmd_args.mlp_type))
    hist = {}
    hist['train_loss'] = train_loss_hist
    hist['train_accu'] = train_accu_hist
    hist['train_prec'] = train_prec_hist
    hist['train_recall'] = train_recall_hist
    hist['test_loss'] = test_loss_hist
    hist['test_accu'] = test_accu_hist
    hist['test_prec'] = test_prec_hist
    hist['test_recall'] = test_recall_hist
    df = pd.DataFrame.from_dict(hist)
    df.to_csv('%s_hist.txt' % cmd_args.data, float_format='%.6f')
