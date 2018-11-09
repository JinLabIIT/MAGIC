#!/usr/bin/python3.7
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
from ml_utils import cmd_args, gHP, storeConfusionMatrix
from ml_utils import computePrScores, loadGraphsMayCache, kFoldSplit
from e2e_model import Classifier
from hyperparameters import HyperParameterIterator


if __name__ == '__main__':
    log.setLevel("INFO")

    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    start_time = time.process_time()
    graphs = loadGraphsMayCache(isTestSet=True)
    data_ready_time = time.process_time() - start_time
    log.info('Test dataset ready takes %.2fs' % data_ready_time)
