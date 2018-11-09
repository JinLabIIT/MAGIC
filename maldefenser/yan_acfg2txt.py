
# coding: utf-8

# ### Convert All ACFGs

# In[ ]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle as pkl
import numpy as np
import scipy as sp
import scipy.sparse
import pandas as pd

"""
Read files under input_dir, aggregate ACFGs into
the txt format defined in https://github.com/littlepretty/pytorch_DGCNN/tree/master/data.
"""
input_dir = 'AllAcfg/'
output_dir = 'PytorchDGCNN/data/ACFG/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(input_dir + 'graph_pathnames.csv', 'rb') as f:
    graph_pathnames = [line.strip() for line in f]
    
print(graph_pathnames[100:120])


# In[ ]:


graph_sizes = pd.read_csv(input_dir + 'graph_sizes.csv', header=0)
print(graph_sizes)

label_names = sorted(list(graph_sizes.columns))
print(label_names)
graph_cnts = graph_sizes.count()
print("Input: %s, Output: %s\n%s" % (input_dir, output_dir, graph_cnts))


# In[ ]:


def list2str(l1, l2):
    """
    Merge two list, then return space seperated string format.
    """
    return " ".join([str(x) for x in (list(l1) + list(l2))])


output = open(output_dir + 'ACFG.txt', 'wb')
output.write("%d\n" % sum(graph_cnts.tolist()))
test_cnt = 0
for graph_pathname in graph_pathnames:
    label = label_names.index(graph_pathname.split('/')[1])
    graph_id = graph_pathname.split('/')[2][:-8] # ignore .gpickle
    
    features = np.loadtxt(input_dir + graph_id + '.features.txt', dtype=int, ndmin=2)
    sp_adjacent_mat = sp.sparse.load_npz(input_dir + graph_id + '.adjacent.npz')
    output.write("%d %d\n" % (features.shape[0], label))
    test_cnt += 1
    
    sp_adjacent = sp.sparse.find(sp_adjacent_mat)
    indices = {}
    for i in range(len(sp_adjacent[0])):
        if sp_adjacent[0][i] not in indices:
            indices[sp_adjacent[0][i]] = []
            
        indices[sp_adjacent[0][i]].append(sp_adjacent[1][i])
        
    for (i, feature) in enumerate(features):
        neighbors = indices[i] if i in indices else []
        output.write("1 %d %s\n" % (len(neighbors), list2str(neighbors, feature)))

output.close()
print("[Finished] Convert %d ACFGs to DGCNN txt format" % test_cnt)

