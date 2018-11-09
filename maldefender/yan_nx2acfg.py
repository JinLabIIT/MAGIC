#/usr/bin/python3.7
import glob
import glog as log
import os
import pickle as pkl
import numpy as np
import scipy as sp
import pandas as pd
from collections import Counter
from networkx import number_of_nodes, adjacency_matrix
from node_attributes import node_features

class_dirnames = glob.glob('../IdaPro/AllCfg/*')
print(class_dirnames)
class_names = sorted(['Rbot', 'Koobface', 'Sdbot', 'Swizzor', 'Lmir',
                      'Bagle', 'Zbot', 'Bifrose', 'Ldpinch', 'Hupigon',
                      'Benign', 'Vundo', 'Zlob'])
graph_sizes = {x: [] for x in class_names}
output_dir = '../IdaPro/AllAcfg/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print('mkdir %s' % output_dir)

graph_pathnames = []
print(str(len(class_names)) + " types of CFGs: " + str(class_names))


"""Path name format: class/graph_id/pkl_name"""
graph_pathnames_output = open(output_dir + 'graph_pathnames.csv', 'wb')
total = 0
for class_dirname in class_dirnames:
    class_name = class_dirname.split('/')[1]
    if class_name not in class_names:
        continue

    print("Processing %s CFGs" % class_name)
    pkl_pathnames = glob.glob(class_dirname + '/*')

    if len(pkl_pathnames) == 0:
        print('[Warning] %s is empty' % data_dirname)

    for pkl_pathname in pkl_pathnames:
        if pkl_pathname[-8:] != '.gpickle':
            print('[Warning] %s is not gpickle file' % pkl_pathname)
            continue

        total += 1
        graph_pathnames_output.write(pkl_pathname + '\n')

        G = pkl.load(open(pkl_pathname, 'rb'))
        graph_sizes[class_name].append(number_of_nodes(G))
        graph_id = pkl_pathname.split('/')[2][:-8] # ignore '.gpickle'

        features = node_features(G)
        np.savetxt(output_dir + graph_id + '.features.txt', features, fmt="%d")
        np.savetxt(output_dir + graph_id + '.label.txt', np.array([class_name]), fmt="%s")
        # np.savetxt(output_dir + graph_id + '.adjacent.txt', adjacency_matrix(G).todense(), fmt="%d")
        sp.sparse.save_npz(output_dir + graph_id + '.adjacent', adjacency_matrix(G))

print("%d CFGs" % total)
graph_size_pd = pd.DataFrame.from_dict(graph_sizes, orient='index').T
graph_size_pd.to_csv(output_dir + 'graph_sizes.csv', index=False, header=True)


# In[6]:


from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv(output_dir + 'graph_sizes.csv', header=0)
label_cnts = {key: data[key].count() for key in class_names}
print("Graph size distribution by class:")
print(label_cnts)
print("Total #graphs:", sum(label_cnts.values()))

max_graph_sizes = data.max()
print("Max graph size for each class:")
print(max_graph_sizes)


# In[18]:


def plot_hist_in_range(data, left=1, right=200):
    for column in data:
        df = data[column].dropna()
        max_num_nodes = df.max()
        plt.hist(df, bins=np.arange(left, right, 1), density=False,
                 histtype='step', label=column)

    plt.legend()
    plt.title('Histogram of graph size')
    plt.grid(True)
    plt.show()

max_graph_size = max(data.max())
print("maximum graph size =", max_graph_size)
plot_hist_in_range(data, right=600)
