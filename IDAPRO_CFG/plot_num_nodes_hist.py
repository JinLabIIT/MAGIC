from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('num_nodes_hist.csv')
max_num_nodes = data['num_nodes'].max()
plt.hist(data['num_nodes'], bins=np.arange(0, max_num_nodes, 1),
         histtype='step', label='num_nodes')
plt.legend()
plt.title('Histogram of num_nodes')
plt.grid(True)
plt.show()
