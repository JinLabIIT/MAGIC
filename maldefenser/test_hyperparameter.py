import numpy as np
import glog as log
import networkx as nx
import unittest
from hyperparameters import HyperParameterIterator
from ml_utils import S2VGraph, normalizeFeatures


class TestHpyerParameterIterator(unittest.TestCase):
    def testIterator(self):
        cnt = 0
        iter = HyperParameterIterator('full_gpu1.hp')
        for hyperparameter in iter:
            log.info(hyperparameter)
            cnt += 1

        self.assertEqual(cnt, iter.getLimit(), '#hyper-combination returned')


class TestMlUtils(unittest.TestCase):
    def testMinMaxNormalize(self):
        graphs = []
        for i in range(11):
            nodeFeatures = np.random.randint(low=0, high=100, size=(i + 1, 13))
            g = S2VGraph('binaryId', g=nx.Graph(),
                         label='label', node_tags=[],
                         node_features=nodeFeatures)
            graphs.append(g)

        normalizeFeatures(graphs, operation='min_max')
        for (i, g) in enumerate(graphs):
            log.debug(f'Feature shape after norm: {g.node_features.shape}')
            self.assertEqual(g.node_features.shape[0], i + 1, 'First dim size')
            self.assertEqual(g.node_features.shape[1], 13, 'Second dim size')
            for x in np.nditer(g.node_features):
                self.assertLessEqual(x, 1.0, 'Not <= 1')
                self.assertGreaterEqual(x, 0.0, 'Not >= 0')

    def testZeroMeanNormalize(self):
        graphs = []
        for i in range(11):
            nodeFeatures = np.random.randint(low=0, high=100, size=(i + 1, 13))
            g = S2VGraph('binaryId', g=nx.Graph(),
                         label='label', node_tags=[],
                         node_features=nodeFeatures)
            graphs.append(g)

        normalizeFeatures(graphs, operation='zero_mean')
        for (i, g) in enumerate(graphs):
            log.debug(f'Feature shape after norm: {g.node_features.shape}')
            self.assertEqual(g.node_features.shape[0], i + 1, 'First dim size')
            self.assertEqual(g.node_features.shape[1], 13, 'Second dim size')


if __name__ == '__main__':
    log.setLevel("INFO")
    unittest.main()
