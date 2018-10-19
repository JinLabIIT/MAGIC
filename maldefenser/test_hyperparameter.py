import glog as log
import unittest
from hyperparameters import HyperParameterIterator


class TestHpyerParameterIterator(unittest.TestCase):
    def testIterator(self):
        cnt = 0
        iter = HyperParameterIterator('hp.txt')
        for hyperparameter in iter:
            log.info(hyperparameter)
            cnt += 1

        self.assertEqual(cnt, iter.getLimit(), '#hyper-combination returned')


if __name__ == '__main__':
    log.setLevel("INFO")
    unittest.main()
