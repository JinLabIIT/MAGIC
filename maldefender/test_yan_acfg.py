import glog as log
import numpy as np
import unittest
import time
from yan_nx2acfg import iterAllDirectories


class TestYanAcfgFormatter(unittest.TestCase):
    def setUp(self):
        super(TestYanAcfgFormatter, self).setUp()
        # self.skipTest('Uncomment me to skip this test case')

    # @unittest.skip("Uncomment to run")
    def testAcfg2DgcnnFormat(self):
        expectedRet = [
            [1],      # number of graphs
            [7, 0, 'Email-Worm.Win32.Bagle.bd'],  # number of nodes, label of graph, graph id
            #            0    1    2    3    4    5    6    7    8    9   10   11   12
            [1,0,      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 00  0
            [1,1,2,    0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 6.0, 0.0, 0.0, 1.0, 8.0],  # 56  1
            [1,1,3,    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0],  # 74  2
            [1,0,      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0],  # 79  3
            [1,0,      1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],  # 74  4
            [1,1,6,    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],  # 81  5
            [1,1,4,    1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 3.0],  # 86  6
        ]
        expLines = []
        for row in expectedRet:
            strElems = [str(x) for x in row]
            expLines.append(" ".join(strElems))

        outputDir = '../DataSamples/TestAcfg2DgcnnFormat'
        iterAllDirectories(cfgDirPrefix='../DataSamples', outputDir=outputDir)
        with open(outputDir + '/' + 'YANACFG.txt') as file:
            lineNum = 1
            for line in file:
                resultLine = line.rstrip('\n')
                self.assertEqual(expLines[lineNum - 1],
                                 resultLine,
                                 'unequal L%d' % lineNum)
                lineNum += 1


if __name__ == '__main__':
    log.setLevel("INFO")
    unittest.main()
