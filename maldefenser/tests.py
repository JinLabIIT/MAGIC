import glog as log
import numpy as np
import unittest
import time
from cfg_builder import ControlFlowGraphBuilder, AcfgBuilder
from acfg_pipeline import AcfgWorker, AcfgMaster
from dp_utils import delCodeSegLog, evalHexAddSubExpr
from dp_utils import loadBinaryIds, cmpInstDict


class TestCfgBuildedr(unittest.TestCase):
    def setUp(self):
        super(TestCfgBuildedr, self).setUp()
        # self.skipTest('Uncomment me to run this test case')

    @unittest.skip("Uncomment to run")
    def testParseInstructions(self):
        pathPrefix = '../TrainSet'
        binaryIds = [
            '0A32eTdBKayjCWhZqDOQ',
            'exGy3iaKJmRprdHcB0NO',
            '0Q4ALVSRnlHUBjyOb1sw',
            'jERVLnaTwhHFrZbvNfCy',
            'LgeBlyYQAD1NiVGRuxwk',
            '0qjuDC7Rhx9rHkLlItAp',
            '65cjJpPCUQiLDRyXfWd4',
            '426c9FYfeVQbJnygpdKH',
            '5RwWjtmMKlLiXqer8fHG',
            'ELf4J1FhcetA82H0qvTu',
            '5tMCNKDogQ2x7zwUbpcZ',
            '7vS8qWAMU6VzbglhF4r3',
            '1IpWLz6eyhVxDAfQMKEd',
        ]
        for bId in binaryIds:
            log.info('Processing ' + pathPrefix + '/' + bId + '.asm')
            cfgBuilder = ControlFlowGraphBuilder(bId, pathPrefix)
            cfgBuilder.parseInstructions()

    # @unittest.skip("Uncomment to run")
    def testBuildControlFlowGraph(self):
        pathPrefix = '../DataSamples'
        bId = 'test'
        log.info('[Test] Build CFG from ' + pathPrefix + '/' + bId + '.asm')
        cfgBuilder = ControlFlowGraphBuilder(bId, pathPrefix)
        cfgBuilder.buildControlFlowGraph()
        cfgBuilder.exportToNxGraph()
        expectedBlocks = ['-2', '-1', 'ff',
                          '401048', '401050', '401052', '401054',
                          '401064', '40106d', '401076', '401079',
                          '401080', '401084', '401090', '4010a3',
                          '4010a9', '4010ac', '4010ae', '4010b3',
                          '4010b5', '4010b7',
                          ]
        expectedBlocks = [int(x, 16) for x in expectedBlocks]
        edgeDict = {
            '-2': ['401079', '401090'],
            'ff': ['4010b7'],
            '401048': ['401050', '401048'],
            '401050': ['401054'],
            '401054': ['401064'],
            '401064': ['40106d', '4010ae', '401084', '401064'],
            '40106d': ['401076', '401054'],
            '401076': ['401079'],
            '401079': ['401080', '401079', '-2'],
            '401084': ['401090', '4010a3', '401064'],
            '401090': ['4010a9', '-2'],
            '4010a3': ['4010a9'],
            '4010ac': ['4010ae'],
            '4010ae': ['401064', '-2'],
            '4010b3': ['4010b5', '401084'],
            '4010b5': ['4010b7', '-1'],
            '4010b7': ['ff'],
        }
        expectedEdges = []
        for (src, destinations) in edgeDict.items():
            for dst in destinations:
                expectedEdges.append((int(src, 16), int(dst, 16)))

        for block in expectedBlocks:
            self.assertTrue(block in cfgBuilder.cfg.nodes(),
                            '%s not in CFG' % block)
        for edge in expectedEdges:
            self.assertTrue(edge in cfgBuilder.cfg.edges(),
                            '(%s, %s) not in CFG' % (edge[0], edge[1]))

        self.assertEqual(cfgBuilder.cfg.number_of_nodes(), len(expectedBlocks),
                         '#nodes in CFG != expected #nodes')
        self.assertEqual(cfgBuilder.cfg.number_of_edges(), len(expectedEdges),
                         '#edge in CFG != expected #edges')

    @unittest.skip("Uncomment to run")
    def testBuildControlFlowGraphBatch(self):
        pathPrefix = '../TrainSet'
        binaryIds = [
            # '0A32eTdBKayjCWhZqDOQ',
            'exGy3iaKJmRprdHcB0NO',
            # '0Q4ALVSRnlHUBjyOb1sw',
            # 'jERVLnaTwhHFrZbvNfCy',
            # 'LgeBlyYQAD1NiVGRuxwk',
            # '0qjuDC7Rhx9rHkLlItAp',
            # '65cjJpPCUQiLDRyXfWd4',
            # '426c9FYfeVQbJnygpdKH',
            # '5RwWjtmMKlLiXqer8fHG',
            # 'ELf4J1FhcetA82H0qvTu',
            # '5tMCNKDogQ2x7zwUbpcZ',
            # '7vS8qWAMU6VzbglhF4r3',
            # '1IpWLz6eyhVxDAfQMKEd',
        ]
        for bId in binaryIds:
            log.info('Processing ' + pathPrefix + '/' + bId + '.asm')
            cfgBuilder = ControlFlowGraphBuilder(bId, pathPrefix)
            cfgBuilder.buildControlFlowGraph()

    @unittest.skip("Uncomment to run")
    def testEmptyCodeSeg(self):
        pathPrefix = '../TrainSet'
        binaryIds = [
            # Empty ones in trainset
            'da3XhOZzQEbKVtLgMYWv',
            'a9oIzfw03ED4lTBCt52Y',
            'fRLS3aKkijp4GH0Ds6Pv',
            '6tfw0xSL2FNHOCJBdlaA',
            'd0iHC6ANYGon7myPFzBe',
            '58kxhXouHzFd4g3rmInB',
            'fyH8oWql4rg7tEJSLpIB',
            'IidxQvXrlBkWPZAfcqKT',
            'cf4nzsoCmudt1kwleOTI',
            'GXFP0dYvns5NoQtIBECf',
        ]
        # Empty ones in testset
        # ZOtweKduNMynmpiG4brh
        # y5l1PF7qGvsQSDgmRkKn
        # TroLhDaQ2qkKe4XmtPEd
        # spRNUv6MFb8ihB9JXk5r
        # VZ2rzALmJS38uIG5wR1X
        # N2TJvMjcebxGKq1YDC9k
        # xYr76sCtHa2dD48FiGkK
        # YvpzOeBSu7Tmia3wKlLf
        # W8VtX0E95TSzxJuGqiI4
        # uzRUIAil6dVwWsCvhbKD
        # W8aI0V7G5lFTpOgSvjf6
        # pLY05AFladXWQ9fDZnhb
        # QpHV1IWD72EnAyB3FowM
        delCodeSegLog()
        for bId in binaryIds:
            log.info('Processing ' + pathPrefix + '/' + bId + '.asm')
            cfgBuilder = ControlFlowGraphBuilder(bId, pathPrefix)
            seenInst = cfgBuilder.parseInstructions()
            self.assertEqual(len(seenInst), 0, "%s.asm is not empty" % bId)

    # @unittest.skip("Uncomment to run")
    def testEvalHexExpr(self):
        expressions = ['14769F + 48D - 48Dh - 14769Fh+ 14769F',
                       '4477DAB5F7',
                       '435C89+4',
                       '47h -4444DFFF + 4444DFFFh']
        expectedRet = [0x14769F, 0x4477DAB5F7, 0x435C8D, 0x47]
        for expr, expected in zip(expressions, expectedRet):
            self.assertEqual(evalHexAddSubExpr(expr), expected)

    # @unittest.skip("Uncomment to run")
    def testNodeAttributes(self):
        pathPrefix = '../DataSamples'
        bId = 'test'
        log.info('Processing ' + pathPrefix + '/' + bId + '.asm')
        acfgBuilder = AcfgBuilder(bId, pathPrefix)
        features, adjacency = acfgBuilder.getAttributedCfg()
        print(features)
        expRet = np.array([
            #0  1  2  3  4  5  6  7  8  9 10 11 12
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 1],  # -2  0
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # -1  1
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],  # FF  2
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 2, 1],  # 48  3
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1],  # 50  4
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],  # 52  5
            [0, 0, 0, 0, 0, 2, 0, 3, 0, 8, 0, 1, 5],  # 54  6
            [1, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 4, 5],  # 64  7
            [1, 0, 1, 0, 0, 1, 0, 0, 0, 2, 0, 2, 3],  # 6d  8
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1],  # 76  9
            [1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 3, 4],  # 79  10
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 2],  # 80  11
            [1, 0, 1, 1, 0, 2, 0, 0, 1, 3, 0, 3, 6],  # 84  12
            [1, 2, 1, 0, 0, 1, 0, 0, 4, 3, 0, 2, 9],  # 90  13
            [0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 1, 3],  # a3  14
            [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 2],  # a9  15
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1],  # ac  16
            [1, 0, 0, 0, 0, 0, 0, 2, 0, 4, 0, 2, 3],  # ae  17
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 1],  # b3  18
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1],  # b5  19
            [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 2],  # b7  20
        ], dtype=np.float32)
        self.assertEqual(features.shape, expRet.shape, 'unequal shape')
        for (i, row) in enumerate(expRet):
            for (j, item) in enumerate(row):
                self.assertEqual(features[i][j], item, 'at [%d, %d]' % (i, j))


class TestAcfgPipeline(unittest.TestCase):
    def setUp(self):
        super(TestAcfgPipeline, self).setUp()
        # self.skipTest('Uncomment me to run this test case')

    @unittest.skip("Takes about 1 day to")
    def testDiscoverInstDict(self):
        pathPrefix = '../TestSet'
        binaryIds = loadBinaryIds(pathPrefix, None)
        worker = AcfgWorker(pathPrefix, binaryIds)
        worker.discoverInstDictionary('TestSetInstDictionary')

    def testCmpInstDict(self):
        trainDictPath = 'InstDictionary.csv'
        testDictPath = 'TestSetInstDictionary.csv'
        print(cmpInstDict(trainDictPath, testDictPath))

    # @unittest.skip("Uncomment to run")
    def testWorkerRun(self):
        pathPrefix = '../TrainSet'
        binaryIds1 = [
            'exGy3iaKJmRprdHcB0NO',
            '0Q4ALVSRnlHUBjyOb1sw',
        ]
        binaryIds2 = [
            'cqdUoQDaZfGkt5ilBe7n',
            'BKpbxgMPWUNZosdnO8Ak',
        ]
        worker1 = AcfgWorker(pathPrefix, binaryIds1)
        worker1.start()
        worker2 = AcfgWorker(pathPrefix, binaryIds2)
        worker2.start()

        worker1.join()
        worker2.join()

    def testAggregateDgcnnFormat(self):
        pathPrefix = '../DataSamples'
        labelPath = '../trainLabels.csv'
        binaryIds = ['test',]
        master = AcfgMaster(pathPrefix, labelPath, 'test', binaryIds)
        master.bId2Label['test'] = '1'
        master.dispatchWorkers(1)
        expectedRet = [
            [1],      # number of graphs
            [21, 1],  # number of nodes, label of graph
            #               0  1  2  3  4  5  6  7  8  9 10 11 12
            [1,2,10,13,     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 1.0],  # -2  0
            [1,0,           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],  # -1  1
            [1,1,20,        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],  # FF  2
            [1,2,3,4,       1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 1.0],  # 48  3
            [1,1,6,         1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0],  # 50  4
            [1,0,           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # 52  5
            [1,1,7,         0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0, 8.0, 0.0, 1.0, 5.0],  # 54  6
            [1,4,7,8,12,17, 1.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 4.0, 5.0],  # 64  7
            [1,2,6,9,       1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 2.0, 3.0],  # 6d  8
            [1,1,10,        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],  # 76  9
            [1,3,0,10,11,   1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 3.0, 4.0],  # 79  10
            [1,0,           0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0],  # 80  11
            [1,3,7,13,14,   1.0, 0.0, 1.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 3.0, 0.0, 3.0, 6.0],  # 84  12
            [1,2,0,15,      1.0, 2.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 4.0, 3.0, 0.0, 2.0, 9.0],  # 90  13
            [1,1,15,        0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0],  # a3  14
            [1,0,           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0],  # a9  15
            [1,1,17,        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0],  # ac  16
            [1,2,0,7,       1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 4.0, 0.0, 2.0, 3.0],  # ae  17
            [1,2,12,19,     1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 1.0],  # b3  18
            [1,2,1,20,      1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0],  # b5  19
            [1,1,2,         0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0],  # b7  20
        ]
        expLines = []
        for row in expectedRet:
            strElems = [str(x) for x in row]
            expLines.append(" ".join(strElems))

        with open(pathPrefix + '/' + 'test.txt') as file:
            lineNum = 1
            for line in file:
                resultLine = line.rstrip('\n')
                self.assertEqual(expLines[lineNum - 1], resultLine,
                                 'L%d exp != result' % lineNum)
                lineNum += 1

    # @unittest.skip("Uncomment to run")
    def testMasterDispatch(self):
        pathPrefix = '../TrainSet'
        labelPath = '../trainLabels.csv'
        binaryIds = [
            'exGy3iaKJmRprdHcB0NO',
            '0Q4ALVSRnlHUBjyOb1sw',
            'cqdUoQDaZfGkt5ilBe7n',
            'BKpbxgMPWUNZosdnO8Ak',
        ]
        master = AcfgMaster(pathPrefix, labelPath,
                            'TestMasterDispatch', binaryIds)
        master.dispatchWorkers(4)

    # @unittest.skip("Uncomment to run")
    def testIfSkipEmptyCfgs(self):
        pathPrefix = '../TrainSet'
        labelPath = '../trainLabels.csv'
        binaryIds = [
            'da3XhOZzQEbKVtLgMYWv',
            'a9oIzfw03ED4lTBCt52Y',
            'fRLS3aKkijp4GH0Ds6Pv',
            '6tfw0xSL2FNHOCJBdlaA',
            'd0iHC6ANYGon7myPFzBe',
            '58kxhXouHzFd4g3rmInB',
            'fyH8oWql4rg7tEJSLpIB',
            'IidxQvXrlBkWPZAfcqKT',
            'cf4nzsoCmudt1kwleOTI',
            'GXFP0dYvns5NoQtIBECf',
        ]
        master = AcfgMaster(pathPrefix, labelPath,
                            'TestIfSkipEmptyCfgs', binaryIds)
        master.dispatchWorkers(1)
        with open(pathPrefix + '/TestIfSkipEmptyCfgs.txt') as f:
            content = f.read()
            self.assertEqual(int(content), 0, '#graphs should be zero')

    # @unittest.skip("Uncomment to run")
    def testValidAddrFormat(self):
        pathPrefix = '../TrainSet'
        labelPath = '../trainLabels.csv'
        binaryIds = [
            '1x2u5Ws7tzFRAgyqoJBV',
        ]
        master = AcfgMaster(pathPrefix, labelPath,
                            'TestValidAddrFormat', binaryIds)
        master.dispatchWorkers(1)


class TestAcfgRunningTime(unittest.TestCase):
    def setUp(self):
        super(TestAcfgRunningTime, self).setUp()
        self.skipTest('Measuring running time may takes hours or days. Don\'t run me unless you are sure')

    def testRunningTime(self):
        pathPrefix = '../TrainSet'
        labelPath = '../trainLabels.csv'

        master1 = AcfgMaster(pathPrefix, labelPath, 'TestRunningTime1')
        start = time.process_time()
        master1.dispatchWorkers(1)
        runtime1 = time.process_time() - start
        log.info(f'Running time of 1-thread: {runtime1} seconds')

        master2 = AcfgMaster(pathPrefix, labelPath, 'TestRunningTime8')
        start = time.process_time()
        master2.dispatchWorkers(8)
        runtime2 = time.process_time() - start
        log.info(f'Running time of 8-thread: {runtime2} seconds')


if __name__ == '__main__':
    log.setLevel("INFO")
    unittest.main()
