import glog as log
import numpy as np
import unittest
import time
from instructions import Instruction
from cfg_builder import ControlFlowGraphBuilder, AcfgBuilder
from acfg_pipeline import AcfgWorker, AcfgMaster
from dp_utils import delCodeSegLog, evalHexAddSubExpr
from dp_utils import loadBinaryIds, cmpInstDict
from hyperparameters import HyperParameterIterator


def featuresInTestAsm():
    op = np.array([
            #0  1  2  3  4  5  6  7  8  9 10
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # -2  0
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # -1  1
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # FF  2
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],  # 48  3
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 50  4
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 52  5
            [0, 0, 0, 0, 0, 2, 0, 3, 0, 8, 0],  # 54  6
            [1, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0],  # 64  7
            [1, 0, 1, 0, 0, 1, 0, 0, 0, 2, 0],  # 6d  8
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],  # 76  9
            [1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0],  # 79  10
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],  # 80  11
            [1, 0, 1, 1, 0, 2, 0, 0, 1, 3, 0],  # 84  12
            [1, 2, 1, 0, 0, 1, 0, 0, 4, 3, 0],  # 90  13
            [0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0],  # a3  14
            [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],  # a9  15
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0],  # ac  16
            [1, 0, 0, 0, 0, 0, 0, 2, 0, 4, 0],  # ae  17
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # b3  18
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # b5  19
            [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # b7  20
        ], dtype=np.float32)

    oneGram = np.zeros((op.shape[0], 3))
    oneGramIndices = [
        [],
        [],
        [],
        [],  #48  3
        [],  #50  4
        [],  #52  5
        [0, 0, 0, 0, 1,],  # 54  6
        [],  #64  7
        [],  #6D 8
        [],  #76 9
        [],  #79 10
        [],  #80 11
        [0, 0, 0,],  #84 12
        [1, 0, 0, 0,],  #90 13
        [],  #A3 14
        [],  #A9 15
        [
            0, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2
        ],  #AC 16
        [1],  #AE 17
        [],  #B3 18
        [],  #B5 19
        [],  #B7 20
    ]
    for i, indices in enumerate(oneGramIndices):
        for j in indices:
            oneGram[i, j] += 1

    fourGram = np.zeros((op.shape[0], 10))
    fourGramIndices = [
        [],[],[],
        [],  # 48  3
        [],  # 50  4
        [],  # 52  5
        [4, 5, 8],  # 54  6
        [],  #64  7
        [],  #6D  8
        [],  #76  9
        [],  #79 10
        [],  #80 11
        [2, 6],  #84 12
        [1, 7],  #90 13
        [3],  #A3, 14
        [],  #A9 15
        [0]*45,  #AC 16
        [],  #AE 17
        [],  #B3 18
        [],  #B5 19
        [9]  #B7 20
    ]
    for i, indices in enumerate(fourGramIndices):
        for j in indices:
            fourGram[i, j] += 1

    ngram = np.zeros((op.shape[0], 257))
    nonZeroIndices = [
        [],
        [],
        [],
        [0x8E, 0x7D],  # 48  3
        [0x5E, 0x47],  # 50  4
        [0xC2, 0x04],  # 52  5
        [
            0x00, 0x10, 0x00, 0x00, 0xF0, 0xF0, 0x00, 0x8B, 0xFF, 0x55,
            0x8B, 0xDD, 0x8B, 0x44
        ],  # 54  6
        [0x8A, 0x08, 0x8B, 0x54, 0x88, 0x0B, 0x88, 0x0A, 0xC3,
         0x55],  # 64  7
        [0xCC, 0xCC, 0x8B, 0x44, 0x8D, 0x50],  #6D 8
        [0x8D, 0x50],  #76 9
        [0x8A, 0x08, 0x40, 0x45, 0x84, 0xC9, 0x75, 0xF9],  #79 10
        [0x2B, 0xC2, 0xC3, 0x45],  #80 11
        [
            0x5D, 0xC3, 0x00, 0x00, 0x08, 0x56, 0x0D, 0x2F, 0x06, 0x00,
            0x75, 0x1D
        ],  #84 12
        [
            0x5D, 0xC3, 0x8B, 0xFF, 0x56, 0x04, 0x00, 0x00, 0x00, 0x83,
            0xC4, 0x14, 0x6A, 0x16, 0x58, 0xEB, 0x0A
        ],  #90 13
        [0xF0, 0xF0, 0xF0, 0x01, 0x33, 0xC0],  #A3 14
        [0x5E, 0xC3],  #A9 15
        [
            0x00, 0x100, 0x100, 0x100, 0x100, 0x100, 0x100, 0x100, 0x100, 0x100,
            0x100, 0x100, 0x100, 0x100, 0x100, 0x100, 0x100, 0x100, 0x100,
            0x100, 0x100, 0x100, 0x100, 0x100, 0x100, 0x100, 0x100, 0x100,
            0x100, 0x100, 0x100, 0x100, 0x100, 0x100, 0x100, 0x100, 0x100,
            0x100, 0x100, 0x100, 0x100, 0x100, 0x100, 0x100, 0x100, 0x100,
            0x100, 0x100, 0x100
        ],  #AC 16
        [0x8B, 0xFF],  #AE 17
        [0x8B, 0x45, 0x08],  #B3 18
        [0x56],  #B5 19
        [0x84, 0x0D, 0x2F, 0x06],  #B7 20
    ]
    for i, indices in enumerate(nonZeroIndices):
        for j in indices:
            ngram[i, j] += 1

    spChars = np.zeros((op.shape[0], len(Instruction.specialChars)))
    spCharsInBlock = [''] * op.shape[0]
    spCharsInBlock[6] = '?@???@?$@@@@@$@@@@@@@@@@[]'
    spCharsInBlock[7] = '[][]'
    spCharsInBlock[8] = '[]'
    spCharsInBlock[9] = '[]'
    spCharsInBlock[10] = '[]??'
    spCharsInBlock[12] = '[]'
    spCharsInBlock[13] = '[]'
    spCharsInBlock[14] = '[]'
    spCharsInBlock[16] = '?'
    spCharsInBlock[17] = '[]'
    for i, chars in enumerate(spCharsInBlock):
        for c in chars:
            spChars[i][Instruction.spChar2Idx[c]] += 1

    degree = np.array([[2, 0, 1, 2, 1, 0, 1, 4, 2, 1, 3, 0, 3, 2, 1, 0, 1, 2, 2, 2, 1]])
    numInsts = np.array([[1, 1, 1, 1, 1, 1, 5, 5, 3, 1, 4, 2, 6, 9, 3, 2, 1, 3, 1, 1, 2]])
    return np.concatenate((op, oneGram, fourGram, spChars,
                           degree.T, numInsts.T), axis=1)


class TestCfgBuildedr(unittest.TestCase):
    def setUp(self):
        super(TestCfgBuildedr, self).setUp()
        # self.skipTest('Uncomment me to skip this test case')

    @unittest.skip("Comment to run")
    def testParseInstructions(self):
        pathPrefix = '../../MSACFG/TrainSet'
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

    # @unittest.skip("Comment to run")
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

    @unittest.skip("Comment to run")
    def testBuildControlFlowGraphBatch(self):
        pathPrefix = '../../MSACFG/TrainSet'
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

    @unittest.skip("Comment to run")
    def testEmptyCodeSeg(self):
        pathPrefix = '../../MSACFG/TrainSet'
        binaryIds = [
            # Empty ones in trainset: BinaryId => label
            'da3XhOZzQEbKVtLgMYWv',  # => 1
            'a9oIzfw03ED4lTBCt52Y',  # => 1
            'fRLS3aKkijp4GH0Ds6Pv',  # => 1
            '6tfw0xSL2FNHOCJBdlaA',  # => 1
            'd0iHC6ANYGon7myPFzBe',  # => 1
            '58kxhXouHzFd4g3rmInB',  # => 1
            'fyH8oWql4rg7tEJSLpIB',  # => 5
            'IidxQvXrlBkWPZAfcqKT',  # => 1
            'cf4nzsoCmudt1kwleOTI',  # => 1
            'GXFP0dYvns5NoQtIBECf',  # => 6
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

    @unittest.skip("Comment to run")
    def testEvalHexExpr(self):
        expressions = ['14769F + 48D - 48Dh - 14769Fh+ 14769F',
                       '4477DAB5F7',
                       '435C89+4',
                       '47h -4444DFFF + 4444DFFFh']
        expectedRet = [0x14769F, 0x4477DAB5F7, 0x435C8D, 0x47]
        for expr, expected in zip(expressions, expectedRet):
            self.assertEqual(evalHexAddSubExpr(expr), expected)

    # @unittest.skip("Comment to run")
    def testNodeAttributes(self):
        pathPrefix = '../DataSamples'
        bId = 'test'
        log.info('Processing ' + pathPrefix + '/' + bId + '.asm')
        acfgBuilder = AcfgBuilder(bId, pathPrefix)
        features, adjacency = acfgBuilder.getAttributedCfg()
        for line in features:
            log.debug(list(line))

        expRet = featuresInTestAsm()
        self.assertEqual(features.shape, expRet.shape, 'unequal shape')
        for (i, row) in enumerate(expRet):
            for (j, item) in enumerate(row):
                self.assertEqual(features[i][j], item, 'at [%d, %d]' % (i, j))


class TestAcfgPipeline(unittest.TestCase):
    def setUp(self):
        super(TestAcfgPipeline, self).setUp()
        # self.skipTest('Uncomment me to skip this test case')

    @unittest.skip("Takes about 1 day to finish")
    def testDiscoverInstDict(self):
        pathPrefix = '../../MSACFG/TestSet'
        binaryIds = loadBinaryIds(pathPrefix, None)
        worker = AcfgWorker(pathPrefix, binaryIds)
        worker.discoverInstDictionary('TestSetInstDictionary')

    @unittest.skip("Comment to run")
    def testCmpInstDict(self):
        trainDictPath = 'InstDictionary.csv'
        testDictPath = 'TestSetInstDictionary.csv'
        print(cmpInstDict(trainDictPath, testDictPath))

    @unittest.skip("Comment to run")
    def testWorkerRun(self):
        pathPrefix = '../../MSACFG/TrainSet'
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
        binaryIds = ['test',]
        master = AcfgMaster(pathPrefix, None, 'test', binaryIds)
        master.dispatchWorkers(1)
        featureVectors = featuresInTestAsm()
        expectedRet = [
            [1],      # number of graphs
            [21, '?', 'test'],  # number of nodes, label of graph, graph_id
            [1,2,10,13,     ],  # -2  0
            [1,0,           ],  # -1  1
            [1,1,20,        ],  # FF  2
            [1,2,3,4,       ],  # 48  3
            [1,1,6,         ],  # 50  4
            [1,0,           ],  # 52  5
            [1,1,7,         ],  # 54  6
            [1,4,7,8,12,17, ],  # 64  7
            [1,2,6,9,       ],  # 6d  8
            [1,1,10,        ],  # 76  9
            [1,3,0,10,11,   ],  # 79  10
            [1,0,           ],  # 80  11
            [1,3,7,13,14,   ],  # 84  12
            [1,2,0,15,      ],  # 90  13
            [1,1,15,        ],  # a3  14
            [1,0,           ],  # a9  15
            [1,1,17,        ],  # ac  16
            [1,2,0,7,       ],  # ae  17
            [1,2,12,19,     ],  # b3  18
            [1,2,1,20,      ],  # b5  19
            [1,1,2,         ],  # b7  20
        ]
        for (i, featureVector) in enumerate(featureVectors):
            for feature in featureVector:
                expectedRet[i + 2].append(feature)

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

    # @unittest.skip("Comment to skip")
    def testMasterDispatch(self):
        pathPrefix = '../../MSACFG/TrainSet'
        labelPath = '../../MSACFG/trainLabels.csv'
        binaryIds = [
            'cqdUoQDaZfGkt5ilBe7n',
            'jgOs7KiB0aTEzvSUJVPp',
            '6RQtx0X42zOelTDaZnvi',
            'HaTioeY3kbvJW2LXtOwF',
            'Fnda3PuqJT6Ep5vjOWCk',
            'exGy3iaKJmRprdHcB0NO',
            'bZz2OoQmqx0PdGBhaHKk',
            '0Q4ALVSRnlHUBjyOb1sw',
            'hIkK1vBdj9fDJPcUWzA8',
        ]
        labels = ['1', '2', '3', '4', '5','6', '7', '8', '9']
        master = AcfgMaster(pathPrefix, labelPath,
                            'TestMasterDispatch', binaryIds)
        master.dispatchWorkers(3)
        for (i, bId) in enumerate(master.binaryIds):
            log.info(f'label({bId}) = {master.bId2Label[bId]}')
            self.assertEqual(master.bId2Label[bId], labels[i], 'Wrong label')

    def testMasterTestSet(self):
        pathPrefix = '../../MSACFG/TestSet'
        labelPath = None
        binaryIds = [
            'mkJhUOLaM2BnXG3S4cb5',
            'OuCDNHRltBVZJGzv4KI8',
            'R1hJ8LGFtZj35BlNxm7q',
            'tdsNQiObHye2g1JfoB84',
            'mnNcAzZ6aCpsvd8D30tT',
        ]
        master = AcfgMaster(pathPrefix, labelPath,
                            'TestMasterTestSet', binaryIds)
        master.dispatchWorkers(1)

    @unittest.skip("Comment to run")
    def testIfSkipEmptyCfgs(self):
        pathPrefix = '../../MSACFG/TrainSet'
        labelPath = '../../MSACFG/trainLabels.csv'
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

    # @unittest.skip("Comment to run")
    def testValidAddrFormat(self):
        pathPrefix = '../../MSACFG/TrainSet'
        labelPath = '../../MSACFG/trainLabels.csv'
        binaryIds = [
            '1x2u5Ws7tzFRAgyqoJBV',
            'i4f81CyIkZtEprWaOVRS',
        ]
        master = AcfgMaster(pathPrefix, labelPath,
                            'TestValidAddrFormat', binaryIds)
        master.dispatchWorkers(1)


class TestAcfgRunningTime(unittest.TestCase):
    def setUp(self):
        super(TestAcfgRunningTime, self).setUp()
        self.skipTest('Measuring running time may takes hours or days. Don\'t run me unless you are sure')

    def testRunningTime(self):
        pathPrefix = '../../MSACFG/TrainSet'
        labelPath = '../../MSACFG/trainLabels.csv'

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
