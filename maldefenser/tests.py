import glog as log
import numpy as np
import unittest
from cfg_builder import ControlFlowGraphBuilder, AcfgBuilder
from acfg_pipeline import AcfgWorker
from utils import delCodeSegLog, evalHexAddSubExpr


class TestCfgBuildedr(unittest.TestCase):
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
        log.info('Processing ' + pathPrefix + '/' + bId + '.asm')
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
        expectedBlocks = ["%08X" % int(x, 16) for x in expectedBlocks]
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
                expectedEdges.append(("%08X" % int(src, 16),
                                      "%08X" % int(dst, 16)))

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
            'a9oIzfw03ED4lTBCt52Y',
            'da3XhOZzQEbKVtLgMYWv',
        ]
        delCodeSegLog()
        for bId in binaryIds:
            log.info('Processing ' + pathPrefix + '/' + bId + '.asm')
            cfgBuilder = ControlFlowGraphBuilder(bId, pathPrefix)
            cfgBuilder.parseInstructions()

    @unittest.skip("Uncomment to run")
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
    @unittest.skip("Uncomment to run")
    def testDiscoverInstDict(self):
        pathPrefix = '../TrainSet'
        binaryIds = [
            'exGy3iaKJmRprdHcB0NO',
            '0Q4ALVSRnlHUBjyOb1sw',
            'cqdUoQDaZfGkt5ilBe7n',
            'BKpbxgMPWUNZosdnO8Ak',
        ]
        worker = AcfgWorker(pathPrefix, binaryIds)
        worker.discoverInstDictionary('ut_seen_inst')

    # @unittest.skip("Uncomment to run")
    def testWorkerRun(self):
        pathPrefix = '../TrainSet'
        labelPath = '../trainLabels.csv'
        binaryIds1 = [
            'exGy3iaKJmRprdHcB0NO',
            '0Q4ALVSRnlHUBjyOb1sw',
        ]
        binaryIds2 = [
            'cqdUoQDaZfGkt5ilBe7n',
            'BKpbxgMPWUNZosdnO8Ak',
        ]
        worker1 = AcfgWorker(pathPrefix, binaryIds1, labelPath)
        worker1.start()
        worker2 = AcfgWorker(pathPrefix, binaryIds2, labelPath)
        worker2.start()
        
        worker1.join()
        worker2.join()


if __name__ == '__main__':
    log.setLevel("INFO")
    unittest.main()
