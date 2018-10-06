import glog as log
import numpy as np
import unittest
from cfg_builder import ControlFlowGraphBuilder
from process_graphs import DataProvider
from node_attributes import nodeFeatures
from utils import delCodeSegLog, evalHexAddSubExpr


class TestCfgBuildedr(unittest.TestCase):
    @unittest.skip("Uncomment to run")
    def test_parseInstructions(self):
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
    def test_buildSingle(self):
        pathPrefix = '../DataSamples'
        bId = 'test'
        log.info('Processing ' + pathPrefix + '/' + bId + '.asm')
        cfgBuilder = ControlFlowGraphBuilder(bId, pathPrefix)
        cfgBuilder.buildControlFlowGraph()
        expectedBlocks = ['-2', '-1', 'ff',
                          '401048', '401050', '401052', '401054',
                          '401064', '40106d', '401076', '401079',
                          '401080', '401084', '401090', '4010a3',
                          '4010a9', '4010ac', '4010ae', '4010b3',
                          '4010b5', '4010b7',
                          ]
        expectedBlocks = ["%08X" % int(x, 16) for x in expectedBlocks]
        edgeDict = {'401048': ['401050', '401048'],
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
                    'ff': ['4010b7'],
                    '-2': ['401079', '401090'],}
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
    def test_build(self):
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
    def test_emptyCodeSeg(self):
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
    def test_discoverInstDict(self):
        pathPrefix = '../TrainSet'
        binaryIds = [
            'exGy3iaKJmRprdHcB0NO',
            '0Q4ALVSRnlHUBjyOb1sw',
            'cqdUoQDaZfGkt5ilBe7n',
            'BKpbxgMPWUNZosdnO8Ak',
        ]
        dataProvider = DataProvider(pathPrefix)
        dataProvider.discoverInstDictionary(binaryIds, 'ut_seen_inst')

    @unittest.skip("Uncomment to run")
    def test_evalHexExpr(self):
        expressions = ['14769F + 48D - 48Dh - 14769Fh+ 14769F',
                       '4477DAB5F7',
                       '435C89+4',
                       '47h -4444DFFF + 4444DFFFh']
        expectedRet = [0x14769F, 0x4477DAB5F7, 0x435C8D, 0x47]
        for expr, expected in zip(expressions, expectedRet):
            self.assertEqual(evalHexAddSubExpr(expr), expected)

    # @unittest.skip("Uncomment to run")
    def test_nodeAttributes(self):
        pathPrefix = '../DataSamples'
        bId = 'test'
        log.info('Processing ' + pathPrefix + '/' + bId + '.asm')
        cfgBuilder = ControlFlowGraphBuilder(bId, pathPrefix)
        cfgBuilder.buildControlFlowGraph()
        features = nodeFeatures(cfgBuilder.cfg)
        print(features)
        expRet = np.array([
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # -2
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # -1
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # FF
            [1, 0, 0, 0, 0, 1, 0, 0, 1, 0],  # 48
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 50
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 52
            [0, 0, 0, 0, 0, 2, 0, 0, 7, 0],  # 54
            [1, 2, 0, 0, 0, 2, 0, 0, 0, 0],  # 64
            [1, 0, 1, 0, 0, 1, 0, 0, 2, 0],  # 6d
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 76
            [1, 1, 1, 0, 0, 1, 0, 0, 1, 0],  # 79
            [1, 0, 1, 0, 0, 0, 1, 0, 1, 0],  # 80
            [1, 0, 1, 1, 0, 2, 0, 0, 3, 0],  # 84
            [1, 1, 1, 0, 0, 1, 0, 0, 3, 0],  # 90
            [0, 0, 1, 0, 0, 3, 0, 0, 0, 0],  # a3
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # a9
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # ac
            [1, 0, 0, 0, 0, 0, 0, 0, 4, 0],  # ae
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # b3
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # b5
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # b7
        ], dtype=np.float32)
        for (i, row) in enumerate(expRet):
            for (j, item) in enumerate(row):
                self.assertEqual(features[i][j], item, 'at [%d, %d]' % (i, j))


if __name__ == '__main__':
    log.setLevel("INFO")
    unittest.main()
