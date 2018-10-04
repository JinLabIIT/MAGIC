import glog as log
import unittest
from cfg_builder import ControlFlowGraphBuilder
from process_graphs import DataProvider
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
        cfgBuilder.printCfg()
        expectedBlocks = ['-2', '-1', 'ff',
                          '401048', '401050', '401052', '401054',
                          '401064', '40106a', '40106c',
                          '40106d', '401076', '401079', '40107e',
                          '401080', '401084', '401090', '401092',
                          '40109a', '4010a3', '4010a9', '4010ac',
                          '4010ae', '4010b3', '4010b5', '4010b7',
                          '4010b9',
                          ]
        expectedBlocks = ["%08X" % int(x, 16) for x in expectedBlocks]
        expectedEdges = [('401048', '401050'), ('401048', '401048'),
                         ('401050', '401054'),
                         ('401054', '401064'),
                         ('401064', '40106a'), ('401064', '4010ae'),
                         ('40106a', '401084'), ('40106a', '40106c'),
                         ('40106c', '40106d'), ('40106c', '401064'),
                         ('40106d', '401076'), ('40106d', '401054'),
                         ('401076', '401079'),
                         ('401079', '40107e'), ('401079', '-2'),
                         ('40107e', '401080'), ('40107e', '401079'),
                         ('401084', '401090'), ('401084', '4010a3'),
                         ('401090', '401092'), ('401090', '-2'),
                         ('401092', '40109a'), ('401092', '-2'),
                         ('40109a', '4010a9'),
                         ('4010a3', '4010a9'),
                         ('4010ac', '4010ae'),
                         ('4010ae', '-2'),
                         ('4010b3', '4010b5'), ('4010b3', '401084'),
                         ('4010b5', '4010b7'), ('4010b5', '-1'),
                         ('4010b7', '4010b9'), ('4010b7', 'ff'),
                         ]
        expectedEdges = [("%08X" % int(x, 16), "%08X" % int(y, 16))
                         for (x, y) in expectedEdges]
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

    # @unittest.skip("Uncomment to run")
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

    # @unittest.skip("Uncomment to run")
    def test_evalHexExpr(self):
        expressions = ['14769F + 48D - 48Dh - 14769Fh+ 14769F',
                       '4477DAB5F7',
                       '435C89+4',
                       '47h -4444DFFF + 4444DFFFh']
        expectedRet = [0x14769F, 0x4477DAB5F7, 0x435C8D, 0x47]
        for expr, expected in zip(expressions, expectedRet):
            self.assertEqual(evalHexAddSubExpr(expr), expected)


if __name__ == '__main__':
    log.setLevel("INFO")
    unittest.main()
