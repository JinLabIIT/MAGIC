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
