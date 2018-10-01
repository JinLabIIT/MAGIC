import glog as log
import unittest
from cfg_builder import ControlFlowGraphBuilder
from process_graphs import DataProvider
from utils import delCodeSegLog


class TestCfgBuildedr(unittest.TestCase):
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

    def test_build(self):
        pathPrefix = '../DataSamples'
        binaryIds = [
            'test',
        ]
        for bId in binaryIds:
            log.info('Processing ' + pathPrefix + '/' + bId + '.asm')
            cfgBuilder = ControlFlowGraphBuilder(bId, pathPrefix)
            cfgBuilder.buildControlFlowGraph()

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


if __name__ == '__main__':
    log.setLevel("INFO")
    unittest.main()
