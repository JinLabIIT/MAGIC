import glog as log
import unittest
from cfg_builder import ControlFlowGraphBuilder


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
        ]
        for bId in binaryIds:
            log.info('Processing ' + bId + '.asm')
            cfgBuilder = ControlFlowGraphBuilder(bId, pathPrefix)
            cfgBuilder.parseInstructions()

    def test_build(self):
        pathPrefix = '../DataSamples'
        binaryIds = [
            'test',
        ]
        for bId in binaryIds:
            log.info('Processing ' + bId + '.asm')
            cfgBuilder = ControlFlowGraphBuilder(bId, pathPrefix)
            cfgBuilder.buildControlFlowGraph()


if __name__ == '__main__':
    log.setLevel("INFO")
    unittest.main()
