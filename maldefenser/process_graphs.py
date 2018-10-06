#!/usr/bin/python3.7
import pandas as pd
import glog as log
import glob
import cfg_builder
from typing import List
from utils import delCodeSegLog


class DataProvider(object):
    """Handle data location stuff"""

    def __init__(self, pathPrefix: str) -> None:
        super(DataProvider, self).__init__()
        self.seenInst: set = set()
        self.pathPrefix: str = pathPrefix
        self.binaryIds: List[str] = []

    def getBinaryIds(self) -> List[str]:
        self.binaryIds = []
        for path in glob.glob(self.pathPrefix + '/*.asm', recursive=False):
            filename = path.split('/')[-1]
            id = filename.split('.')[0]
            self.binaryIds.append(id)

        return self.binaryIds

    def exportSeenInst(self, exportTo: str) -> None:
        instColumn = {'Inst': sorted(list(self.seenInst))}
        df = pd.DataFrame(data=instColumn)
        df.to_csv('%s.csv' % exportTo)

    def discoverInstDictionary(self, binaryIds: List[str], exportTo: str):
        for (i, bId) in enumerate(binaryIds):
            log.info(f'Processing {i}/{len(binaryIds)} {bId}.asm')
            cfgBuilder = cfg_builder.ControlFlowGraphBuilder(bId, self.pathPrefix)
            cfgBuilder.parseInstructions()
            log.debug(f'{len(cfgBuilder.instBuilder.seenInst)} unique insts in {bId}.asm')
            self.seenInst = self.seenInst.union(cfgBuilder.instBuilder.seenInst)

        self.exportSeenInst(exportTo)


if __name__ == '__main__':
    log.setLevel("INFO")
    pathPrefix = '../TrainSet'
    delCodeSegLog()
    dataProvider = DataProvider(pathPrefix)
    binaryIds = dataProvider.getBinaryIds()
    log.info(f'{binaryIds}')
    dataProvider.discoverInstDictionary(binaryIds, 'seen_inst')
