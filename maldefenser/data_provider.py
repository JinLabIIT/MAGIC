#!/usr/bin/python3.7
import pandas as pd
import numpy as np
import scipy as sp
import glog as log
import glob
import cfg_builder
from typing import List, Dict
from utils import delCodeSegLog


class DataProvider(object):
    """Handle data location/storage stuff"""

    def __init__(self, pathPrefix: str, labelPath: str = '') -> None:
        super(DataProvider, self).__init__()
        self.seenInst: set = set()
        self.pathPrefix: str = pathPrefix
        self.labelPath: str = labelPath
        self.binaryIds: List[str] = []
        delCodeSegLog()

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

    def discoverInstDictionary(self, binaryIds: List[str], exportTo: str) -> None:
        for (i, bId) in enumerate(binaryIds):
            log.info(f'Processing {i}/{len(binaryIds)} {bId}.asm')
            cfgBuilder = cfg_builder.ControlFlowGraphBuilder(bId, self.pathPrefix)
            cfgBuilder.parseInstructions()
            log.debug(f'{len(cfgBuilder.instBuilder.seenInst)} unique insts in {bId}.asm')
            self.seenInst = self.seenInst.union(cfgBuilder.instBuilder.seenInst)

        self.exportSeenInst(exportTo)

    def loadLabel(self) -> Dict[str, str]:
        df = pd.read_csv(self.labelPath, header=0,
                         dtype={'Id': str, 'Class': str})
        id2Label = {k.lstrip('"').rstrip('"'): v
                    for (k, v) in zip(df['Id'], df['Class'])}
        return id2Label

    def storeMatrices(self, binaryIds: List[str]) -> None:
        id2Label = self.loadLabel()
        for (i, bId) in enumerate(binaryIds):
            if bId not in id2Label:
                log.error(f'Unable to label program {bId}')
                continue

            acfgBuilder = cfg_builder.AcfgBuilder(bId, self.pathPrefix)
            features, adjMatrix = acfgBuilder.getAttributedCfg()
            filePrefix = self.pathPrefix + '/' + bId
            np.savetxt(filePrefix + '.features.txt', features, fmt="%d")
            np.savetxt(filePrefix + '.label.txt',
                       np.array([id2Label[bId]]), fmt="%s")
            sp.sparse.save_npz(filePrefix + '.adjacent', adjMatrix)


if __name__ == '__main__':
    log.setLevel("INFO")
    pathPrefix = '../TrainSet'
    dataProvider = DataProvider(pathPrefix, '../trainLabels.csv')
    dataProvider.storeMatrices([])
