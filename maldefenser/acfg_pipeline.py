#!/usr/bin/python3.7
import pandas as pd
import numpy as np
import scipy as sp
import glog as log
import glob
import os
import time
import cfg_builder
import threading
from typing import List, Dict
from utils import delCodeSegLog, list2Str


class AcfgWorker(threading.Thread):
    """Handle/convert a batch of binary to ACFG"""

    def __init__(self, pathPrefix: str,
                 binaryIds: List[str]) -> None:
        super(AcfgWorker, self).__init__()
        self.seenInst: set = set()
        self.pathPrefix: str = pathPrefix
        self.binaryIds: List[str] = binaryIds

        self.featureMatrices: Dict[str, np.array] = {}
        self.adjMatrices: Dict[str, np.array] = {}

    def exportSeenInst(self, exportTo: str) -> None:
        instColumn = {'Inst': sorted(list(self.seenInst))}
        df = pd.DataFrame(data=instColumn)
        df.to_csv('%s.csv' % exportTo)

    def discoverInstDictionary(self, exportTo: str) -> None:
        idCnt = len(self.binaryIds)
        for (i, bId) in enumerate(self.binaryIds):
            log.debug(f'[DiscoverInstDict] Processing {i}/{idCnt} {bId}.asm')
            cfgBuilder = cfg_builder.ControlFlowGraphBuilder(bId,
                                                             self.pathPrefix)
            cfgBuilder.parseInstructions()
            instCnt = len(cfgBuilder.instBuilder.seenInst)
            log.debug(f'[DiscoverInstDict] Found {instCnt} unique insts')
            self.seenInst = self.seenInst.union(cfgBuilder.instBuilder.seenInst)

        self.exportSeenInst(exportTo)

    def run(self) -> None:
        idCnt = len(self.binaryIds)
        for (i, bId) in enumerate(self.binaryIds):
            log.info(f'[{self.name}] Working on {i + 1}th/{idCnt} binary {bId}')
            acfgBuilder = cfg_builder.AcfgBuilder(bId, self.pathPrefix)
            features, adjMatrix = acfgBuilder.getAttributedCfg()
            self.featureMatrices[bId] = features
            self.adjMatrices[bId] = adjMatrix

        log.info(f'[{self.name}] Generated {idCnt} ACFGs')

    def saveAcfg(self, bId: str) -> None:
        filePrefix = self.pathPrefix + '/' + bId
        np.savetxt(filePrefix + '.features.txt', features, fmt="%d")
        sp.sparse.save_npz(filePrefix + '.adjacent', adjMatrix)


class AcfgMaster(object):

    def __init__(self,
                 pathPrefix: str,
                 labelPath: str,
                 outputTxtName: str = 'ACFG',
                 binaryIds: List[str] = None) -> None:
        super(AcfgMaster, self).__init__()
        self.pathPrefix = pathPrefix
        self.labelPath = labelPath
        self.outputTxtName = outputTxtName
        delCodeSegLog()
        self.bId2Label: Dict[str, str] = self.loadLabel()
        if binaryIds is None:
            self.binaryIds: List[str] = self.loadDefaultBinaryIds()
            # self.binaryIds = self.binaryIds[:10]
        else:
            self.binaryIds: List[str] = binaryIds

        self.bId2Worker: Dict[str, AcfgWorker] = {}

    def loadLabel(self) -> Dict[str, str]:
        df = pd.read_csv(self.labelPath, header=0,
                         dtype={'Id': str, 'Class': str})
        bId2Label = {k.lstrip('"').rstrip('"'): v
                     for (k, v) in zip(df['Id'], df['Class'])}
        return bId2Label

    def loadDefaultBinaryIds(self) -> List[str]:
        """
        Instead of just return bId2Label.keys(), check if binary file
        do exist under pathPrefix directory
        """
        binaryIds = []
        for path in glob.glob(self.pathPrefix + '/*.asm', recursive=False):
            filename = path.split('/')[-1]
            id = filename.split('.')[0]
            binaryIds.append(id)
            assert id in self.bId2Label

        return binaryIds

    def dispatchWorkers(self, numWorkers: int) -> None:
        bIdPerWorker = len(self.binaryIds) // numWorkers
        workers: List[AcfgWorker] = []
        for i in range(0, len(self.binaryIds), bIdPerWorker):
            endIdx = min(i + bIdPerWorker, len(self.binaryIds))
            binaryIdBatch = self.binaryIds[i: endIdx]
            worker = AcfgWorker(self.pathPrefix, binaryIdBatch)
            workers.append(worker)
            for bId in binaryIdBatch:
                self.bId2Worker[bId] = worker

            worker.start()

        for worker in workers:
            worker.join()

        self.aggregateDgcnnFormat()

    def neighborsFromAdjacentMatrix(self, spAdjacentMat) -> Dict[int, List[int]]:
        spAdjacent = sp.sparse.find(spAdjacentMat)
        indices = {}
        for i in range(len(spAdjacent[0])):
            if spAdjacent[0][i] not in indices:
                indices[spAdjacent[0][i]] = []

            indices[spAdjacent[0][i]].append(spAdjacent[1][i])

        return indices

    def aggregateDgcnnFormat(self) -> None:
        log.debug(f"[AggrDgcnnFormat] Aggregate ACFGs to txt format")
        numBinaries = len(self.binaryIds)
        for bId in self.binaryIds:
            if self.bId2Worker[bId].featureMatrices[bId] is None:
                numBinaries -= 1
            elif self.bId2Worker[bId].adjMatrices[bId] is None:
                numBinaries -= 1

        output = open(self.pathPrefix + '/' + self.outputTxtName + '.txt', 'w')
        output.write("%d\n" % numBinaries)
        for (b, bId) in enumerate(self.binaryIds):
            log.debug(f"[AggrDgcnnFormat] Processing {b + 1}th/{numBinaries} ACFG")
            label = self.bId2Label[bId]
            features = self.bId2Worker[bId].featureMatrices[bId]
            spAdjacentMat = self.bId2Worker[bId].adjMatrices[bId]
            if features is None or spAdjacentMat is None:
                log.warning(f'[AggrDgcnnFormat:{bId}] Empty CFG and features')
                continue

            output.write("%d %s\n" % (features.shape[0], label))
            indices = self.neighborsFromAdjacentMatrix(spAdjacentMat)
            for (i, feature) in enumerate(features):
                neighbors = indices[i] if i in indices else []
                output.write("1 %d %s\n" %
                             (len(neighbors), list2Str(neighbors, feature)))

        output.close()
        log.info(f"[AggrDgcnnFormat] {numBinaries}/{len(self.binaryIds)} converted")

    def loadAcfgMatrices(self, bId):
        filePrefix = self.pathPrefix + '/' + bId
        features = np.loadtxt(filePrefix + '.features.txt',
                              dtype=int, ndmin=2)
        sp_adjacent_mat = sp.sparse.load_npz(filePrefix + '.adjacent.npz')
        return features, sp_adjacent_mat

    def clearTmpFiles(self) -> None:
        log.debug(f"[ClearTmpFiles] Remove temporary files ****")
        for (i, bId) in enumerate(self.binaryIds):
            filePrefix = self.pathPrefix + '/' + bId
            for ext in ['.label.txt', '.features.txt', '.adjacent.npz']:
                os.remove(filePrefix + ext)

        log.debug(f"[ClearTmpFiles] {len(self.binaryIds)} files removed ****")


if __name__ == '__main__':
    log.setLevel("INFO")
    pathPrefix = '../TrainSet'
    labelPath = '../trainLabels.csv'
    master = AcfgMaster(pathPrefix, labelPath)

    start = time.process_time()
    master.dispatchWorkers(1)
    runtime = time.process_time() - start
    log.info(f'Running time of 1-thread: {runtime} seconds')
