import glob
import pandas as pd
import glog as log
from typing import Dict, List
from copy import deepcopy


class HyperParameterIterator(object):
    def __init__(self, hpPath) -> None:
        self.hp = self._loadHyperParameters(hpPath)
        log.debug(f'Hyperparameter atomic values: {self.hp}')
        self.combinations = self.recursive(deepcopy(self.hp))
        log.info(f'#Combinations = {len(self.combinations)}')
        log.debug(f'#limit = {self.getLimit()}')
        assert len(self.combinations) == self.getLimit()
        self.curr = 0

    def _loadHyperParameters(self, hpPath: str) -> Dict[str, List[float]]:
        result = {}
        with open(hpPath, 'r') as f:
            for line in f:
                if line.startswith('#') or line == "\n":
                    continue

                item1, item2 = line.split('=')
                name = item1.lstrip(' ').rstrip(' ')
                value = item2.lstrip(' ').rstrip(' ')
                result[name] = eval(value)

        return result

    def getLimit(self) -> int:
        limit = 1
        for val in self.hp.values():
            limit *= len(val)

        return limit

    def recursive(self, hpSet: Dict[str, List[int]]) -> List[Dict[str, int]]:
        combinations = []
        if len(hpSet) == 0:
            return combinations

        key = list(hpSet.keys())[0]
        values = deepcopy(hpSet[key])
        del hpSet[key]

        subCombinations = self.recursive(hpSet)
        if len(subCombinations) == 0:
            for x in values:
                combinations.append({key: x})
        else:
            for x in values:
                for subComb in subCombinations:
                    mergeComb = deepcopy(subComb)
                    log.debug(f'{mergeComb} + {key}: {x}')
                    mergeComb[key] = x
                    combinations.append(mergeComb)

        return combinations

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, int]:
        if self.curr == len(self.combinations):
            raise StopIteration
        else:
            result = self.combinations[self.curr]
            self.curr += 1
            return result


def hpWithMinLoss(filename: str):
    with open(filename, 'r') as f:
        comment = f.readline()
        hp = eval(comment[2:])

    df = pd.read_csv(open(filename, 'r'), comment='#', header='infer')
    log.debug(f'{filename}: {list(df.columns.values)}')
    validLoss = df['AvgValidLoss']
    optRow = df.loc[validLoss.idxmin(axis=0)]
    hp['optNumEpochs'] = optRow['Epoch'] + 1
    hp['optLoss'] = optRow['AvgValidLoss']
    f.close()

    return hp


def parseHpTuning(prefix: str, gpuIdList: List[int] = [1]):
    optHp = {'optLoss': 100000000.0}
    for gpuId in gpuIdList:
        path = prefix + 'Gpu%sRun*.csv' % gpuId
        for filename in glob.glob(path, recursive=False):
            hp = hpWithMinLoss(filename)
            if hp['optLoss'] < optHp['optLoss']:
                optHp = deepcopy(hp)

    return optHp
