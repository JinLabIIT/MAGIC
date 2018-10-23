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


def f1MetricForHp(filename: str):
    with open(filename, 'r') as f:
        comment = f.readline()
        hp = eval(comment[2:])

    df = pd.read_csv(open(filename, 'r'), comment='#', header='infer')
    log.debug(f'{filename}: {list(df.columns.values)}')
    f1Score = df['AvgValidF1']
    optRow = df.loc[f1Score.idxmax(axis=0)]
    hp['optNumEpochs'] = optRow['Epoch'] + 1
    hp['optAccu'] = optRow['AvgValidAccu']
    hp['optF1'] = optRow['AvgValidF1']
    hp['optPrec'] = optRow['AvgValidPrec']
    hp['optRecl'] = optRow['AvgValidRecl']
    f.close()

    return hp


def parseHpTuning(prefix: str):
    optHp = {'optF1': 0.0}
    for filename in glob.glob(prefix + 'Run*.hist', recursive=False):
        hp = f1MetricForHp(filename)
        if hp['optF1'] > optHp['optF1']:
            optHp = deepcopy(hp)

    return optHp