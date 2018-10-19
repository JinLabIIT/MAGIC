import glog as log
from typing import Dict, List
from copy import deepcopy


batchSize = range(40, 81, 40)
cvFold = [5]
numEpochs = [2]
sortPoolingRatio = [0.6]
s2vOutDim = range(100, 201, 200)
regHidden = range(100, 201, 200)
msgPassLv = range(2, 5, 4)
lr = [0.00012]
dropOutRate = [0.0, 0.4]
convSize = [[32, 32, 32, 1]]


class HyperParameterIterator(object):
    def __init__(self) -> None:
        self.hp = {
            'batchSize': batchSize,
            'cvFold': cvFold,
            'numEpochs': numEpochs,
            'sortPoolingRatio': sortPoolingRatio,
            's2vOutDim': s2vOutDim,
            'regHidden': regHidden,
            'msgPassLv': msgPassLv,
            'lr': lr,
            'dropOutRate': dropOutRate,
            'convSize': convSize,
        }
        self.combinations = self.recursive(deepcopy(self.hp))
        log.info(f'#Combinations = {len(self.combinations)}')
        log.info(f'#limit = {self.getLimit()}')
        assert len(self.combinations) == self.getLimit()
        self.curr = 0

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
