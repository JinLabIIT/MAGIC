#!/usr/bin/python3.7
import re
import glog as log
from typing import List

AddrNotFound = -1
FakeCalleeAddr = -2
CodeSegLogName = 'EmptyCodeSeg.err'


def findAddrInOperators(operators: List[str]) -> int:
    hexPattern = re.compile(r'^[0-9A-Fa-f]+$')
    for item in operators:
        for part in item.split('_'):
            if hexPattern.match(part) is not None:
                log.debug(f'"{part}" in {operators} is convertiable to hex int')
                return int(part, 16)
            else:
                log.debug(f'"{part}" is NOT convertiable to hex int')

    return AddrNotFound


def delCodeSegLog() -> None:
    with open(CodeSegLogName, 'w') as errFile:
        errFile.write('binaryId\n')


def addCodeSegLog(binaryId) -> None:
    with open(CodeSegLogName, 'a') as errFile:
        errFile.write('%s\n' % binaryId)

