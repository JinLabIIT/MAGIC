#!/usr/bin/python3.7
import re
import glog as log
from typing import List

AddrNotFound = -1
FakeCalleeAddr = -2


def findAddrInOperators(operators: List[str]) -> int:
    hexPattern = re.compile(r'$[1-9A-Fa-f][0-9A-Fa-f]*$')
    for item in operators:
        for part in item.split('_'):
            if hexPattern.match(part) is not None:
                log.info(f'"{part}" in {operators} is convertiable to hex int')
                return int(part, 16)
            else:
                log.debug(f'"{part}" is NOT convertiable to hex int')

    return AddrNotFound


def discoverInstDictionary():
    binaryIds = ['test']
    seenInst = set()
    for bId in binaryIds:
        log.info('Processing ' + bId + '.asm')
        cfgBuilder = ControlFlowGraphBuilder(bId)
        cfgBuilder.build()
        log.debug('%d unique insts in %s.asm' % (len(
            cfgBuilder.instBuilder.seenInst), bId))
        seenInst = seenInst.union(cfgBuilder.instBuilder.seenInst)

    exportSeenInst(seenInst)
