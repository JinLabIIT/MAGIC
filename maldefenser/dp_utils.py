#!/usr/bin/python3.7
import re
import glob
import glog as log
from typing import List, Dict

FakeCalleeAddr = -2
InvalidAddr = -1
CodeSegLogName = 'EmptyCodeSeg.err'


def evalHexAddSubExpr(expr: str) -> int:
    val, op, curr = None, None, 0
    for (i, c) in enumerate(expr):
        if c in ['+', '-']:
            if val is None:
                val = curr
            else:
                val = (val + curr) if op is '+' else (val - curr)

            curr = 0
            op = c
        elif c is not 'h' and c is not ' ':
            curr = curr * 16 + int(c, 16)

    if op is None:
        return curr
    else:
        return (val + curr) if op is '+' else (val - curr)


def baseAddrInExpr(expr: str) -> int:
    curr = 0
    for (i, c) in enumerate(expr):
        if c in ['+', '-', '*', '/', 'h', ' ']:
            break

        curr = curr * 16 + int(c, 16)

    return curr


def findAddrInOperators(operators: List[str]) -> int:
    """
    Find possible address in operators.
    For call/syscall inst, this func may return false positive address;
    the remedy is to do a second check on if addr is invalid,
    in which case the returned addr should be treated as FakeCalleeAddr.
    """
    hexPattern = re.compile(r'[0-9A-Fa-f]+h?([\+\-\*\/][0-9A-Fa-f]+)?$')
    for item in operators:
        for part in item.split('_'):
            if hexPattern.match(part) is not None:
                log.debug(f'[FindAddr] Convert "{part}" in {operators} to hex')
                return baseAddrInExpr(part)

    log.debug(f'[FindAddr] "{operators}" NOT convertiable to hex')
    return FakeCalleeAddr


def delCodeSegLog() -> None:
    with open(CodeSegLogName, 'w') as errFile:
        errFile.write('binaryId\n')


def addCodeSegLog(binaryId) -> None:
    with open(CodeSegLogName, 'a') as errFile:
        errFile.write('%s\n' % binaryId)


def list2Str(l1, l2):
    """
    Merge two list, then return space seperated string format.
    """
    return " ".join([str(x) for x in (list(l1) + list(l2))])


def matchConstant(line: str) -> List[int]:
    """Parse the numeric/string constants in an operand"""
    operand = line.strip('\n\r\t ')
    numericCnts = 0
    stringCnts = 0
    """
    Whole operand is a num OR leading num in expression.
    E.g. "0ABh", "589h", "0ABh" in "0ABh*589h"
    """
    wholeNum = r'^([1-9][0-9A-F]*|0[A-F][0-9A-F]*)h?.*'
    pattern = re.compile(wholeNum)
    if pattern.match(operand):
        numericCnts += 1
        log.debug(f'[MatchConst] Match whole number in {operand}')
        # numerics.append('%s:WHOLE/LEAD' % operand)
    """Number inside expression, exclude the leading one."""
    numInExpr = r'([+*/:]|-)([1-9][0-9A-F]*|0[A-F][0-9A-F]*)h?'
    pattern = re.compile(numInExpr)
    match = pattern.findall(operand)
    if len(match) > 0:
        numericCnts += 1
        log.debug(f'[MatchConst] Match in-expression number in {operand}')
        # numerics.append('%s:%d' % (operand, len(match)))
    """Const string inside double/single quote"""
    strRe = r'["\'][^"]+["\']'
    pattern = re.compile(strRe)
    match = pattern.findall(operand)
    if len(match) > 0:
        stringCnts += 1
        log.debug(f'[MatchConst] Match str const in {operand}')
        # strings.append('%s:%d' % (operand, len(match)))

    return [numericCnts, stringCnts]


def loadBinaryIds(pathPrefix: str,
                  bId2Label: Dict[str, str] = None) -> List[str]:
    """
    Instead of just return @bId2Label.keys(), check if binary file
    do exist under @pathPrefix directory
    """
    binaryIds = []
    for path in glob.glob(pathPrefix + '/*.asm', recursive=False):
        filename = path.split('/')[-1]
        id = filename.split('.')[0]
        binaryIds.append(id)
        if bId2Label is not None:
            assert id in bId2Label

    return binaryIds
