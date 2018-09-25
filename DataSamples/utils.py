#!/usr/bin/python3.7
import re
import glog as log
from typing import List


AddrNotFound = -1
FakeCalleeAddr = -2


def representsInt(s: str) -> bool:
    try:
        log.info(f'{s} is convertiable to hex int')
        int(s, 16)
        return True
    except ValueError:
        log.debug(f'{s} is NOT convertiable to hex int')
        return False


def findAddrInOperators(operators: List[str]) -> int:
    log.info(operators)
    for item in operators:
        log.info(item.split('_'))
        for part in item.split('_'):
            if len(part) > 0 and representsInt(part) is True:
                return int(part, 16)

    return AddrNotFound
