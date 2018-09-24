#!/usr/bin/python3.7
import glog as log
from typing import List


AddrNotFound = -1
FakeCalleeAddr = -2


def representsInt(s: str) -> bool:
    try:
        log.debug(f'{s} is convertiable to hex int')
        int(s, 16)
        return True
    except ValueError:
        return False


def findAddrInOperators(operators: List[str]) -> int:
    for item in operators:
        for part in item.split('_'):
            if representsInt(part) is True:
                return int(part, 16)

    return AddrNotFound
