#!/usr/bin/python3.7
import re
import glog as log
from typing import List
from utils import findAddrInOperators, FakeCalleeAddr, AddrNotFound


class Instruction(object):
    """Abstract assembly instruction, used as default for unknown ones"""

    def __init__(self, addr: str) -> None:
        super(Instruction, self).__init__()
        self.address: int = int(addr, 16)
        self.size: int = 0
        self.operand: str = None

        self.start: bool = False
        self.branchTo: int = AddrNotFound
        self.fallThrough: bool = True
        self.call: bool = False
        self.ret: bool = False

    def accept(self, builder):
        builder.visitDefault(self)

    def findAddrInInst(self) -> int:
        return None


class DataInst(Instruction):
    """Variable/data declaration statements don't have operand"""

    def __init__(self, addr: str, operators: List[str]) -> None:
        super(DataInst, self).__init__(addr)
        self.operators: List[str] = operators

    def accept(self, builder):
        builder.visitDefault(self)

class RegularInst(Instruction):
    """Regular instruction"""

    def __init__(self, addr: str, operand: str, operators: List[str]) -> None:
        super(RegularInst, self).__init__(addr)
        self.operand = operand
        self.operators: List[str] = operators

    def accept(self, builder):
        builder.visitDefault(self)


class CallingInst(Instruction):
    """Calling"""

    def __init__(self, addr: str, operand: str, operators: List[str]) -> None:
        super(CallingInst, self).__init__(addr)
        self.operand = operand
        self.operators: List[str] = operators

    def accept(self, builder):
        builder.visitCalling(self)

    def findAddrInInst(self):
        addr = findAddrInOperators(self.operators)
        if addr < 0:
            return FakeCalleeAddr
        else:
            return addr


class ConditionalJumpInst(Instruction):
    """ConditionalJump"""

    def __init__(self, addr: str, operand: str, operators: List[str]) -> None:
        super(ConditionalJumpInst, self).__init__(addr)
        self.operand = operand
        self.operators: List[str] = operators

    def accept(self, builder):
        builder.visitConditionalJump(self)

    def findAddrInInst(self):
        return findAddrInOperators(self.operators)


class UnconditionalJumpInst(Instruction):
    """UnconditionalJump"""

    def __init__(self, addr: str, operand: str, operators: List[str]) -> None:
        super(UnconditionalJumpInst, self).__init__(addr)
        self.operand = operand
        self.operators: List[str] = operators

    def accept(self, builder):
        builder.visitUnconditionalJump(self)

    def findAddrInInst(self):
        return findAddrInOperators(self.operators)


class EndHereInst(Instruction):
    """EndHere"""

    def __init__(self, addr: str, operand: str, operators: List[str]) -> None:
        super(EndHereInst, self).__init__(addr)
        self.operand = operand
        self.operators: List[str] = operators

    def accept(self, builder):
        builder.visitEndHere(self)


class InstBuilder(object):
    """Create instructions based on string content"""

    def __init__(self):
        super(InstBuilder, self).__init__()
        self.seenInst: set = set()

    def createInst(self, progLine: str) -> Instruction:
        DataInstList = ['dd', 'db', 'dw', 'dq']
        CallingInstList = ['call']
        ConditionalJumpInstList = ['jnz', 'jz']
        UnconditionalJumpInstList = ['jmp']
        EndHereInstList = ['reti', 'retn']
        RegularInstList = ['add', 'align', 'and', 'cdq', 'dec',
                           'div', 'fdivrp', 'lea', 'mov', 'push',
                           'pop', 'sub', 'xor', ]

        elems = progLine.rstrip('\n').split(' ')
        address = elems[0]
        if len(elems) > 2 and elems[2] in DataInstList:
            elems[1], elems[2] = elems[2], elems[1]

        operand = elems[1]
        operators = elems[2:] if len(elems[2:]) > 0 else []
        operators = [op.rstrip(',') for op in operators]

        # Handle data declaration seperately
        instPattern = re.compile('^[a-z]+$')
        if instPattern.match(operand) is None:
            return DataInst(address, [operand] + operators)

        self.seenInst.add(operand)
        if operand in CallingInstList:
            return CallingInst(address, operand, operators)
        elif operand in ConditionalJumpInstList:
            return ConditionalJumpInst(address, operand, operators)
        elif operand in UnconditionalJumpInstList:
            return UnconditionalJumpInst(address, operand, operators)
        elif operand in EndHereInstList:
            return EndHereInst(address, operand, operators)
        elif operand in RegularInstList:
            return RegularInst(address, operand, operators)
        else:
            return Instruction(address)
