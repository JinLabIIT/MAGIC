#!/usr/bin/python3.7
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


class AddInst(Instruction):
    """add op1, op2"""

    def __init__(self, addr: str, operators: List[str]) -> None:
        super(AddInst, self).__init__(addr)
        self.operand = 'add'
        if len(operators) != 2:
            log.debug('Invalid operators for add inst: %s' % operators)

        self.operators: List[str] = operators


class AlignInst(Instruction):
    """align op1"""

    def __init__(self, addr: str, operators: List[str]) -> None:
        super(AlignInst, self).__init__(addr)
        self.operand = 'align'
        if len(operators) != 1:
            log.debug('Invalid operators for align inst: %s' % operators)

        self.operators: str = operators


class AndInst(Instruction):
    """and op1, op2"""

    def __init__(self, addr: str, operators: List[str]) -> None:
        super(AndInst, self).__init__(addr)
        self.operand = 'and'
        if len(operators) != 2:
            log.debug('Invalid operators for add inst: %s' % operators)

        self.operators: List[str] = operators


class CallInst(Instruction):
    """call op1"""

    def __init__(self, addr: str, operators: List[str]) -> None:
        super(CallInst, self).__init__(addr)
        self.operand = 'call'
        if len(operators) != 1:
            log.debug('Invalid operators for call inst: %s' % operators)

        self.operators: str = operators

    def accept(self, builder):
        builder.visitCall(self)

    def findAddrInInst(self):
        addr = findAddrInOperators(self.operators)
        if addr < 0:
            return FakeCalleeAddr
        else:
            return addr


class CdqInst(Instruction):
    """cdq: convert doubleword to quadword"""

    def __init__(self, addr: str, operators: List[str]) -> None:
        super(CdqInst, self).__init__(addr)
        self.operand = 'cdq'
        if len(operators) != 0:
            log.debug('Invalid operators for cdq inst: %s' % operators)


class CmpInst(Instruction):
    """cmp op1, op2"""

    def __init__(self, addr: str, operators: List[str]) -> None:
        super(CmpInst, self).__init__(addr)
        self.operand = 'cmp'
        if len(operators) != 2:
            log.debug('Invalid operators for cmp inst: %s' % operators)

        self.operators = operators


class DbInst(Instruction):
    """db ..."""

    def __init__(self, addr: str, operators: List[str]) -> None:
        super(DbInst, self).__init__(addr)
        self.operand = 'db'
        self.operators = operators


class DdInst(Instruction):
    """dd ..."""

    def __init__(self, addr: str, operators: List[str]) -> None:
        super(DdInst, self).__init__(addr)
        self.operand = 'dd'
        self.operators = operators


class DecInst(Instruction):
    """dec op1"""

    def __init__(self, addr: str, operators: List[str]) -> None:
        super(DecInst, self).__init__(addr)
        self.operand = 'dec'
        if len(operators) != 1:
            log.debug('Invalid operators for dec inst: %s' % operators)

        self.operators = operators


class DivInst(Instruction):
    """div  op1"""

    def __init__(self, addr: str, operators: List[str]) -> None:
        super(DivInst, self).__init__(addr)
        self.operand = 'div'
        if len(operators) != 1:
            log.debug('Invalid operators for div    inst: %s' % operators)

        self.operators = operators


class DwInst(Instruction):
    """dw op1"""

    def __init__(self, addr: str, operators: List[str]) -> None:
        super(DwInst(addr, operators), self).__init__(addr)
        self.operand = 'dw'
        self.operators = operators


class FdivrpInst(Instruction):
    """fdivrp op1"""

    def __init__(self, addr: str, operators: List[str]) -> None:
        super(FdivrpInst, self).__init__(addr)
        self.operand = 'fdivrp'
        if len(operators) != 1:
            log.debug('Invalid operators for fdivrp inst: %s' % operators)

        self.operators = operators


class JmpInst(Instruction):
    """jmp [short] addr"""

    def __init__(self, addr: str, operators: List[str]) -> None:
        super(JmpInst, self).__init__(addr)
        self.operand = 'jmp'
        if len(operators) != 1:
            log.debug('Invalid operators for jmp inst: %s' % operators)

        self.operators: List[str] = operators

    def accept(self, builder):
        builder.visitJmp(self)

    def findAddrInInst(self):
        return findAddrInOperators(self.operators)


class JnzInst(Instruction):
    """jnz op1, op2"""

    def __init__(self, addr: str, operators: List[str]) -> None:
        super(JnzInst, self).__init__(addr)
        self.operand = 'jnz'
        if len(operators) != 1:
            log.debug('Invalid operators for jnz inst: %s' % operators)

        self.operators: List[str] = operators

    def accept(self, builder):
        builder.visitJnz(self)

    def findAddrInInst(self):
        return findAddrInOperators(self.operators)


class LeaInst(Instruction):
    """lea op1, op2"""

    def __init__(self, addr: str, operators: List[str]) -> None:
        super(LeaInst, self).__init__(addr)
        self.operand = 'lea'
        if len(operators) != 2:
            log.debug('Invalid operators for lea inst: %s' % operators)

        self.operators: List[str] = operators


class MovInst(Instruction):
    """mov op1, op2"""

    def __init__(self, addr: str, operators: List[str]) -> None:
        super(MovInst, self).__init__(addr)
        self.operand = 'mov'
        if len(operators) != 2:
            log.debug('Invalid operators for mov inst: %s' % operators)

        self.operators: List[str] = operators


class PushInst(Instruction):
    """push op1"""

    def __init__(self, addr: str, operators: List[str]) -> None:
        super(PushInst, self).__init__(addr)
        self.operand = 'push'
        if len(operators) != 1:
            log.debug('Invalid operators for push inst: %s' % operators)

        self.operators: List[str] = operators


class PopInst(Instruction):
    """pop op1"""

    def __init__(self, addr: str, operators: List[str]) -> None:
        super(PopInst, self).__init__(addr)
        self.operand = 'pop'
        if len(operators) != 1:
            log.debug('Invalid operators for pop inst: %s' % operators)

        self.operators: List[str] = operators


class RetiInst(Instruction):
    """reti"""

    def __init__(self, addr: str, operators: List[str]) -> None:
        super(RetiInst, self).__init__(addr)
        self.operand = 'reti'
        if len(operators) != 0:
            log.debug('Invalid operators for reti inst: %s' % operators)

        self.operators: List[str] = operators

    def accept(self, builder):
        builder.visitReti(self)


class RetnInst(Instruction):
    """retn op1, op2"""

    def __init__(self, addr: str, operators: List[str]) -> None:
        super(RetnInst, self).__init__(addr)
        self.operand = 'retn'
        if len(operators) != 1:
            log.debug('Invalid operators for retn inst: %s' % operators)

        self.operators: List[str] = operators

    def accept(self, builder):
        builder.visitRetn(self)


class SubInst(Instruction):
    """sub op1, op2"""

    def __init__(self, addr: str, operators: List[str]) -> None:
        super(SubInst, self).__init__(addr)
        self.operand = 'sub'
        if len(operators) != 2:
            log.debug('Invalid operators for sub inst: %s' % operators)

        self.operators: List[str] = operators


class XorInst(Instruction):
    """xor op1, op2"""

    def __init__(self, addr: str, operators: List[str]) -> None:
        super(XorInst, self).__init__(addr)
        self.operand = 'xor'
        if len(operators) != 2:
            log.debug('Invalid operators for xor inst: %s' % operators)

        self.operators: List[str] = operators


class InstBuilder(object):
    """Create instructions based on string content"""

    def __init__(self):
        super(InstBuilder, self).__init__()
        self.seenInst: set = set()

    def createInst(self, progLine: str) -> Instruction:
        elems = progLine.rstrip('\n').split(' ')
        address = elems[0]
        if len(elems) > 2 and elems[2] in ['dd', 'dw', 'db']:
            elems[1], elems[2] = elems[2], elems[1]

        operand = elems[1]
        operators = elems[2:] if len(elems[2:]) > 0 else []
        operators = [op.rstrip(',') for op in operators]

        self.seenInst.add(operand)
        if operand == 'add':
            return AddInst(address, operators)
        elif operand == 'call':
            return CallInst(address, operators)
        elif operand == 'jmp':
            return JmpInst(address, operators)
        elif operand == 'jnz':
            return JnzInst(address, operators)
        elif operand == 'lea':
            return LeaInst(address, operators)
        elif operand == 'mov':
            return MovInst(address, operators)
        elif operand == 'push':
            return PushInst(address, operators)
        elif operand == 'pop':
            return PopInst(address, operators)
        elif operand == 'reti':
            return RetiInst(address, operators)
        elif operand == 'retn':
            return RetnInst(address, operators)
        elif operand == 'sub':
            return SubInst(address, operators)
        elif operand == 'xor':
            return XorInst(address, operators)
        else:
            return Instruction(address)
