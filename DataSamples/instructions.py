#!/usr/bin/python3.7
import glog as log
import cfg_builder
from typing import List


class Instruction(object):
    """Abstract assembly instruction, used as default for unknown ones"""

    def __init__(self, addr: str) -> None:
        super(Instruction, self).__init__()
        self.address: int = int(addr, 16)
        self.size: int = 0
        self.operand: str = None

        self.start: bool = False
        self.branchTo: int = -1
        self.fallThrough: bool = True
        self.call: bool = False
        self.ret: bool = False

    def accept(self, builder: cfg_builder.ControlFlowGraphBuilder):
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

    def accept(self, builder: cfg_builder.ControlFlowGraphBuilder):
        builder.visitAdd(self)


class AlignInst(Instruction):
    """align op1"""

    def __init__(self, addr: str, operators: List[str]) -> None:
        super(AlignInst, self).__init__(addr)
        self.operand = 'align'
        if len(operators) != 1:
            log.debug('Invalid operators for align inst: %s' % operators)

        self.operators: str = operators[0]

    def accept(self, builder: cfg_builder.ControlFlowGraphBuilder):
        builder.visitAlign(self)


class AndInst(Instruction):
    """and op1, op2"""

    def __init__(self, addr: str, operators: List[str]) -> None:
        super(AndInst, self).__init__(addr)
        self.operand = 'and'
        if len(operators) != 2:
            log.debug('Invalid operators for add inst: %s' % operators)

        self.operators: List[str] = operators

    def accept(self, builder: cfg_builder.ControlFlowGraphBuilder):
        builder.visitAnd(self)


class CallInst(Instruction):
    """call op1"""

    def __init__(self, addr: str, operators: List[str]) -> None:
        super(CallInst, self).__init__(addr)
        self.operand = 'call'
        if len(operators) != 1:
            log.debug('Invalid operators for call inst: %s' % operators)

        self.operators: str = operators[0]

    def accept(self, builder: cfg_builder.ControlFlowGraphBuilder):
        builder.visitCall(self)


class CdqInst(Instruction):
    """cdq: convert doubleword to quadword"""

    def __init__(self, addr: str, operators: List[str]) -> None:
        super(CdqInst, self).__init__(addr)
        self.operand = 'cdq'
        if len(operators) != 0:
            log.debug('Invalid operators for cdq inst: %s' % operators)

    def accept(self, builder: cfg_builder.ControlFlowGraphBuilder):
        builder.visitCdq(self)


class CmpInst(Instruction):
    """cmp op1, op2"""

    def __init__(self, addr: str, operators: List[str]) -> None:
        super(CmpInst, self).__init__(addr)
        self.operand = 'cmp'
        if len(operators) != 2:
            log.debug('Invalid operators for cmp inst: %s' % operators)

        self.operators = operators

    def accept(self, builder: cfg_builder.ControlFlowGraphBuilder):
        builder.visitCmp(self)


class DbInst(Instruction):
    """db ..."""

    def __init__(self, addr: str, operators: List[str]) -> None:
        super(DbInst, self).__init__(addr)
        self.operand = 'db'
        self.operators = operators

    def accept(self, builder: cfg_builder.ControlFlowGraphBuilder):
        builder.visitDb(self)


class DdInst(Instruction):
    """dd ..."""

    def __init__(self, addr: str, operators: List[str]) -> None:
        super(DdInst, self).__init__(addr)
        self.operand = 'dd'
        self.operators = operators

    def accept(self, builder: cfg_builder.ControlFlowGraphBuilder):
        builder.visitDd(self)


class DecInst(Instruction):
    """dec op1"""

    def __init__(self, addr: str, operators: List[str]) -> None:
        super(DecInst, self).__init__(addr)
        self.operand = 'dec'
        if len(operators) != 1:
            log.debug('Invalid operators for dec inst: %s' % operators)

        self.operators = operators

    def accept(self, builder: cfg_builder.ControlFlowGraphBuilder):
        builder.visitDec(self)


class DivInst(Instruction):
    """div  op1"""

    def __init__(self, addr: str, operators: List[str]) -> None:
        super(DivInst, self).__init__(addr)
        self.operand = 'div'
        if len(operators) != 1:
            log.debug('Invalid operators for div    inst: %s' % operators)

        self.operators = operators

    def accept(self, builder: cfg_builder.ControlFlowGraphBuilder):
        builder.visitDiv(self)


class DwInst(Instruction):
    """dw op1"""

    def __init__(self, addr: str, operators: List[str]) -> None:
        super(DwInst(addr, operators), self).__init__(addr)
        self.operand = 'dw'
        self.operators = operators

    def accept(self, builder: cfg_builder.ControlFlowGraphBuilder):
        builder.visit(self)


class FdivrpInst(Instruction):
    """fdivrp op1"""

    def __init__(self, addr: str, operators: List[str]) -> None:
        super(FdivrpInst, self).__init__(addr)
        self.operand = 'fdivrp'
        if len(operators) != 1:
            log.debug('Invalid operators for fdivrp inst: %s' % operators)

        self.operators = operators

    def accept(self, builder: cfg_builder.ControlFlowGraphBuilder):
        builder.visitFdivrp(self)


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
        else:
            return Instruction(address)
