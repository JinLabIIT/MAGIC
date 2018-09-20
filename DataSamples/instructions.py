#!/usr/bin/python3.7
from typing import List


class Instruction(object):
    """Assembly instruction."""

    def __init__(self) -> None:
        super(Instruction, self).__init__()
        self.address: int = -1
        self.size: int = 0
        self.operand: str = None
        self.operators: List[str] = []

        self.start: bool = False
        self.branchTo: int = -1
        self.fallThrough: bool = True
        self.call: bool = False
        self.ret: bool = False

    def accept(self, builder):
        builder.visit(self)


class AddInst(Instruction):
    """ADD A, B"""

    def __init__(self, addr: str, operators: List[str]) -> None:
        super(AddInst, self).__init__()
        self.address = int(addr, 16)
        self.operand = 'add'
        self.operators.extend(operators)

    def accept(self, builder):
        builder.visitADD(self)


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
        self.seenInst.add(operand)
        if operand == 'add':
            op1, op2 = elems[-2], elems[-1]
            return AddInst(address, [op1, op2])

        return Instruction()
