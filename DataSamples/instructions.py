#!/usr/bin/python3
import glog as log


class Instruction(object):
    """Assembly instruction."""
    def __init__(self):
        super(Instruction, self).__init__()
        self.address = -1
        self.operand = None
        self.operators = []
        self.start = False
        self.branchTo = -1
        self.fallThrough = True
        self.call = False
        self.ret = False

    def accept(self, builder):
        builder.visit(self)

class ADDInst(Instruction):
    """ADD A, B"""
    def __init__(self, instAsStr):
        super(ADDInst, self).__init__()
        elems = instAsStr.split(' ')
        self.address = elems[0]
        self.operand = elems[1]
        self.operators.append(elems[2])
        self.oeprators.append(elems[3])

    def accept(self, builder):
        builder.visitADD(self)
