#!/usr/bin/python3.7
import re
import glog as log
from instructions_data import *
from typing import List
from utils import findAddrInOperators, matchConstant


class Instruction(object):
    """Abstract assembly instruction, used as default for unknown ones"""

    # Type of instruction, mapped to feature vector index.
    operandTypes = {'trans': 0, 'call': 1, 'math': 2, 'cmp': 3,
                    'crypto': 4, 'mov': 5, 'term': 6, 'def': 7,
                    'other': 8}
    # Type of const values in operator, mapped to feature vector index.
    operatorTypes = {'num_const': 9, 'str_cont': 10}

    def __init__(self, addr: str,
                 operand: str = '',
                 operators: List[str] = []) -> None:
        super(Instruction, self).__init__()
        if isinstance(addr, str):
            self.address = int(addr, 16)
        else:
            self.address = addr

        self.operand: str = operand
        self.operators: List[str] = operators

        self.size: int = 0
        self.start: bool = False
        self.branchTo: int = None
        self.fallThrough: bool = True
        self.call: bool = False
        self.ret: bool = False

    def accept(self, builder):
        """Tell builder how to visit the instruction"""
        builder.visitDefault(self)

    def findAddrInInst(self) -> int:
        """Jumping instructions should override to provide jump address"""
        return None

    def getOperandFeatures(self) -> List[int]:
        return [0] * len(Instruction.operandTypes)

    def getOperatorFeatures(self) -> List[int]:
        features = [0] * len(Instruction.operatorTypes)
        for operator in self.operators:
            numeric_cnts, string_cnts = matchConstant(operator)
            features[0] += numeric_cnts
            features[1] += string_cnts

        return features

    def __repr__(self) -> str:
        operators = " ".join(self.operators)
        return "%X: %s %s" % (self.address, self.operand, operators)


class DataInst(Instruction):
    """Variable/data declaration statements don't have operand"""

    def __init__(self, addr: str, operators: List[str]) -> None:
        super(DataInst, self).__init__(addr, operators=operators)

    def accept(self, builder):
        builder.visitDefault(self)

    def getOperandFeatures(self) -> List[int]:
        features = [0] * len(Instruction.operandTypes)
        features[Instruction.operandTypes['def']] += 1
        return features


class RegularInst(Instruction):
    """Regular instruction"""

    def __init__(self, addr: str,
                 operand: str,
                 operators: List[str]) -> None:
        super(RegularInst, self).__init__(addr, operand, operators)

    def accept(self, builder):
        builder.visitDefault(self)


class CallingInst(Instruction):
    """Calling"""

    def __init__(self, addr: str,
                 operand: str,
                 operators: List[str]) -> None:
        super(CallingInst, self).__init__(addr, operand, operators)

    def accept(self, builder):
        builder.visitCalling(self)

    def findAddrInInst(self):
        return findAddrInOperators(self.operators)

    def getOperandFeatures(self) -> List[int]:
        features = [0] * len(Instruction.operandTypes)
        features[Instruction.operandTypes['call']] += 1
        return features


class ConditionalJumpInst(Instruction):
    """ConditionalJump"""

    def __init__(self, addr: str,
                 operand: str,
                 operators: List[str]) -> None:
        super(ConditionalJumpInst, self).__init__(addr, operand, operators)

    def accept(self, builder):
        builder.visitConditionalJump(self)

    def findAddrInInst(self):
        return findAddrInOperators(self.operators)

    def getOperandFeatures(self) -> List[int]:
        features = [0] * len(Instruction.operandTypes)
        features[Instruction.operandTypes['trans']] += 1
        return features


class UnconditionalJumpInst(Instruction):
    """UnconditionalJump"""

    def __init__(self, addr: str,
                 operand: str,
                 operators: List[str]) -> None:
        super(UnconditionalJumpInst, self).__init__(addr,
                                                    operand,
                                                    operators)

    def accept(self, builder):
        builder.visitUnconditionalJump(self)

    def findAddrInInst(self):
        return findAddrInOperators(self.operators)

    def getOperandFeatures(self) -> List[int]:
        features = [0] * len(Instruction.operandTypes)
        features[Instruction.operandTypes['trans']] += 1
        return features


class RepeatInst(Instruction):
    """Repeat just the instruction: conditional jump to itself"""

    def __init__(self, addr: str,
                 operand: str,
                 operators: List[str]) -> None:
        super(RepeatInst, self).__init__(addr, operand, operators)

    def accept(self, builder):
        builder.visitConditionalJump(self)

    def findAddrInInst(self):
        return self.address

    def getOperandFeatures(self) -> List[int]:
        features = [0] * len(Instruction.operandTypes)
        features[Instruction.operandTypes['trans']] += 1
        nestedInstStr = " ".join(['00000000'] + self.operators)
        print(nestedInstStr)
        nestedInst = InstBuilder().createInst(nestedInstStr)
        nestedFeatures = nestedInst.getOperandFeatures()

        return [x + y for (x, y) in zip(features, nestedFeatures)]


class EndHereInst(Instruction):
    """EndHere"""

    def __init__(self, addr: str,
                 operand: str,
                 operators: List[str]) -> None:
        super(EndHereInst, self).__init__(addr, operand, operators)

    def accept(self, builder):
        builder.visitEndHere(self)

    def getOperandFeatures(self) -> List[int]:
        features = [0] * len(Instruction.operandTypes)
        features[Instruction.operandTypes['term']] += 1
        return features


class InstBuilder(object):
    """Create instructions based on string content"""

    def __init__(self):
        super(InstBuilder, self).__init__()
        self.seenInst: set = set()

    def createInst(self, progLine: str) -> Instruction:
        elems = progLine.rstrip('\n').split(' ')
        address = elems[0]
        if len(elems) > 2 and elems[2] in DataInstDict:
            elems[1], elems[2] = elems[2], elems[1]

        operand = elems[1]
        operators = elems[2:] if len(elems[2:]) > 0 else []
        operators = [op.rstrip(',') for op in operators]

        # Handle data declaration seperately
        instPattern = re.compile('^[a-z]+$')
        if instPattern.match(operand) is None:
            return DataInst(address, [operand] + operators)

        self.seenInst.add(operand)
        if operand in CallingInstDict:
            return CallingInst(address, operand, operators)
        elif operand in ConditionalJumpInstDict:
            return ConditionalJumpInst(address, operand, operators)
        elif operand in UnconditionalJumpInstDict:
            return UnconditionalJumpInst(address, operand, operators)
        elif operand in EndHereInstDict:
            return EndHereInst(address, operand, operators)
        elif operand in RepeatInstDict:
            return RepeatInst(address, operand, operators)
        elif operand in RegularInstDict:
            return RegularInst(address, operand, operators)
        else:
            return Instruction(address)
