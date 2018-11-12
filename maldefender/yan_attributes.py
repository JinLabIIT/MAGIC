#!/usr/bin/python2.7

import re
import glog as log
import numpy as np
from networkx import number_of_nodes
from python23_common import matchConstant


DataInst = ['dd', 'db', 'dw', 'dq', 'dt', 'extrn', 'unicode']
DataInstDict = {k: v for v, k in enumerate(DataInst)}

TransferInst = [
    'enter', 'leave', 'jcxz', 'ja', 'jb', 'jbe', 'jcxz', 'jecxz', 'jg', 'jge',
    'jl', 'jle', 'jmp', 'jnb', 'jno', 'jnp', 'jns', 'jnz', 'jo', 'jp', 'js',
    'jz', 'loop', 'loope', 'loopne', 'rep', 'repe', 'repne', 'wait'
]
TransferInstDict = {k: v for v, k in enumerate(TransferInst)}

TermInst = [
    'end', 'iret', 'iretw', 'retf', 'reti', 'retfw', 'retn', 'retnw',
    'sysexit', 'sysret', 'xabort'
]
TermInstDict = {k: v for v, k in enumerate(TermInst)}

CallInst = ['VxDCall', 'call', 'int', 'into']
CallInstDict = {k: v for v, k in enumerate(CallInst)}

MovInst = [
    'cmovb', 'cmovno', 'cmovz', 'fcmovb', 'fcmovne', 'fcmovu', 'mov', 'movapd',
    'movaps', 'movd', 'movdqa', 'movdqu', 'movhlps', 'movhps', 'movlhps',
    'movlpd', 'movlps', 'movntdq', 'movntps', 'movntq', 'movq', 'movsb',
    'movsd', 'movsldup', 'movss', 'movsw', 'movsx', 'movups', 'movzx',
    'pmovmskb'
]
MovInstDict = {k: v for v, k in enumerate(MovInst)}

CryptoInst = [
    'aesdec', 'aesdeclast', 'aesenc', 'aesenclast', 'aesimc', 'aeskeygenassist'
]
CryptoInstDict = {k: v for v, k in enumerate(CryptoInst)}

ConversionInst = [
    'punpckldq', 'cbw', 'cdq', 'cvtdq2pd', 'cvtdq2ps', 'cvtpi2ps', 'cvtsd2si',
    'cvtsi2sd', 'cvtsi2ss', 'cvttps2pi', 'cvttsd2si', 'cwd', 'cwde', 'pf2id',
    'pi2fd'
]
MathInst = [
    'aaa', 'aad', 'aam', 'aas', 'adc', 'add', 'addpd', 'addps', 'addsd',
    'addss', 'and', 'andnpd', 'andnps', 'andpd', 'andps', 'bswap', 'btc',
    'btr', 'bts', 'cli', 'cmc', 'daa', 'das', 'dec', 'div', 'divsd', 'divss',
    'emms', 'f2xm1', 'fabs', 'fadd', 'faddp', 'fchs', 'fclex', 'fcos', 'fdiv',
    'fdivp', 'fdivr', 'fdivrp', 'fiadd', 'fidiv', 'fidivr', 'fimul', 'fisub',
    'fisubr', 'fldl2e', 'fldlg2', 'fldln2', 'fldpi', 'fldz', 'fmul', 'fpatan',
    'fprem', 'fprem1', 'fptan', 'frndint', 'fscale', 'fsin', 'fsincos',
    'fsqrt', 'fsub', 'fsubp', 'fsubr', 'fsubrp', 'fxch', 'fyl2x', 'idiv',
    'imul', 'inc', 'in'
    'maxss', 'minss', 'mul', 'mulpd', 'mulps', 'mulsd', 'mulss', 'neg', 'not',
    'or', 'orpd', 'orps', 'paddb', 'paddb', 'paddd', 'paddq', 'paddsb',
    'paddsw', 'paddusb', 'paddusw'
    'paddw', 'pand', 'pandn', 'pfacc', 'pfadd', 'pfmax', 'pfmin', 'pfmul',
    'pfnacc', 'pfpnacc', 'pfrcp', 'pfrcpit1', 'pfrcpit2', 'pfrsqit1',
    'pfrsqrt', 'pfsub', 'pfsubr', 'pmaddwd', 'pmaxsw', 'pminsw', 'pminub',
    'pmulhuw', 'pmulhw', 'pmullw', 'pmuludq', 'por', 'pslld', 'psllq', 'psllw',
    'psrad', 'psraw', 'psrld', 'psrldq', 'psrlq', 'psrlw', 'psubb', 'psubd',
    'psubq', 'psubsb', 'psubsw', 'psubusb', 'psubusw', 'psubw', 'pswapd',
    'punpckhbw', 'punpckhdq', 'punpckhwd', 'punpcklbw', 'punpckldq',
    'punpcklqdq', 'punpcklwd', 'pxor', 'rcl', 'rcpps', 'rcpss', 'rcr', 'rol',
    'ror', 'rsqrtss', 'sal', 'sar', 'sbb', 'setb', 'setbe', 'setl', 'setle',
    'setnb', 'setnbe', 'setnl', 'setnle', 'setns', 'setnz', 'seto', 'sets',
    'setz', 'shl', 'shld', 'shr', 'shrd', 'sqrtsd', 'sqrtss', 'stc', 'std',
    'sti', 'sub', 'subpd', 'subps', 'subsd', 'subss', 'test', 'unpckhpd',
    'unpckhps', 'unpcklpd', 'unpcklps', 'vpmaxsw', 'xadd', 'xchg', 'xlat',
    'xor', 'xorpd', 'xorps'
]
MathInst += ConversionInst
MathInstDict = {k: v for v, k in enumerate(MathInst)}

CmpInst = [
    'cmp', 'cmpnlepd', 'cmpeqsd', 'cmpltpd', 'cmpltsd', 'cmpneqpd', 'cmpneqps',
    'cmpnlepd', 'cmps', 'cmpsb', 'cmpsd', 'comisd', 'comiss', 'fcom', 'fcomp',
    'fcomp5', 'fcompp', 'ficom', 'setnle', 'fucom', 'fucomi', 'fucompp',
    'pcmpeqb', 'pcmpeqd', 'pcmpeqw', 'pcmpgtb', 'pcmpgtd', 'pcmpgtw',
    'pfcmpeq', 'pfcmpge', 'pfcmpgt', 'scasb', 'scasd', 'ucomisd', 'ucomiss'
]
CmpInstDict = {k: v for v, k in enumerate(CmpInst)}

read = ['rdtsc', 'bt', 'bound', 'cpuid', 'ins', 'insb', 'insd']
load = [
    'fist', 'fistp', 'fisttp', 'fld', 'fld1', 'fldcw', 'fldenv', 'lahf', 'lar',
    'ldmxcsr', 'lds', 'lea', 'les', 'lgs', 'lods', 'lodsb', 'lodsd', 'lodsw',
    'out', 'outs', 'outsb', 'outsd', 'pop', 'popa', 'popf', 'popfw',
    'prefetchnta', 'prefetcht0', 'wbinvd'
]


def classifyOperator(operator):
    if operator in TransferInstDict:
        return Instruction.operandTypes['trans']
    elif operator in CallInstDict:
        return Instruction.operandTypes['call']
    elif operator in MathInstDict:
        return Instruction.operandTypes['math']
    elif operator in CmpInstDict:
        return Instruction.operandTypes['cmp']
    elif operator in CryptoInstDict:
        return Instruction.operandTypes['crypto']
    elif operator in MovInstDict:
        return Instruction.operandTypes['mov']
    elif operator in TermInstDict:
        return Instruction.operandTypes['term']
    elif operator in DataInstDict:
        return Instruction.operandTypes['def']
    else:
        return Instruction.operandTypes['other']


class Instruction(object):
    """Abstract assembly instruction, used as default for unknown ones"""
    # Type of instruction, mapped to feature vector index.
    operandTypes = {'trans': 0, 'call': 1, 'math': 2, 'cmp': 3,
                    'crypto': 4, 'mov': 5, 'term': 6, 'def': 7,
                    'other': 8}
    # Type of const values in operator, mapped to feature vector index.
    operatorTypes = {'num_const': len(operandTypes),
                     'str_const': len(operandTypes) + 1}


class Block(object):
    """Block of control flow graph."""
    instDim = len(Instruction.operandTypes) + len(Instruction.operatorTypes)
    """Types of structual-related vertex features"""
    vertexTypes = {'degree': instDim, 'num_inst': instDim + 1}

    @staticmethod
    def getAttributesDim():
        return Block.instDim + len(Block.vertexTypes)


def nodeFeatures(G):
    """
    Extract features in each node: need to be consist with cfg_builder
    """
    features = np.zeros((number_of_nodes(G), Block.getAttributesDim()))
    numConstIdx = Instruction.operatorTypes['num_const']
    strConstIdx = Instruction.operatorTypes['str_const']
    degreeIdx = Block.vertexTypes['degree']
    numInstIdx = Block.vertexTypes['num_inst']
    orderedNodes = []
    for (i, (node, attributes)) in enumerate(sorted(G.nodes(data=True))):
        orderedNodes.append(node)
        instructions = attributes['Ins']
        features[i, degreeIdx] = G.degree(node)
        features[i, numInstIdx] = len(instructions)
        log.debug('%s\'s instructions: %s' % (node, instructions))
        log.debug('%s\'s degree = %s' % (node, G.degree(node)))
        log.debug('%s\'s attributes = %s\n' % (node, attributes))

        for (addr, inst) in instructions:
            if len(inst) == 0:
                log.debug('Empty inst \'%s\' in %s' % (inst, node))
                continue
            """
            Format of assembly code: "operator operand, operand, ..., operand"
            """
            operatorClass = classifyOperator(inst[0])
            features[i, operatorClass] += 1
            if len(inst) == 1:
                log.debug('Inst \'%s\' has no operator' % inst)
                continue

            for part in inst[1].split(','):
                commentIdx = part.find(';')
                operand = part if commentIdx == -1 else part[:commentIdx]
                numericCnts, stringCnts = matchConstant(operand)
                features[i, numConstIdx] += numericCnts
                features[i, strConstIdx] += stringCnts

    return features, orderedNodes
