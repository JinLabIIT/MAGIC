import numpy as np
from networkx import number_of_nodes
from dp_utils import matchConstant
from instructions import Instruction
from cfg_builder import Block

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


def classifyOperator(operator: str):
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


def nodeFeatures(G):
    """
    Extract features in each node: trying to be consist with CfgBuilder
    """
    features = np.zeros((number_of_nodes(G), Block.getAttributesDim()))
    for (i, (node, attributes)) in enumerate(G.nodes(data=True)):
        instructions = attributes['Ins']
        for (addr, inst) in instructions:
            if len(inst) == 0:
                break
            """
            Format of assembly code: "operator operand, operand, ..., operand"
            """
            operatorClass = classifyOperator(inst[0])
            features[i, operatorClass] += 1
            if len(inst) == 1:
                continue

            for part in inst[1].split(','):
                commentIdx = part.find(';')
                operand = part if commentIdx == -1 else part[:commentIdx]
                numericCnts, stringCnts = matchConstant(operand)
                features[i, -2] += numericCnts
                features[i, -1] += stringCnts

    return features
