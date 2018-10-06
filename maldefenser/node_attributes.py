import re
import numpy as np
import networkx as nx
import glog as log
from typing import List


transfer = ['enter', 'leave', 'jcxz', 'ja', 'jb', 'jbe',
            'jcxz', 'jecxz', 'jg', 'jge', 'jl', 'jle', 'jmp', 'jnb',
            'jno', 'jnp', 'jns', 'jnz', 'jo', 'jp', 'js', 'jz', 'loop',
            'loope', 'loopne', 'rep', 'repe', 'repne', 'wait']

call = ['VxDCall', 'call', 'int', 'into']

mov = ['cmovb', 'cmovno', 'cmovz', 'fcmovb', 'fcmovne', 'fcmovu',
       'mov', 'movapd', 'movaps', 'movd', 'movdqa', 'movdqu',
       'movhlps', 'movhps', 'movlhps', 'movlpd', 'movlps', 'movntdq',
       'movntps', 'movntq', 'movq', 'movsb', 'movsd', 'movsldup',
       'movss', 'movsw', 'movsx', 'movups', 'movzx', 'pmovmskb']
terminate = ['end',
             'iret', 'iretw',
             'retf', 'reti', 'retfw', 'retn', 'retnw',
             'sysexit', 'sysret',
             'xabort']
conversion = ['punpckldq', 'cbw', 'cdq',
              'cvtdq2pd', 'cvtdq2ps', 'cvtpi2ps', 'cvtsd2si',
              'cvtsi2sd', 'cvtsi2ss', 'cvttps2pi', 'cvttsd2si',
              'cwd', 'cwde', 'pf2id', 'pi2fd']

crypto = ['aesdec', 'aesdeclast', 'aesenc', 'aesenclast', 'aesimc',
          'aeskeygenassist']

arithemtic = ['aaa', 'aad', 'aam', 'aas', 'adc', 'add', 'addpd', 'addps',
              'addsd', 'addss', 'and', 'andnpd', 'andnps', 'andpd', 'andps',
              'bswap', 'btc', 'btr', 'bts', 'cli', 'cmc', 'daa', 'das', 'dec',
              'div', 'divsd', 'divss', 'emms', 'f2xm1', 'fabs', 'fadd',
              'faddp', 'fchs', 'fclex', 'fcos', 'fdiv', 'fdivp',
              'fdivr', 'fdivrp', 'fiadd', 'fidiv', 'fidivr', 'fimul', 'fisub',
              'fisubr', 'fldl2e', 'fldlg2', 'fldln2', 'fldpi', 'fldz',
              'fmul', 'fpatan', 'fprem', 'fprem1', 'fptan', 'frndint',
              'fscale', 'fsin', 'fsincos', 'fsqrt', 'fsub', 'fsubp', 'fsubr',
              'fsubrp', 'fxch', 'fyl2x', 'idiv', 'imul', 'inc', 'in'
              'maxss', 'minss', 'mul', 'mulpd', 'mulps', 'mulsd', 'mulss',
              'neg', 'not', 'or', 'orpd', 'orps', 'paddb', 'paddb',
              'paddd', 'paddq', 'paddsb', 'paddsw', 'paddusb', 'paddusw'
              'paddw', 'pand', 'pandn', 'pfacc', 'pfadd', 'pfmax', 'pfmin',
              'pfmul', 'pfnacc', 'pfpnacc', 'pfrcp',
              'pfrcpit1', 'pfrcpit2', 'pfrsqit1', 'pfrsqrt',
              'pfsub', 'pfsubr', 'pmaddwd', 'pmaxsw', 'pminsw', 'pminub',
              'pmulhuw', 'pmulhw', 'pmullw', 'pmuludq', 'por', 'pslld',
              'psllq', 'psllw', 'psrad', 'psraw', 'psrld', 'psrldq', 'psrlq',
              'psrlw', 'psubb', 'psubd', 'psubq', 'psubsb', 'psubsw',
              'psubusb', 'psubusw', 'psubw', 'pswapd',
              'punpckhbw', 'punpckhdq', 'punpckhwd', 'punpcklbw',
              'punpckldq', 'punpcklqdq', 'punpcklwd', 'pxor', 'rcl', 'rcpps',
              'rcpss', 'rcr', 'rol', 'ror', 'rsqrtss', 'sal', 'sar', 'sbb',
              'setb', 'setbe', 'setl', 'setle', 'setnb', 'setnbe', 'setnl',
              'setnle', 'setns', 'setnz', 'seto', 'sets', 'setz', 'shl',
              'shld', 'shr', 'shrd', 'sqrtsd', 'sqrtss', 'stc', 'std', 'sti',
              'sub', 'subpd', 'subps', 'subsd', 'subss', 'test',
              'unpckhpd', 'unpckhps', 'unpcklpd', 'unpcklps', 'vpmaxsw',
              'xadd', 'xchg', 'xlat', 'xor', 'xorpd', 'xorps']
arithemtic += conversion

compare = ['cmp', 'cmpnlepd', 'cmpeqsd', 'cmpltpd', 'cmpltsd',
           'cmpneqpd', 'cmpneqps', 'cmpnlepd', 'cmps', 'cmpsb', 'cmpsd',
           'comisd', 'comiss', 'fcom', 'fcomp', 'fcomp5', 'fcompp', 'ficom',
           'setnle', 'fucom', 'fucomi', 'fucompp', 'pcmpeqb',
           'pcmpeqd', 'pcmpeqw', 'pcmpgtb', 'pcmpgtd', 'pcmpgtw',
           'pfcmpeq', 'pfcmpge', 'pfcmpgt', 'scasb', 'scasd',
           'ucomisd', 'ucomiss']

read = ['rdtsc', 'bt', 'bound', 'cpuid', 'ins', 'insb', 'insd']

load = ['fist', 'fistp', 'fisttp', 'fld', 'fld1', 'fldcw', 'fldenv',
        'lahf', 'lar', 'ldmxcsr', 'lds', 'lea', 'les', 'lgs',
        'lods', 'lodsb', 'lodsd', 'lodsw', 'out', 'outs', 'outsb', 'outsd',
        'pop', 'popa', 'popf', 'popfw', 'prefetchnta', 'prefetcht0',
        'wbinvd']


def classifyOperand(operand: str) -> int:
    if operand in transfer:
        log.info(f'{operand} belong to transfer')
        return 0
    elif operand in call:
        log.info(f'{operand} belong to call')
        return 1
    elif operand in arithemtic:
        log.info(f'{operand} belong to arithemtic')
        return 2
    elif operand in compare:
        log.info(f'{operand} belong to compare')
        return 3
    elif operand in crypto:
        log.info(f'{operand} belong to crypto')
        return 4
    elif operand in mov:
        log.info(f'{operand} belong to move')
        return 5
    elif operand in terminate:
        log.info(f'{operand} belong to terminate')
        return 6
    else:
        log.error(f'Unable to classify "{operand}"')
        return 7


def matchConstant(line: str) -> List[int]:
    """Parse the numeric/string constants in an operand"""
    operand = line.strip('\n\r\t ')
    numericCnts = 0
    stringCnts = 0

    """
    Whole operand is a num OR leading num in expression.
    E.g. "0ABh", "589h", "0ABh" in "0ABh*589h"
    """
    wholeNum = r'^([1-9][0-9A-F]*|0[A-F][0-9A-F]*)h?.*'
    pattern = re.compile(wholeNum)
    if pattern.match(operand):
        numericCnts += 1
        log.info(f'Match whole number in {operand}')
        # numerics.append('%s:WHOLE/LEAD' % operand)

    """Number inside expression, exclude the leading one."""
    numInExpr = r'([+*/:]|-)([1-9][0-9A-F]*|0[A-F][0-9A-F]*)h?'
    pattern = re.compile(numInExpr)
    match = pattern.findall(operand)
    if len(match) > 0:
        numericCnts += 1
        log.info(f'Match in-expression number in {operand}')
        # numerics.append('%s:%d' % (operand, len(match)))

    """Const string inside double/single quote"""
    strRe = r'["\'][^"]+["\']'
    pattern = re.compile(strRe)
    match = pattern.findall(operand)
    if len(match) > 0:
        stringCnts += 1
        log.info(f'Match str const in {operand}')
        # strings.append('%s:%d' % (operand, len(match)))

    return [numericCnts, stringCnts]


def nodeFeatures(G: nx.DiGraph):
    """
    Extract features in each node:
    7 operator features + 2 operand features.
    """
    log.info('Extract attributes from blocks')
    features = np.zeros((G.number_of_nodes(), 8 + 2))
    for (i, (node, attributes)) in enumerate(G.nodes(data=True)):
        block = attributes['block']
        log.debug(f'Process block {block.startAddr}')
        for inst in block.instList:
            operator_class = classifyOperand(inst.operand)
            features[i, operator_class] += 1
            for operator in inst.operators:
                numeric_cnts, string_cnts = matchConstant(operator)
                features[i, -2] += numeric_cnts
                features[i, -1] += string_cnts

    return features
