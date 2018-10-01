#!/usr/bin/python3.7
import re
import glog as log
from typing import List
from utils import findAddrInOperators, FakeCalleeAddr, AddrNotFound

DataInstList = ['dd', 'db', 'dw', 'dq',
                'extrn']
CallingInstList = ['call',
                   'int', 'into']
ConditionalJumpInstList = ['ja', 'jb', 'jbe', 'jcxz', 'jecxz', 'jg', 'jge',
                           'jl', 'jle', 'jnb', 'jno', 'jnp', 'jns', 'jnz',
                           'jo', 'jp', 'js', 'jz',
                           'loop', 'loope', 'loopne', 'loopw',
                           'loopwe', 'loopwne',]
UnconditionalJumpInstList = ['jmp']
EndHereInstList = ['end',
                   'iret', 'iretw',
                   'reti', 'retn']
RegularInstList = ['aaa', 'aad', 'aam', 'aas', 'adc', 'add', 'addpd', 'addps',
                   'addsd', 'addss', 'addsubpd', 'addsubps', 'align', 'and',
                   'andnpd', 'andnps', 'andpd', 'andps', 'arpl',
                   'bound', 'bsf', 'bsr', 'bswap', 'bt', 'btc', 'btr', 'bts',
                   'cbw', 'cdq', 'clc', 'cld', 'clflush', 'cli', 'clts',
                   'cmc', 'cmova', 'cmovb', 'cmovbe', 'cmovg', 'cmovge',
                   'cmovl', 'cmovle', 'cmovnb', 'cmovno', 'cmovnp',
                   'cmovns', 'cmovnz', 'cmovo', 'cmovp', 'cmovs', 'cmovz',
                   'cmp', 'cmpeqps', 'cmpeqsd', 'cmpeqss', 'cmpleps',
                   'cmplesd', 'cmpltpd', 'cmpltps', 'cmpltsd', 'cmpneqpd',
                   'cmpneqps', 'cmpnlepd', 'cmpnlesd', 'cmpps', 'cmps',
                   'cmpsb', 'cmpsd', 'cmpsw', 'cmpxchg', 'comisd',
                   'comiss', 'cpuid', 'cwd', 'cwde',
                   'daa', 'das', 'dec', 'div', 'divps', 'divsd', 'divss',
                   'emms', 'enter', 'enterw', 'extractps',
                   'fabs', 'fadd', 'faddp', 'fbld', 'fbstp', 'fchs', 'fclex',
                   'fcmovb', 'fcmovbe', 'fcmove', 'fcmovnb', 'fcmovnbe',
                   'fcmovne', 'fcmovnu', 'fcmovu', 'fcom', 'fcomi', 'fcomip',
                   'fcomp', 'fcompp', 'fcos', 'fdecstp', 'fdiv', 'fdivp',
                   'fdivr', 'fdivrp', 'femms', 'ffree', 'ffreep', 'fiadd',
                   'ficom', 'ficomp', 'fidiv', 'fidivr', 'fild', 'fimul',
                   'fincstp', 'fist', 'fistp', 'fisttp', 'fisub', 'fisubr',
                   'fld', 'fldcw', 'fldenv', 'fldpi', 'fldz', 'fmul',
                   'fmulp', 'fnclex', 'fndisi','fneni', 'fninit', 'fnop',
                   'fnsave', 'fnstcw', 'fnstenv', 'fnstsw', 'fpatan',
                   'fprem', 'fptan', 'frndint', 'frstor', 'fsave',
                   'fscale', 'fsetpm', 'fsin', 'fsincos', 'fsqrt', 'fst',
                   'fstcw', 'fstenv', 'fstp', 'fstsw', 'fsub', 'fsubp',
                   'fsubr', 'fsubrp', 'ftst', 'fucom', 'fucomi', 'fucomip',
                   'fucomp', 'fucompp', 'fxam', 'fxch', 'fxsave', 'fxtract',
                   'getsec',
                   'hlt', 'hnt', 'ht',
                   'icebp', 'idiv', 'imul', 'in', 'inc', 'ins', 'insb',
                   'insd', 'insertps', 'insw', 'invd',
                   'lahf', 'lar', 'lddqu', 'ldmxcsr', 'lds', 'lea',
                   'leave', 'leavew', 'les', 'lfence', 'lfs', 'lgdt',
                   'lgs', 'lidt', 'lldt', 'lock', 'lods', 'lodsb',
                   'lodsd', 'lodsw', 'lsl', 'lss',
                   'maskmovq', 'maxps', 'maxss', 'minps', 'minsd', 'minss',
                   'mov', 'movapd', 'movaps', 'movd', 'movdqa', 'movdqu',
                   'movhlps', 'movhpd', 'movhps', 'movlhps', 'movlpd',
                   'movlps', 'movmskpd', 'movmskps', 'movntdq', 'movnti',
                   'movntps', 'movntq', 'movq', 'movs', 'movsb', 'movsd',
                   'movss', 'movsw', 'movsx', 'movups', 'movzx', 'mpsadbw',
                   'mul', 'mulpd', 'mulps', 'mulsd', 'mulss',
                   'neg', 'nop', 'not',
                   'or', 'orpd', 'orps', 'out', 'outs',
                   'outsb', 'outsd', 'outsw',
                   'pabsw', 'packssdw', 'packsswb', 'packusdw',
                   'packuswb', 'paddb', 'paddd', 'paddq', 'paddsb',
                   'paddsw', 'paddusb', 'paddusw', 'paddw', 'palignr',
                   'pand', 'pandn', 'pause', 'pavgb', 'pavgusb', 'pavgw',
                   'pblendvb', 'pcmpeqb', 'pcmpeqd', 'pcmpeqw', 'pcmpgtb',
                   'pcmpgtd', 'pcmpgtw', 'pextrb', 'pextrd', 'pextrw',
                   'pfacc', 'pfadd', 'pfcmpeq', 'pfcmpge', 'pfcmpgt',
                   'pfmax', 'pfmin', 'pfmul', 'pfnacc', 'pfpnacc',
                   'pfrcp', 'pfrsqrt', 'pfsub', 'pfsubr', 'phaddd',
                   'phaddw', 'phminposuw', 'phsubd', 'pinsrd',
                   'pinsrw', 'pmaddubsw', 'pmaddwd', 'pmaxsw', 'pmaxub',
                   'pminsw', 'pminub', 'pminud', 'pminuw', 'pmovmskb',
                   'pmovzxbd', 'pmovzxwd', 'pmulhrsw', 'pmulhuw', 'pmulhw',
                   'pmullw', 'pmuludq', 'pop', 'popa', 'popaw', 'popf',
                   'popfw', 'por', 'pqit', 'prefetch', 'prefetchnta',
                   'prefetchw', 'psadbw', 'pshufb', 'pshufd', 'pshufhw',
                   'pshuflw', 'pshufw', 'psignw', 'pslld', 'pslldq',
                   'psllq', 'psllw', 'psrad', 'psraw','psrld', 'psrldq',
                   'psrlq', 'psrlw', 'psubb', 'psubd', 'psubq', 'psubsb',
                   'psubsw', 'psubusb', 'psubusw', 'psubw', 'pswapd',
                   'ptest', 'punpckhbw', 'punpckhdq', 'punpckhqdq', 'punpckhwd', 'punpcklbw', 'punpckldq', 'punpcklqdq', 'punpcklwd',
                   'push', 'pusha', 'pushaw', 'pushf', 'pushfw', 'pxor',
                   'rc', 'rcl', 'rcpps', 'rcpss', 'rcr', 'rdmsr', 'rdpmc',
                   'rdrand', 'rdtsc',
                   ]


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
