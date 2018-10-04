#!/usr/bin/python3.7
import re
import glog as log
from typing import List
from utils import findAddrInOperators


DataInstList = ['dd', 'db', 'dw', 'dq',
                'extrn',
                'unicode',]
DataInstDict = {k: v for v, k in enumerate(DataInstList)}
CallingInstList = ['call',
                   'int', 'into',
                   'syscall', 'sysenter',]
CallingInstDict = {k: v for v, k in enumerate(CallingInstList)}
ConditionalJumpInstList = ['ja', 'jb', 'jbe', 'jcxz', 'jecxz', 'jg', 'jge',
                           'jl', 'jle', 'jnb', 'jno', 'jnp', 'jns', 'jnz',
                           'jo', 'jp', 'js', 'jz',
                           'loop', 'loope', 'loopne', 'loopw',
                           'loopwe', 'loopwne',]
ConditionalJumpInstDict = {k: v for v, k in
                           enumerate(ConditionalJumpInstList)}
UnconditionalJumpInstList = ['jmp']
UnconditionalJumpInstDict = {k: v for v, k in
                             enumerate(UnconditionalJumpInstList)}
EndHereInstList = ['end',
                   'iret', 'iretw',
                   'retf', 'reti', 'retfw', 'retn', 'retnw',
                   'sysexit', 'sysret',
                   'xabort',
                   ]
EndHereInstDict = {k: v for v, k in enumerate(EndHereInstList)}
RepeatInstList = ['rep', 'repe', 'repne']
RepeatInstDict = {k: v for v, k in enumerate(RepeatInstList)}
RegularInstList = ['aaa', 'aad', 'aam', 'aas', 'adc', 'add', 'addpd',
                   'addps', 'addsd', 'addss', 'addsubpd', 'addsubps',
                   'align', 'and', 'andnpd', 'andnps', 'andpd',
                   'andps', 'arpl',
                   'bound', 'bsf', 'bsr', 'bswap', 'bt', 'btc',
                   'btr', 'bts',
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
                   'fabs', 'fadd', 'faddp', 'fbld', 'fbstp', 'fchs',
                   'fclex', 'fcmovb', 'fcmovbe', 'fcmove', 'fcmovnb',
                   'fcmovnbe', 'fcmovne', 'fcmovnu', 'fcmovu', 'fcom',
                   'fcomi', 'fcomip', 'fcomp', 'fcompp', 'fcos',
                   'fdecstp', 'fdiv', 'fdivp',
                   'fdivr', 'fdivrp', 'femms', 'ffree', 'ffreep', 'fiadd',
                   'ficom', 'ficomp', 'fidiv', 'fidivr', 'fild', 'fimul',
                   'fincstp', 'fist', 'fistp', 'fisttp', 'fisub', 'fisubr',
                   'fld', 'fldcw', 'fldenv', 'fldpi', 'fldz', 'fmul',
                   'fmulp', 'fnclex', 'fndisi','fneni', 'fninit', 'fnop',
                   'fnsave', 'fnstcw', 'fnstenv', 'fnstsw', 'fpatan',
                   'fprem', 'fptan', 'frndint', 'frstor', 'fsave',
                   'fscale', 'fsetpm', 'fsin', 'fsincos', 'fsqrt', 'fst',
                   'fstcw', 'fstenv', 'fstp', 'fstsw', 'fsub', 'fsubp',
                   'fsubr', 'fsubrp', 'ftst', 'fucom', 'fucomi',
                   'fucomip', 'fucomp', 'fucompp', 'fxam', 'fxch',
                   'fxsave', 'fxtract',
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
                   'ptest', 'punpckhbw', 'punpckhdq', 'punpckhqdq',
                   'punpckhwd', 'punpcklbw', 'punpckldq', 'punpcklqdq',
                   'punpcklwd', 'push', 'pusha', 'pushaw',
                   'pushf', 'pushfw', 'pxor',
                   'rc', 'rcl', 'rcpps', 'rcpss', 'rcr', 'rdmsr', 'rdpmc',
                   'rdrand', 'rdtsc', 'rol', 'ror', 'roundps', 'rsldt',
                   'rsm', 'rsqrtps', 'rsqrtss', 'rsts',
                   'sahf', 'sal', 'sar', 'sbb', 'scas', 'scasb', 'scasd',
                   'scasw', 'setalc', 'setb', 'setbe', 'setl', 'setle',
                   'setnb', 'setnbe', 'setnl', 'setnle', 'setno', 'setnp',
                   'setns', 'setnz', 'seto', 'setp', 'sets', 'setz',
                   'sfence', 'sgdt', 'shl', 'shld', 'shr', 'shrd',
                   'shufpd', 'shufps', 'sidt', 'sldt', 'sqrtps',
                   'sqrtsd', 'sqrtss', 'stc', 'std', 'sti',
                   'stmxcsr', 'stos', 'stosb', 'stosd',
                   'stosw', 'str', 'sub', 'subpd', 'subps', 'subsd',
                   'subss', 'svldt', 'svts',
                   'test',
                   'ucomisd','ucomiss', 'unpckhpd', 'unpckhps',
                   'unpcklpd', 'unpcklps',
                   'vaddps', 'vaddsd', 'vaddss', 'vaddsubpd', 'vandnpd',
                   'vcmppd', 'vcmpps', 'vcmpss', 'vcomisd', 'vdivpd',
                   'vdivps', 'vdivsd', 'vdivss', 'verw', 'vhsubpd',
                   'vhsubps', 'vmaxpd', 'vmaxsd', 'vmaxss', 'vminsd',
                   'vminss', 'vmload', 'vmovapd', 'vmovaps', 'vmovd',
                   'vmovddup', 'vmovdqa', 'vmovdqu', 'vmovhps', 'vmovlhps',
                   'vmovntdq', 'vmovntpd', 'vmovntps', 'vmovntsd',
                   'vmovsd', 'vmovsldup', 'vmovss', 'vmovupd', 'vmovups',
                   'vmptrld', 'vmptrst', 'vmread', 'vmulps', 'vmulss',
                   'vmwrite', 'vorpd','vpackssdw', 'vpacksswb',
                   'vpackuswb', 'vpaddb', 'vpaddd', 'vpaddq',
                   'vpaddsb', 'vpaddsw',
                   'vpaddusb', 'vpaddusw', 'vpaddw', 'vpandn', 'vpcext',
                   'vpclmulqdq', 'vpcmpeqb', 'vpcmpeqd',
                   'vpcmpeqw', 'vpcmpgtb', 'vpcmpgtd', 'vpcmpgtw',
                   'vpermilps', 'vpermq',
                   'vpextrw', 'vpinsrw', 'vpmaddwd', 'vpmaxub', 'vpmulhuw',
                   'vpmulhw', 'vpmullw', 'vpsadbw', 'vpshufhw', 'vpshuflw',
                   'vpsllq', 'vpsllw', 'vpsrad', 'vpsrld', 'vpsrlq',
                   'vpsrlw', 'vpsubb', 'vpsubd', 'vpsubusb', 'vpunpckhbw',
                   'vpunpckhdq', 'vpunpckhqdq', 'vpunpckhwd', 'vpunpcklbw',
                   'vpunpckldq', 'vpunpcklqdq', 'vpunpcklwd', 'vpxor',
                   'vrcpss', 'vrsqrtss', 'vshufpd', 'vshufps', 'vsqrtpd',
                   'vsqrtps', 'vsqrtsd', 'vsubpd', 'vsubps', 'vsubsd',
                   'vucomiss', 'vunpckhps', 'vunpcklpd', 'vunpcklps',
                   'vxorps', 'vzeroupper',
                   'wait', 'wbinvd', 'wrmsr',
                   'xadd', 'xbegin', 'xchg', 'xlat', 'xor',
                   'xorpd', 'xorps', 'xrstor', 'xsaveopt']
RegularInstDict = {k: v for v, k in enumerate(RegularInstList)}


class Instruction(object):
    """Abstract assembly instruction, used as default for unknown ones"""

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

    def __repr__(self) -> str:
        return "%X: %s" % (self.address, self.operand)


class DataInst(Instruction):
    """Variable/data declaration statements don't have operand"""

    def __init__(self, addr: str, operators: List[str]) -> None:
        super(DataInst, self).__init__(addr, operators=operators)

    def accept(self, builder):
        builder.visitDefault(self)

    def __repr__(self) -> str:
        operators = " ".join(self.operators)
        return "%X: %s %s" % (self.address, self.operand, operators)


class RegularInst(Instruction):
    """Regular instruction"""

    def __init__(self, addr: str,
                 operand: str,
                 operators: List[str]) -> None:
        super(RegularInst, self).__init__(addr, operand, operators)

    def accept(self, builder):
        builder.visitDefault(self)

    def __repr__(self) -> str:
        operators = " ".join(self.operators)
        return "%X: %s %s" % (self.address, self.operand, operators)

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

    def __repr__(self) -> str:
        operators = " ".join(self.operators)
        return "%X: %s %s" % (self.address, self.operand, operators)

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

    def __repr__(self) -> str:
        operators = " ".join(self.operators)
        return "%X: %s %s" % (self.address, self.operand, operators)

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

    def __repr__(self) -> str:
        operators = " ".join(self.operators)
        return "%X: %s %s" % (self.address, self.operand, operators)


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

    def __repr__(self) -> str:
        operators = " ".join(self.operators)
        return "%X: %s %s" % (self.address, self.operand, operators)

class EndHereInst(Instruction):
    """EndHere"""

    def __init__(self, addr: str,
                 operand: str,
                 operators: List[str]) -> None:
        super(EndHereInst, self).__init__(addr, operand, operators)

    def accept(self, builder):
        builder.visitEndHere(self)

    def __repr__(self) -> str:
        operators = " ".join(self.operators)
        return "%X: %s %s" % (self.address, self.operand, operators)

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
