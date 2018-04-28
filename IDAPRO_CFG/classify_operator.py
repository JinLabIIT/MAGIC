transfer = ['enter', 'leave', 'jcxz', 'iret', 'ja', 'jb', 'jbe',
            'jcxz', 'jecxz', 'jg', 'jge', 'jl', 'jle', 'jmp', 'jnb',
            'jno', 'jnp', 'jns', 'jnz', 'jo', 'jp', 'js', 'jz', 'loop',
            'loope', 'loopne', 'rep', 'repe', 'repne', 'retf', 'retn', 'wait']

call = ['VxDCall', 'call', 'int', 'into']

mov = ['cmovb', 'cmovno', 'cmovz', 'fcmovb', 'fcmovne', 'fcmovu',
       'mov', 'movapd', 'movaps', 'movd', 'movdqa', 'movdqu',
       'movhlps', 'movhps', 'movlhps', 'movlpd', 'movlps', 'movntdq',
       'movntps', 'movntq', 'movq', 'movsb', 'movsd', 'movsldup',
       'movss', 'movsw', 'movsx', 'movups', 'movzx', 'pmovmskb']

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
           'ucomisd', 'ucomiss',
           ]

read = ['rdtsc', 'bt', 'bound', 'cpuid', 'ins', 'insb', 'insd']

load = ['fist', 'fistp', 'fisttp', 'fld', 'fld1', 'fldcw', 'fldenv',
        'lahf', 'lar', 'ldmxcsr', 'lds', 'lea', 'les', 'lgs',
        'lods', 'lodsb', 'lodsd', 'lodsw', 'out', 'outs', 'outsb', 'outsd',
        'pop', 'popa', 'popf', 'popfw', 'prefetchnta', 'prefetcht0',
        'wbinvd']
