.text:00401050 5E	47						       start:
.text:00401050 5E	47						       jmp     short 401054
.text:00401051 C2 04							     bad_interrupt:
.text:00401051 C2 04							     reti
.text:00401054 54 56                   begin:
.text:00401054 CC CC			             mov     edi, eax
.text:00401060 8B 44						       mov     eax, [esp+8]
.text:00401064 8A 08							     loop:
.text:00401064 8A 08							     mov     cl, [eax]
.text:00401066 8B 54						       mov     edx, [esp+4]
.text:0040106A 88 0A							     mov     [edx], cl
.text:0040106C C3	55						       jnz     loc_00401064; loop
.text:0040106D CC CC							     again:
.text:0040106D CC CC							     sub     eax, 10h
.text:00401070 8B 44						       mov     eax, [esp+4]
.text:00401074 8D 50							     jnz      short loc_00401054; jmp to begin
.text:00401074 8D 50							     lea     edx, [eax+1]
.text:00401077 55	41					         wait:
.text:00401077 8A 08							     mov     cl, [eax]
.text:00401079 40	45						       add     eax, 55h
.text:0040107A 84 C9							     call    _memcpy_s
.text:0040107C 75 F9							     jnz     short loc_401077; jmp to wait
.text:0040107E 2B C2							     sub     eax, edx
.text:00401080 C3	45						       retn    4
