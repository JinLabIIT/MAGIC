.text:00401050 5E 47	start:
.text:00401050 5E 47		jmp     short 401054
.text:00401052 C2 04		bad_interrupt:
.text:00401052 C2 04		reti
.text:00401054 54 56    begin:
.text:00401054 54 56            dd      56Ah
.text:00401054 54 77            dw      49h
.text:00401054 CC CC		mov     edi, eax
.text:00401055 BB AA            db      98h 54h 31h
.text:00401056 47 7B            ?qit@?1??QueryInterface@?$BasePatternProvider@VIVGridProvider@@UIGridProvider@@$02VIVGridProxy@@@@UAGJABU_GUID@@PAPAX@Z@4QBUQITAB@@B QITAB <offset __GUID_b17d6187_0907_464b_a168_0ef17a1572b1, 10h>
.text:00401057 FF DD            xmmword_40382100        xmmword 80000000000000008000000000000000h
.text:00401060 8B 44		mov     eax, [esp+8]
.text:00401064 8A 08	loop:
.text:00401064 8A 08		mov     cl, [eax]
.text:00401066 8B 54		mov     edx, [esp+4]
.text:0040106A 88 0A		call    sub_401084
.text:0040106C C3 55		jnz     loc_00401064 ; loop
.text:0040106D CC CC	again:
.text:0040106D CC CC		sub     eax, 10h
.text:00401070 8B 44		mov     eax, [esp+4]
.text:00401074 8D 50		jnz     short loc_00401054 ; jmp to begin
.text:00401076 8D 50		lea     edx, [eax+1]
.text:00401079 55 41	wait:
.text:00401079 8A 08		mov     cl, [eax]
.text:0040107A 40 45		add     eax, 55h
.text:0040107C 84 C9		call    ??_memcpy_s
.text:0040107E 75 F9		jnz     short loc_401079 ; jmp to wait
.text:00401080 2B C2		sub     eax, edx
.text:00401082 C3 45		retn    4
.text:00401084                  ; =============== S U B R O U T I N E ========================
.text:00401084
.text:00401084	                ; Attributes: bp-based frame
.text:00401084
.text:00401084	        sub_401084      proc near	; CODE XREF: __tzset_nolock+52
.text:00401084	                ; __isindst_nolock+12
.text:00401084
.text:00401084	                arg_0	       = dword ptr  8
.text:00401084                  lpThreadParameter= dword ptr 10
.text:00401084                  ProcessInformation= _PROCESS_INFORMATION ptr -498h
.text:00401084
.text:00401084 8B FF	        mov     edi, edi
.text:00401086 8B 45 08		mov     eax, [ebp+arg_0]
.text:00401088 56			push    esi
.text:0040108A 33 F6		xor     esi, esi
.text:0040108C 3B C6		cmp     eax, esi
.text:0040108E 75 1D		jnz     short loc_4010A3
.text:00401090 E8 CB 71	        call    __errno
.text:00401092 56			push    esi
.text:00401094 56			push    esi
.text:00401096 C7 00            mov     dword ptr [eax],	16h
.text:00401098 E8 38            call    __invalid_parameter
.text:0040109A 83 C4 14		add     esp, 14h
.text:0040109C 6A 16		push    16h
.text:0040109E 58			pop     eax
.text:004010A1 EB 0A		jmp     short loc_4010A9
.text:004010A3				; ---------------------------------
.text:004010A3
.text:004010A3				loc_4010A3:			       ; CODE XREF: sub_41D2CF
.text:004010A3 8B 0D            mov     ecx, dword_43DE14
.text:004010A5 89 08		mov     [eax], ecx
.text:004010A7 33 C0		xor     eax, eax
.text:004010A9
.text:004010A9		loc_4010A9:			       ; CODE XREF: sub_41D2CF
.text:004010A9 5E		pop     esi
.text:004010AB C3		retn
.text:004010AB		sub_401084      endp
.text:004010AC
.text:004010AC ?? ?? ??	?? ?? ?? ?? ?? ?? ?? ??	?? ?? ?? ?? ??+	        dd 200h dup(?)
.text:0042A80E ?? ?? ??	?? ?? ?? ?? ?? ?? ?? ??	?? ?? ?? ?? ??+         _text   ends
.text:0042A80E ?? ?? ??	?? ?? ?? ?? ?? ?? ?? ??	?? ?? ?? ?? ??+
