#!/usr/bin/python3.7
import re
import glog as log
import networkx as nx
import pandas as pd
from collections import OrderedDict
from typing import List, Dict
import instructions as isn


class Block(object):
    """Block of control flow graph."""

    def __init__(self) -> None:
        super(Block, self).__init__()
        self.startAddr = -1
        self.endAddr = -1
        self.instList: List[isn.Instruction] = []
        self.edgeList: List[Block] = []


class ControlFlowGraphBuilder(object):
    """For building a control flow graph from a program"""

    def __init__(self, binaryId: str) -> None:
        super(ControlFlowGraphBuilder, self).__init__()
        self.cfg = nx.Graph()
        self.instBuilder: isn.InstBuilder = isn.InstBuilder()
        self.binaryId: str = binaryId
        self.addr2Inst: OrderedDict[int, isn.Instruction] = {}
        self.program: Dict[str, str] = {}  # Addr to raw string instruction
        self.programEnd: int = -1

    def build(self) -> None:
        self.extractTextSeg()
        self.createProgram()
        self.discoverEntries()
        self.connectBlocks()

    def extractTextSeg(self) -> None:
        """Extract text segment from .asm file"""
        lineNum = 1
        bytePattern = re.compile(r'[A-Z0-9][A-Z0-9]')
        imcompleteByte = re.compile(r'\?\?')
        fileInput = open(self.binaryId + '.asm', 'rb')
        fileOutput = open(self.binaryId + '.txt', 'w')
        for line in fileInput:
            elems = line.split()
            decodedElems = [x.decode("utf-8", "ignore") for x in elems]
            seg = decodedElems.pop(0)
            if seg.startswith('.text') is False:
                # Since text segment maynot always be the head, we cannot break
                log.debug("Line %d is out of text segment" % lineNum)
                continue
            else:
                addr = seg[6:]

            if len(decodedElems) > 0 and imcompleteByte.match(decodedElems[0]):
                log.warning("Ignore imcomplete code at line %d: %s" %
                            (lineNum, " ".join(decodedElems)))
                continue

            startIdx = 0
            while startIdx < len(decodedElems):
                if bytePattern.match(decodedElems[startIdx]):
                    startIdx += 1
                else:
                    break

            if startIdx == len(decodedElems):
                log.debug("No instructions at line %d: %s" % (lineNum, elems))
                continue

            if ';' in decodedElems:
                endIdx = decodedElems.index(';')
            else:
                endIdx = len(decodedElems)

            instElems = [addr] + decodedElems[startIdx: endIdx]
            if len(instElems) > 1:
                log.debug("Processed line %d: '%s' => '%s'" %
                          (lineNum,
                           " ".join(decodedElems),
                           " ".join(instElems)))
                fileOutput.write(" ".join(instElems) + '\n')

            lineNum += 1

        fileInput.close()
        fileOutput.close()

    def isHeaderInfo(self, sameAddrInsts: List[str]) -> bool:
        for inst in sameAddrInsts:
            if inst.startswith('_text segment'):
                return True

        return False

    def aggregate(self, addr: str, sameAddrInsts: List[str]) -> None:
        """
        Case 1: Header info
        Case 2: 'xxxxxx proc near' => keep last inst
        Case 3: 'xxxxxx endp' => ignore second
        Case 4: dd, db, dw instructions => d? var_name
        Case 5: location label followed by regular inst
        Case 6: Just 1 regular inst
        """
        if self.isHeaderInfo(sameAddrInsts):
            self.program[addr] = sameAddrInsts[-1]
            return

        validInst = []
        foundDataDeclare = None
        for inst in sameAddrInsts:
            if inst.find('proc near') != -1:
                continue
            if inst.find('public') != -1:
                continue
            if inst.find('assume') != -1:
                continue
            if inst.find('endp') != -1 or inst.find('ends') != -1:
                continue
            if inst.find(' = ') != -1:
                continue
            if inst.startswith('dw ') or inst.find(' dw ') != -1:
                foundDataDeclare = inst
                continue
            if inst.startswith('dd ') or inst.find(' dd ') != -1:
                foundDataDeclare = inst
                continue
            if inst.startswith('db ') or inst.find(' db ') != -1:
                foundDataDeclare = inst
                continue
            if inst.endswith(':'):
                continue
            validInst.append(inst)

        if len(validInst) == 1:
            self.program[addr] = validInst[0]
        elif foundDataDeclare is not None:
            self.program[addr] = foundDataDeclare
            log.debug('Convert data declare into general format')
        else:
            log.error('Unable to aggregate instructions for addr %s:' % addr)
            for inst in validInst:
                log.error('%s: %s' % (addr, inst))

    def createProgram(self) -> None:
        currAddr = -1
        sameAddrInsts = []
        txtFile = open(self.binaryId + ".txt", 'r')
        for line in txtFile:
            elems = line.rstrip('\n').split(' ')
            addr, inst = elems[0], elems[1:]
            if currAddr == -1:
                currAddr = addr
                sameAddrInsts.append(" ".join(inst))
            else:
                if addr != currAddr:
                    log.debug("Aggreate %d insts for addr %s" %
                              (len(sameAddrInsts), currAddr))
                    self.aggregate(currAddr, sameAddrInsts)
                    sameAddrInsts.clear()

                currAddr = addr
                sameAddrInsts.append(" ".join(inst))

        txtFile.close()
        self.saveProgram()

    def discoverEntries(self) -> None:
        """Create Instruction object for each address"""
        prevAddr = -1
        with open(self.binaryId + '.prog', 'r') as progFile:
            for line in progFile:
                inst = self.instBuilder.createInst(line)
                if prevAddr != -1:
                    self.addr2Inst[prevAddr].size = inst.address - prevAddr

                self.addr2Inst[inst.address] = inst
                self.programEnd = max(inst.address, self.programEnd)
                prevAddr = inst.address
            # Last inst get default size 2
            self.addr2Inst[prevAddr].size = 2

        for addr, inst in self.addr2Inst.items():
            inst.accept(self)

    def enter(self, address: int) -> None:
        if address < 0 or address >= self.programEnd:
            log.error('Unable to enter instruction at %d' % address)

        self.addr2Inst[address].start = True

    def branch(self, inst: isn.Instruction) -> None:
        branchToAddr = inst.findAddrInInst()
        self.addr2Inst[inst.address].branchTo = branchToAddr
        log.info('Found branch from %d to %d' % (inst.address, branchToAddr))
        self.enter(branchToAddr)
        self.enter(inst.address + inst.size)

    def call(self, inst: isn.Instruction) -> None:
        callAddr = inst.findAddrInInst()
        self.addr2Inst[inst.address].call = True
        self.addr2Inst[inst.address].branchTo = callAddr
        log.info('Found call from %d to %d' % (inst.address, callAddr))
        self.enter(callAddr)
        self.enter(inst.address + inst.size)

    def jump(self, inst: isn.Instruction) -> None:
        jumpAddr = inst.findAddrInst()
        self.addr2Inst[inst.address].fallThrough = False
        self.addr2Inst[inst.address].branchTo = jumpAddr
        log.info('Found jump from %d to %d' % (inst.address, jumpAddr))
        self.enter(jumpAddr)
        self.enter(inst.address + inst.size)

    def end(self, inst: isn.Instruction) -> None:
        self.addr2Inst[inst.address].fallThrough = False
        log.info('Found end at %d' % (inst.address))
        self.enter(inst.address + inst.size)

    def visitDefault(self, inst: isn.Instruction) -> None:
        pass

    def visitAdd(self, inst: isn.Instruction) -> None:
        self.add(inst)

    def visitAlign(self, inst: isn.Instruction) -> None:
        self.add(inst)

    def connectBlocks(self) -> None:
        pass

    def saveProgram(self) -> None:
        progFile = open(self.binaryId + '.prog', 'w')
        for (addr, inst) in self.program.items():
            progFile.write(addr + ' ' + inst + '\n')
        progFile.close()


def exportSeenInst(seenInst: set):
    instColumn = {'Inst': sorted(list(seenInst))}
    df = pd.DataFrame(data=instColumn)
    df.to_csv('seen_inst.csv')


if __name__ == '__main__':
    log.setLevel("INFO")
    binaryIds = ['0A32eTdBKayjCWhZqDOQ', '01azqd4InC7m9JpocGv5',
                 '0ACDbR5M3ZhBJajygTuf']
    seenInst = set()
    for bId in binaryIds:
        log.info('Processing ' + bId + '.asm')
        cfgBuilder = ControlFlowGraphBuilder(bId)
        cfgBuilder.build()
        log.debug('%d unique insts in %s.asm' %
                  (len(cfgBuilder.instBuilder.seenInst), bId))
        seenInst = seenInst.union(cfgBuilder.instBuilder.seenInst)

    exportSeenInst(seenInst)
