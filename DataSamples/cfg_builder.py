#!/usr/bin/python3.7
import re
import glog as log
import networkx as nx
import pandas as pd
import instructions as isn
import matplotlib.pyplot as plt
from utils import FakeCalleeAddr
from collections import OrderedDict
from typing import List, Dict


class Block(object):
    """Block of control flow graph."""

    def __init__(self) -> None:
        super(Block, self).__init__()
        self.startAddr = -1
        self.endAddr = -1
        self.instList: List[isn.Instruction] = []
        self.edgeList: List[int] = []


class ControlFlowGraphBuilder(object):
    """For building a control flow graph from a program"""

    def __init__(self, binaryId: str) -> None:
        super(ControlFlowGraphBuilder, self).__init__()
        self.cfg = nx.Graph()
        self.instBuilder: isn.InstBuilder = isn.InstBuilder()
        self.binaryId: str = binaryId
        self.programEnd: int = -1
        self.programStart: int = -1

        self.program: Dict[str, str] = {}  # Addr to raw string instruction
        self.addr2Inst: OrderedDict[int, isn.Instruction] = OrderedDict()
        self.addr2Block: Dict[int, Block] = {}

    def build(self) -> None:
        self.extractTextSeg()
        self.createProgram()
        self.discoverEntries()
        self.connectBlocks()
        self.exportToNxGraph()

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
            log.error(f'Unable to aggregate instructions for addr {addr}')
            for inst in validInst:
                log.error('%s: %s' % (addr, inst))

    def createProgram(self) -> None:
        """Generate unique-addressed program, store in self.program"""
        currAddr = -1
        sameAddrInsts = []
        with open(self.binaryId + ".txt", 'r') as txtFile:
            for line in txtFile:
                elems = line.rstrip('\n').split(' ')
                addr, inst = elems[0], elems[1:]
                if currAddr == -1:
                    currAddr = addr
                    sameAddrInsts.append(" ".join(inst))
                else:
                    if addr != currAddr:
                        log.debug(f"Aggreate {len(sameAddrInsts)} insts for addr {currAddr}")
                        self.aggregate(currAddr, sameAddrInsts)
                        sameAddrInsts.clear()

                    currAddr = addr
                    sameAddrInsts.append(" ".join(inst))

            if len(sameAddrInsts) > 0:
                log.debug(f"Aggreate {len(sameAddrInsts)} insts for addr {currAddr}")
                self.aggregate(currAddr, sameAddrInsts)
                sameAddrInsts.clear()

            self.saveProgram()

    def discoverEntries(self) -> None:
        """Create Instruction object for each address, store in addr2Inst"""
        prevAddr = -1
        with open(self.binaryId + '.prog', 'r') as progFile:
            for line in progFile:
                inst = self.instBuilder.createInst(line)
                if prevAddr != -1:
                    self.addr2Inst[prevAddr].size = inst.address - prevAddr

                self.addr2Inst[inst.address] = inst
                log.info(f'{inst.address} {inst.operand}')
                if self.programStart == -1:
                    self.programStart = inst.address
                self.programEnd = max(inst.address, self.programEnd)
                prevAddr = inst.address
            # Last inst get default size 2
            self.addr2Inst[prevAddr].size = 2

        log.info(f'Program starts at {self.programStart:x} ends at {self.programEnd:x}')
        for addr, inst in self.addr2Inst.items():
            inst.accept(self)

    def enter(self, address: int) -> None:
        if address < 0 or address >= self.programEnd:
            log.error(f'Unable to enter instruction at {address:x}')
        else:
            log.info(f'Enter instruction at {address:x}')
            self.addr2Inst[address].start = True

    def branch(self, inst) -> None:
        branchToAddr = inst.findAddrInInst()
        self.addr2Inst[inst.address].branchTo = branchToAddr
        log.info(f'Found branch from {inst.address:x} to {branchToAddr:x}')
        self.enter(branchToAddr)
        self.enter(inst.address + inst.size)

    def call(self, inst) -> None:
        self.addr2Inst[inst.address].call = True
        # Likely NOT able to find callee's address
        callAddr = inst.findAddrInInst()
        if callAddr != FakeCalleeAddr:
            log.info(f'Found call from {inst.address:x} to {callAddr:x}')
        else:
            log.info(f'Fake call from {inst.address:x} to FakeCalleeAddr')

        self.addr2Inst[inst.address].branchTo = callAddr
        self.enter(callAddr)
        self.enter(inst.address + inst.size)

    def jump(self, inst) -> None:
        jumpAddr = inst.findAddrInInst()
        self.addr2Inst[inst.address].fallThrough = False
        self.addr2Inst[inst.address].branchTo = jumpAddr
        log.info(f'Found jump from {inst.address:x} to {jumpAddr:x}')
        self.enter(jumpAddr)
        self.enter(inst.address + inst.size)

    def end(self, inst) -> None:
        self.addr2Inst[inst.address].fallThrough = False
        log.info(f'Found end at {inst.address:x}')
        self.enter(inst.address + inst.size)

    def visitDefault(self, inst) -> None:
        pass

    def visitCall(self, inst) -> None:
        self.call(inst)

    def visitJmp(self, inst) -> None:
        self.jump(inst)

    def visitJnz(self, inst) -> None:
        self.branch(inst)

    def visitReti(self, inst) -> None:
        self.end(inst)

    def visitRetn(self, inst) -> None:
        self.end(inst)

    def getBlockAtAddr(self, addr: int) -> Block:
        if addr not in self.addr2Block:
            block = Block()
            block.startAddr = addr
            self.addr2Block[addr] = block
            log.info(f'Create new block starting at {addr:x}')

        return self.addr2Block[addr]

    def connectBlocks(self) -> None:
        """Group instructions into blocks connected based on branch and fall through"""
        log.info('**** Create and connecting blocks ****')
        currBlock = None
        for (addr, inst) in sorted(self.addr2Inst.items()):
            if currBlock is None or inst.start is True:
                currBlock = self.getBlockAtAddr(addr)
            nextAddr = addr + inst.size
            nextBlock = currBlock
            if nextAddr in self.addr2Inst:
                nextInst = self.addr2Inst[nextAddr]
                if inst.fallThrough is True and nextInst.start is True:
                    nextBlock = self.getBlockAtAddr(nextAddr)
                    currBlock.edgeList.append(nextBlock.startAddr)
                    log.info(f'block {currBlock.startAddr:x} => next {nextBlock.startAddr:x}')

            if inst.branchTo > 0 or inst.branchTo == FakeCalleeAddr:
                block = self.getBlockAtAddr(inst.branchTo)
                currBlock.edgeList.append(block.startAddr)
                log.info(f'block {currBlock.startAddr:x} => branch {block.startAddr:x}')

            currBlock.instList.append(inst)
            currBlock.endAddr = max(currBlock.endAddr, inst.address)
            self.addr2Block[currBlock.startAddr] = currBlock
            currBlock = nextBlock

    def exportToNxGraph(self):
        """Assume block/node is represented by its startAddr"""
        for (addr, block) in self.addr2Block.items():
            self.cfg.add_node(addr, block=block)

        for (addr, block) in self.addr2Block.items():
            for neighboor in block.edgeList:
                self.cfg.add_edge(addr, neighboor)

        log.debug(f'#nodes in cfg: {nx.number_of_nodes(self.cfg)}')
        log.debug(f'#edges in cfg: {nx.number_of_edges(self.cfg)}')

    def drawCfg(self) -> None:
        nx.draw(self.cfg)
        plt.show()

    def printCfg(self):
        log.info('**** Print CFG ****')
        for (addr, block) in sorted(self.addr2Block.items()):
            log.info(f'block {addr:x} [{block.startAddr:x}, {block.endAddr:x}]')
            for neighboor in block.edgeList:
                log.info(f'block {addr:x} -> {neighboor:x}')

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
    binaryIds = [
        'test',
        ]
    # '0A32eTdBKayjCWhZqDOQ', '01azqd4InC7m9JpocGv5', '0ACDbR5M3ZhBJajygTuf']
    seenInst = set()
    for bId in binaryIds:
        log.info('Processing ' + bId + '.asm')
        cfgBuilder = ControlFlowGraphBuilder(bId)
        cfgBuilder.build()
        log.debug('%d unique insts in %s.asm' %
                  (len(cfgBuilder.instBuilder.seenInst), bId))
        seenInst = seenInst.union(cfgBuilder.instBuilder.seenInst)

    exportSeenInst(seenInst)
