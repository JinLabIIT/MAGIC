#!/usr/bin/python3.7
import re
import os
import glog as log
import networkx as nx
import instructions as isn
import numpy as np
import matplotlib.pyplot as plt
from utils import FakeCalleeAddr, addCodeSegLog, InvalidAddr
from collections import OrderedDict
from typing import List, Dict


class Block(object):
    """Block of control flow graph."""
    instDim = len(isn.Instruction.operandTypes) + \
        len(isn.Instruction.operatorTypes)

    def __init__(self) -> None:
        super(Block, self).__init__()
        self.startAddr = -1
        self.endAddr = -1
        self.instList: List[isn.Instruction] = []
        self.edgeList: List[int] = []

    def getAttributes(self):
        instAttr = np.zeros((1, Block.instDim))
        for inst in self.instList:
            attr = inst.getOperandFeatures()
            attr += inst.getOperatorFeatures()
            instAttr += np.array(attr)

        degree = len(self.edgeList)
        numInst = len(self.instList)
        return np.concatenate((instAttr, [degree, numInst]), axis=None)

    @staticmethod
    def getAttributesDim():
        return Block.instDim + 2


class ControlFlowGraphBuilder(object):
    """For building a control flow graph from a program"""

    def __init__(self, binaryId: str, pathPrefix: str) -> None:
        super(ControlFlowGraphBuilder, self).__init__()
        self.cfg = nx.DiGraph()
        self.instBuilder: isn.InstBuilder = isn.InstBuilder()
        self.binaryId: str = binaryId
        self.filePrefix: str = pathPrefix + '/' + binaryId
        self.programEnd: int = -1
        self.programStart: int = -1

        self.program: Dict[str, str] = {}  # Addr to raw string instruction
        self.addr2Inst: OrderedDict[int, isn.Instruction] = OrderedDict()
        self.addr2InstAux: OrderedDict[int, isn.Instruction] = OrderedDict()

        self.addr2Block: Dict[int, Block] = {}

    def getControlFlowGraph(self) -> nx.DiGraph:
        self.buildControlFlowGraph()
        self.exportToNxGraph()
        return self.cfg

    def buildControlFlowGraph(self) -> None:
        self.parseInstructions()
        self.parseBlocks()

    def parseInstructions(self) -> None:
        """First pass on instructions"""
        self.extractTextSeg()
        self.createProgram()
        self.discoverInsts()
        # self.clearTmpFiles()

    def parseBlocks(self) -> None:
        """Second pass on blocks"""
        self.visitInsts()
        self.connectBlocks()

    def addrInCodeSegment(self, seg: str) -> str:
        segNames = ['.text:', 'CODE:', 'UPX1:', 'seg000:', 'qmoyiu:',
                    '.UfPOkc:', '.brick:', '.icode:', 'seg001:',
                    '.Much:', 'iuagwws:', '.idata:', '.IqR:', '.data:',
                    ]
        for prefix in segNames:
            if seg.startswith(prefix) is True:
                return seg[len(prefix):]

        return "NotInCodeSeg"

    def indexOfInst(self, decodedElems: List[str]) -> int:
        idx = 0
        bytePattern = re.compile(r'^[A-F0-9][A-F0-9]\+?$')
        while idx < len(decodedElems) and bytePattern.match(decodedElems[idx]):
            idx += 1

        return idx

    def indexOfComment(self, decodedElems: List[str]) -> int:
        for (i, elem) in enumerate(decodedElems):
            if elem.find(';') != -1:
                return i

        return len(decodedElems)

    def extractTextSeg(self) -> None:
        """Extract text segment from .asm file"""
        log.info(f'**** Extract code segment from {self.binaryId}.asm ****')
        lineNum = 1
        imcompleteByte = re.compile(r'^\?\?$')
        fileInput = open(self.filePrefix + '.asm', 'rb')
        fileOutput = open(self.filePrefix + '.text', 'w')
        for line in fileInput:
            elems = line.split()
            decodedElems = [x.decode("utf-8", "ignore") for x in elems]
            if len(decodedElems) == 0:
                lineNum += 1
                continue

            seg = decodedElems.pop(0)
            addr = self.addrInCodeSegment(seg)
            if addr is "NotInCodeSeg":
                # Since text segment maynot always be the head, we cannot break
                log.debug("Line %d is out of text segment" % lineNum)
                lineNum += 1
                continue

            if len(decodedElems) > 0 and imcompleteByte.match(decodedElems[0]):
                log.debug(f'Ignore imcomplete code at line {lineNum}: {" ".join(decodedElems)}')
                lineNum += 1
                continue

            startIdx = self.indexOfInst(decodedElems)
            endIdx = self.indexOfComment(decodedElems)
            if startIdx < endIdx:
                instElems = [addr] + decodedElems[startIdx: endIdx]
                s1, s2 = ' '.join(decodedElems), ' '.join(instElems)
                log.debug(f"Processed line {lineNum}: '{s1}' => '{s2}'")
                fileOutput.write(" ".join(instElems) + '\n')
            else:
                l = " ".join(decodedElems)
                log.debug(f'No instruction at line {lineNum}: {l}')

            lineNum += 1

        fileInput.close()
        fileOutput.close()

    def isHeaderInfo(self, sameAddrInsts: List[str]) -> bool:
        for inst in sameAddrInsts:
            if inst.startswith('_text segment') or inst.find('.mmx') != -1:
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

        validInst: List[str] = []
        foundDataDeclare: str = ''
        ptrPattern = re.compile(r'.+=.+ ptr .+')
        for inst in sameAddrInsts:
            if inst.find('proc near') != -1 or inst.find('proc far') != -1:
                continue
            if inst.find('public') != -1:
                continue
            if inst.find('assume') != -1:
                continue
            if inst.find('endp') != -1 or inst.find('ends') != -1:
                continue
            if inst.find(' = ') != -1 or ptrPattern.match(inst):
                log.debug(f'Ptr declare found: {inst}')
                foundDataDeclare += inst + ' '
                continue
            if inst.startswith('dw ') or inst.find(' dw ') != -1:
                foundDataDeclare += inst + ' '
                continue
            if inst.startswith('dd ') or inst.find(' dd ') != -1:
                foundDataDeclare += inst + ' '
                continue
            if inst.startswith('db ') or inst.find(' db ') != -1:
                foundDataDeclare += inst + ' '
                continue
            if inst.startswith('unicode '):
                foundDataDeclare += inst + ' '
                continue
            if inst.endswith(':'):
                continue
            validInst.append(inst)

        if len(validInst) == 1:
            progLine = validInst[0] + ' ' + foundDataDeclare
            self.program[addr] = progLine.rstrip(' ')
        elif len(foundDataDeclare.rstrip(' ')) > 0:
            self.program[addr] = foundDataDeclare.rstrip(' ')
            log.debug('Convert all data declare into unified inst')
        else:
            # Concat unaggregatable insts
            log.debug(f'Unable to aggregate instructions at {addr}')
            progLine = ''
            for inst in validInst:
                progLine += inst.rstrip('\n\\') + ' '
                log.debug('%s: %s' % (addr, inst))

            log.debug(f'Concat to: {progLine}')
            self.program[addr] = progLine.rstrip(' ')

    def createProgram(self) -> None:
        """Generate unique-addressed program, store in self.program"""
        log.info('**** Aggreate to unique-addressed instructions ****')
        currAddr = -1
        sameAddrInsts = []
        with open(self.filePrefix + ".text", 'r') as textFile:
            for line in textFile:
                elems = line.rstrip('\n').split(' ')
                addr, inst = elems[0], elems[1:]
                if currAddr == -1:
                    currAddr = addr
                    sameAddrInsts.append(" ".join(inst))
                else:
                    if addr != currAddr:
                        self.aggregate(currAddr, sameAddrInsts)
                        sameAddrInsts.clear()

                    currAddr = addr
                    sameAddrInsts.append(" ".join(inst))

            if len(sameAddrInsts) > 0:
                self.aggregate(currAddr, sameAddrInsts)
                sameAddrInsts.clear()

            if len(self.program) == 0:
                log.error(f'No code extracted from {self.filePrefix}.asm')

            self.saveProgram()

    def discoverInsts(self) -> None:
        """Create Instruction object for each address, store in addr2Inst"""
        log.info('**** Discover instructions ****')
        prevAddr = -1
        with open(self.filePrefix + '.prog', 'r') as progFile:
            for line in progFile:
                inst = self.instBuilder.createInst(line)
                if prevAddr != -1:
                    self.addr2Inst[prevAddr].size = inst.address - prevAddr

                self.addr2Inst[inst.address] = inst
                log.debug(f'{inst.address:x} {inst.operand}')
                if self.programStart == -1:
                    self.programStart = inst.address
                self.programEnd = max(inst.address, self.programEnd)
                prevAddr = inst.address

            # Last inst get default size 2
            if prevAddr > 0:
                self.addr2Inst[prevAddr].size = 2
            else:
                addCodeSegLog(self.binaryId)

        log.info(f'Program range [{self.programStart:x}, {self.programEnd:x}]')

    def visitInsts(self) -> None:
        log.info('**** Visit instructions ****')
        for addr, inst in self.addr2Inst.items():
            inst.accept(self)

        self.addr2Inst.update(self.addr2InstAux)

    def addAuxilaryInst(self, addr, operandName='') -> None:
        if addr not in self.addr2InstAux:
            self.addr2InstAux[addr] = isn.Instruction(
                addr, operand=operandName)
            self.addr2InstAux[addr].start = True
            self.addr2InstAux[addr].fallThrough = False

    def enter(self, inst, enterAddr: int) -> None:
        if enterAddr == FakeCalleeAddr:
            log.debug(f'Enter extern callee addr from {inst}')
            self.addAuxilaryInst(enterAddr, 'extrn_sym')
        elif enterAddr >= 0 and enterAddr < 256:
            log.debug(f'Enter software interrupt {enterAddr:x}')
            self.addAuxilaryInst(enterAddr, 'softirq_%X' % enterAddr)
        elif enterAddr not in self.addr2Inst:
            if inst.operand in ['call', 'syscall']:
                log.debug(f'Enter extern callee addr from {inst}')
                self.addAuxilaryInst(enterAddr, 'extrn_sym')
            else:
                log.error(f'Enter invalid address {enterAddr:x} from {inst}')
                self.addAuxilaryInst(InvalidAddr, 'invalid')
                self.addr2Inst[inst.address].branchTo = InvalidAddr
        else:
            log.debug(f'Enter instruction at {enterAddr:x} from {inst}')
            self.addr2Inst[enterAddr].start = True

    def branch(self, inst) -> None:
        """Conditional jump to another address or fall throught"""
        branchToAddr = inst.findAddrInInst()
        self.addr2Inst[inst.address].branchTo = branchToAddr
        log.debug(f'Found branch from {inst.address:x} to {branchToAddr:x}')
        self.enter(inst, branchToAddr)
        self.enter(inst, inst.address + inst.size)

    def call(self, inst) -> None:
        """Jump out and then back"""
        self.addr2Inst[inst.address].call = True
        # Likely NOT able to find callee's address (e.g. extern symbols)
        callAddr = inst.findAddrInInst()
        if callAddr != FakeCalleeAddr:
            log.debug(f'Found call from {inst.address:x} to {callAddr:x}')
        else:
            log.debug(f'Fake call from {inst.address:x} to FakeCalleeAddr')

        self.addr2Inst[inst.address].branchTo = callAddr
        self.enter(inst, callAddr)

    def jump(self, inst) -> None:
        """Unconditional jump to another address"""
        jumpAddr = inst.findAddrInInst()
        self.addr2Inst[inst.address].fallThrough = False
        self.addr2Inst[inst.address].branchTo = jumpAddr
        log.debug(f'Found jump from {inst.address:x} to {jumpAddr:x}')
        self.enter(inst, jumpAddr)
        self.enter(inst, inst.address + inst.size)

    def end(self, inst) -> None:
        """Stop fall throught"""
        self.addr2Inst[inst.address].fallThrough = False
        log.debug(f'Found end at {inst.address:x}')
        if inst.address + inst.size <= self.programEnd:
            self.enter(inst, inst.address + inst.size)

    def visitDefault(self, inst) -> None:
        pass

    def visitCalling(self, inst) -> None:
        self.call(inst)

    def visitConditionalJump(self, inst) -> None:
        self.branch(inst)

    def visitUnconditionalJump(self, inst) -> None:
        self.jump(inst)

    def visitEndHere(self, inst) -> None:
        self.end(inst)

    def getBlockAtAddr(self, addr: int) -> Block:
        if addr not in self.addr2Block:
            block = Block()
            block.startAddr = addr
            block.endAddr = addr
            self.addr2Block[addr] = block
            log.debug(f'Create new block starting/ending at {addr:x}')

        return self.addr2Block[addr]

    def connectBlocks(self) -> None:
        """
        Group instructions into blocks, and
        connected based on branch and fall through.
        """
        log.info('**** Create and connect blocks ****')
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
                    addr1, addr2 = currBlock.startAddr, nextBlock.startAddr
                    log.debug(f'Block {addr1:x} falls to {addr2:x}')

            if inst.branchTo is not None:
                block = self.getBlockAtAddr(inst.branchTo)
                if block.startAddr not in currBlock.edgeList:
                    currBlock.edgeList.append(block.startAddr)

                addr1, addr2 = currBlock.startAddr, block.startAddr
                log.debug(f'Block {addr1:x} branches to {addr2:x}')
                if inst.call is True:
                    if currBlock.startAddr not in block.edgeList:
                        block.edgeList.append(currBlock.startAddr)

                    log.debug(f'Block {addr2:x} return back to {addr1:x}')

            currBlock.instList.append(inst)
            currBlock.endAddr = max(currBlock.endAddr, inst.address)
            self.addr2Block[currBlock.startAddr] = currBlock
            currBlock = nextBlock

    def exportToNxGraph(self):
        """Assume block/node is represented by its startAddr"""
        log.info('**** Export to networkx-compatible graph ****')
        for (addr, block) in sorted(self.addr2Block.items()):
            self.cfg.add_node('%08X' % addr, block=block)

        for (addr, block) in self.addr2Block.items():
            for neighboor in block.edgeList:
                self.cfg.add_edge('%08X' % addr, '%08X' % neighboor)

    def drawCfg(self) -> None:
        log.info(f'**** Save graph plot to {self.filePrefix}.pdf ****')
        nx.draw(self.cfg, with_labels=True, font_weight='normal')
        plt.savefig('%s.pdf' % self.filePrefix, format='pdf')
        plt.clf()

    def printCfg(self):
        log.info('**** Print CFG ****')
        log.info(f'#nodes in cfg: {nx.number_of_nodes(self.cfg)}')
        log.info(f'#edges in cfg: {nx.number_of_edges(self.cfg)}')
        for (addr, block) in sorted(self.addr2Block.items()):
            log.info(f'block {addr:x} [{block.startAddr:x}, {block.endAddr:x}]')

        for (addr, block) in sorted(self.addr2Block.items()):
            for neighboor in block.edgeList:
                log.info(f'block {addr:x} -> {neighboor:x}')

        self.drawCfg()

    def saveProgram(self) -> None:
        progFile = open(self.filePrefix + '.prog', 'w')
        for (addr, inst) in self.program.items():
            progFile.write(addr + ' ' + inst + '\n')
        progFile.close()

    def clearTmpFiles(self) -> None:
        log.info('**** Remove temporary files ****')
        for ext in ['.text', '.prog']:
            os.remove(self.filePrefix + ext)


class AcfgBuilder(object):
    def __init__(self, binaryId: str, pathPrefix: str) -> None:
        super(AcfgBuilder, self).__init__()
        self.cfgBuilder = ControlFlowGraphBuilder(binaryId, pathPrefix)
        self.cfg: nx.DiGraph = None

    def extractBlockAttributes(self):
        """
        Extract features in each block.
        """
        log.info('Extract attributes from blocks')
        features = np.zeros((self.cfg.number_of_nodes(),
                             Block.getAttributesDim()))
        for (i, (node, attributes)) in enumerate(self.cfg.nodes(data=True)):
            block = attributes['block']
            log.debug(f'Process block {block.startAddr:x}')
            features[i, :] = block.getAttributes()

        return features

    def getAttributedCfg(self):
        self.cfg = self.cfgBuilder.getControlFlowGraph()
        blockAttrs = self.extractBlockAttributes()
        adjMatrix = nx.adjacency_matrix(self.cfg)
        return [blockAttrs, adjMatrix]
