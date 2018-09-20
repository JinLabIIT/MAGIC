#!/usr/bin/python3
import re
import glog as log
import networkx as nx
from typing import List, Dict
from instructions import Instruction

class Block(object):
    """Block of control flow graph."""

    def __init__(self):
        super(Block, self).__init__()
        self.startAddr = -1
        self.endAddr = -1
        self.instList: List[Instruction] = []
        self.edgeList: List[Block] = []


class ControlFlowGraphBuilder(object):
    """For building a control flow graph from a program"""

    def __init__(self, binaryId: str) -> None:
        super(ControlFlowGraphBuilder, self).__init__()
        self.cfg = nx.Graph()
        self.binaryId: str = binaryId
        self.addr2Inst: Dict[str, Instruction] = {}
        self.program: Dict[str, str] = {}  # Addr to raw string instruction

    def build(self) -> None:
        self.extractTextSeg()
        self.createProgram()
        self.discoverEntries()
        self.connectBlocks()

    def extractTextSeg(self) -> None:
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

    def isStartProcDef(self, sameAddrInsts: List[str]) -> bool:
        for inst in sameAddrInsts:
            if inst.find('proc near') != -1:
                return True

        return False

    def isEndProcDef(self, sameAddrInsts: List[str]) -> bool:
        for inst in sameAddrInsts:
            if inst.find('endp') != -1:
                return True

        return False

    def isDataDef(self, sameAddrInsts: List[str]) -> bool:
        for inst in sameAddrInsts:
            if inst.find('dw') != -1:
                return True
            if inst.find('dd') != -1:
                return True
            if inst.find('db') != -1:
                return True

        return False

    def isLabel(self, sameAddrInsts: List[str]) -> bool:
        for inst in sameAddrInsts:
            if inst.endswith(':'):
                return True

        return False

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
        Case 4: 1+ insts dd, db, dw
        Case 5: location label followed by regular inst
        Case 6: Just 1 regular inst
        """
        if self.isHeaderInfo(sameAddrInsts):
            self.program[addr] = sameAddrInsts[-1]
        elif self.isStartProcDef(sameAddrInsts):
            self.program[addr] = sameAddrInsts[-1]
        elif self.isEndProcDef(sameAddrInsts):
            self.program[addr] = sameAddrInsts[0]
        elif self.isDataDef(sameAddrInsts):
            self.program[addr] = 'dd var_name'
        elif self.isLabel(sameAddrInsts):
            self.program[addr] = sameAddrInsts[-1]
        elif len(sameAddrInsts) == 1:
            self.program[addr] = sameAddrInsts[0]
        else:
            log.error('Unable to aggregate instructions for addr %s:' % addr)
            for inst in sameAddrInsts:
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
        pass

    def connectBlocks(self) -> None:
        pass

    def saveProgram(self) -> None:
        progFile = open(self.binaryId + '.prog', 'w')
        for (addr, inst) in self.program.items():
            progFile.write(addr + ' ' + inst + '\n')
        progFile.close()


if __name__ == '__main__':
    log.setLevel("INFO")
    binaryIds = ['0A32eTdBKayjCWhZqDOQ']
    # , '01azqd4InC7m9JpocGv5', '0ACDbR5M3ZhBJajygTuf']

    for bId in binaryIds:
        log.info('Processing ' + bId + '.asm')
        cfgBuilder = ControlFlowGraphBuilder(bId)
        cfgBuilder.build()
