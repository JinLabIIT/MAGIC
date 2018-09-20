#!/usr/bin/python3
import re
import glog as log


def extractAsmInsts(binaryId):
    lineNum = 1
    bytePattern = re.compile('[A-Z0-9][A-Z0-9]')
    imcompleteByte = re.compile('\?\?')
    fileInput = open(binaryId + '.asm', 'rb')
    fileOutput = open(binaryId + '.txt', 'w')
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
            log.debug("Ignore imcomplete code at line %d: %s" % (lineNum, " ".join(decodedElems)))
            continue

        startIdx = 0
        while startIdx < len(decodedElems) and bytePattern.match(decodedElems[startIdx]):
            startIdx += 1

        if startIdx == len(decodedElems):
            log.debug("No instructions at line %d: %s" % (lineNum, elems))
            continue

        endIdx = decodedElems.index(';') if ';' in decodedElems else len(decodedElems)
        instElems = [addr] + decodedElems[startIdx: endIdx]
        if len(instElems) > 1:
            log.debug("Processed line %d: '%s' => '%s'" % (lineNum, " ".join(decodedElems), " ".join(instElems)))
            fileOutput.write(" ".join(instElems) + '\n')

        lineNum += 1

    fileInput.close()
    fileOutput.close()


if __name__ == '__main__':
    log.setLevel("INFO")
    binaryIds = ['0A32eTdBKayjCWhZqDOQ', '01azqd4InC7m9JpocGv5', '0ACDbR5M3ZhBJajygTuf']

    for bId in binaryIds:
        log.info('Processing ' + bId + '.asm')
        extractAsmInsts(bId)
