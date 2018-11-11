def neighborsFromAdjacentMatrix(spAdjacentMat):
    spAdjacent = sp.sparse.find(spAdjacentMat)
    indices = {}
    for i in range(len(spAdjacent[0])):
        if spAdjacent[0][i] not in indices:
            indices[spAdjacent[0][i]] = []

        indices[spAdjacent[0][i]].append(spAdjacent[1][i])

    return indices


def list2Str(l1, l2):
    """
    Merge two list, then return space seperated string format.
    """
    return " ".join([str(x) for x in (list(l1) + list(l2))])


def matchConstant(line):
    """Parse the numeric/string constants in an operand"""
    operand = line.strip('\n\r\t ')
    numericCnts = 0
    stringCnts = 0
    """
    Whole operand is a num OR leading num in expression.
    E.g. "0ABh", "589h", "0ABh" in "0ABh*589h"
    """
    wholeNum = r'^([1-9][0-9A-F]*|0[A-F][0-9A-F]*)h?.*'
    pattern = re.compile(wholeNum)
    if pattern.match(operand):
        numericCnts += 1
        log.debug('[MatchConst] Match whole number in %s' % operand)
        # numerics.append('%s:WHOLE/LEAD' % operand)
    """Number inside expression, exclude the leading one."""
    numInExpr = r'([+*/:]|-)([1-9][0-9A-F]*|0[A-F][0-9A-F]*)h?'
    pattern = re.compile(numInExpr)
    match = pattern.findall(operand)
    if len(match) > 0:
        numericCnts += 1
        log.debug('[MatchConst] Match in-expression number in %s' % operand)
        # numerics.append('%s:%d' % (operand, len(match)))
    """Const string inside double/single quote"""
    strRe = r'["\'][^"]+["\']'
    pattern = re.compile(strRe)
    match = pattern.findall(operand)
    if len(match) > 0:
        stringCnts += 1
        log.debug('[MatchConst] Match str const in %s' % operand)
        # strings.append('%s:%d' % (operand, len(match)))

    return [numericCnts, stringCnts]
