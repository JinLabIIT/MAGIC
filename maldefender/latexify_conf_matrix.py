#!/usr/bin/python3.7

import pandas as pd
import glog as log


def latexifyConfMatrix(scorePath, familyNames):
    df = pd.read_csv(scorePath, delim_whitespace=True,
                     names=familyNames)
    df.insert(0, 'Family', familyNames)
    log.info('Latex table for confusion matrix:\n' + df.to_latex(index=False))


if __name__ == '__main__':
    scorePaths = [
        '../../MSACFG/Nov16TrainValidScores/MSACFG_valid_confusion_matrix.txt',
        '../../YANACFG/Nov15TrainValidScores/YANACFG_valid_confusion_matrix.txt',
    ]
    families = [
        ['Ramnit', 'Lollipop', 'Kelihos_Ver3', 'Vundo', 'Simda', 'Tracur',
         'Kelihos\_Ver1', 'Obfuscator.ACY', 'Gatak'],
        ['Bagle', 'Benign', 'Bifrose', 'Hupigon', 'Koobface', 'Ldpinch',
         'Lmir', 'Rbot', 'Sdbot', 'Swizzor', 'Vundo', 'Zbot', 'Zlob']
    ]
    for (sp, name) in zip(scorePaths, families):
        latexifyConfMatrix(sp, name)
