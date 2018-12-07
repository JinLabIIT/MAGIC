#!/usr/bin/python3.7

from matplotlib import pyplot as plt
import matplotlib
import pandas as pd
import glog as log

font = {'family': 'normal', 'size': 28}
matplotlib.rc('font', **font)
matplotlib.rcParams["figure.figsize"] = [14, 10]


def plotPrfScores(data, scorePath):
    df = pd.read_csv(scorePath, sep=',')
    log.info('Latex table for scores:\n' + df.to_latex(index=False))
    print(df)
    log.debug(df['Precision'].size)
    x = range(0, df['Precision'].size)

    plt.plot(x, df['Precision'], 'r', ls=':', marker='+',
             linewidth=2, markersize=12.0, markeredgewidth=3,
             clip_on=False, label='Precision')
    plt.plot(x, df['Recall'], 'g', ls='--', marker='x',
             linewidth=2, markersize=12.0, markeredgewidth=3,
             clip_on=False, label='Recall')
    plt.plot(x, df['F1'], 'b', ls='-.', marker='*',
             linewidth=2, markersize=12.0, markeredgewidth=3,
             clip_on=False, label='F1 Score')
    plt.xticks(x, df['Family'], rotation=45)
    plt.xlim((0, len(x) - 1))
    plt.ylabel('Performance Scores')
    minEach = list(df.min(numeric_only=True))
    yBottom = int(min(minEach) * 10) / 10.0
    plt.ylim((yBottom, 1.0))
    plt.grid(which='both', axis='both', ls='-.')
    plt.legend()
    leftDist = 0.10 if data == 'YanAcfg' else 0.14
    plt.subplots_adjust(left=leftDist, bottom=0.16, right=0.96, top=0.97)
    plt.savefig('%sScores.pdf' % data, format='pdf')
    log.info(f'Figure for {data} saved to {data}sScores.pdf')


if __name__ == '__main__':
    scorePaths = {
        # 'MsAcfg': '../../MSACFG/Nov16TrainValidScores/MSACFG_valid_pr_scores.csv',
        'YanAcfg': '../../YANACFG/Nov15TrainValidScores/YANACFG_valid_pr_scores.txt',
    }
    for (name, path) in scorePaths.items():
        plotPrfScores(name, path)
