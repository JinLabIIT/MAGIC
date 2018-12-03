#!/usr/bin/python3.7

import pandas as pd
import glog as log

df = pd.read_csv('filesizes.csv')
log.info(df.shape[0])
sizeDf = pd.DataFrame(0, index=range(df.shape[0]), columns=['sizeInKBytes'])
for (index, x) in df.iterrows():
    if x['size'][-1] == 'M':
        sizeDf['sizeInKBytes'][index] = float(x['size'][:-1]) * 1000
    else:
        sizeDf['sizeInKBytes'][index] = float(x['size'][:-1])

log.info(f'Mean: {sizeDf["sizeInKBytes"].mean() / 1000.0} Mbytes')
log.info(f'Std: {sizeDf["sizeInKBytes"].std()/ 1000.0} Mbytes')
