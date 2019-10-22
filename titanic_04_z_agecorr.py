# age‚É‘Î‚·‚é‘ŠŠÖ‚ğŠm”F‚·‚éB

import numpy as nm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

d_train = pd.read_csv('train.csv')
d_test = pd.read_csv('test.csv')

d_train.corr()
plt.matshow(d_train.corr())