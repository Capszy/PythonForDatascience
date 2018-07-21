import numpy as np
import pandas as pd
from matplot.lib.pyplot as plt
import seaborn as sb
from pylab import rcParams
import scipy
from scipy.stats import pearsonr

rcParams['figure.figsize']=8,4
sb.set_style='whitegrid'

#cars data set
sb.pairplot(cars)

x=cars[['mpg','hp','qsec','wt']]
sb.pairplot(x)

mpg=cars['mpg']
hp=cars['hp']
qsec=cars['qsec']
wt=cars['wt']

pearsonr_coefficient,p_value=pearsonr(mpg,hp)
print 'PearsonR Correlation Coefficient %0.3f'%(pearsonr_coefficient)

corr=x.corr()#corelation values

sb.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values)
#dark is strong degree of corelation