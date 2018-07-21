import numpy as np
import pandas as pd
from matplot.lib.pyplot as plt
import seaborn as sb
from pylab import rcParams
import scipy
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import scale
rcParams['figure.figsize']=8,4
sb.set_style='whitegrid'

#cars dataset

mpg=cars.mpg
plt.plot(mpg)

cars[['mpg']].describe

mpg_matrix=mpg.reshape(-1,1)
scaled=preprocessing.MinMaxScaler(feature_range=(0,10))
scaled_mpg=scaled.fit_transform(mpg_matrix)
plt.plot(scaled_mpg)

standardized_mpg=scale(mpg,axis=0,with_mean=False,with_std=False)
plt.plot(standardized_mpg)

standardized_mpg=scale(mpg)
plt.plot(standardized_mpg)


