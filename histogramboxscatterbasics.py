import numpy as np
import pandas as pd
from pandas import Series,DataFrame
from pandas.tools.plotting import scatter_matrix

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sb

rcParams['figure.figsize']=8,4
sb.set_style='whitegrid'

#cars dataset
mpg.plot(kind='hist')

plt.hist(mpg)

sb.distplot(mpg)

cars.plot(kind='scatter',x='hp',y='mpg',c=['darkgray'],s=150)

sb.regplot(x='hp',y='mpg',data=cars,scatter=true)

sb.pairplot(cars)#scatterplot matrix

cars_df=pd.DataFrame((cars.ix[:,(1,3,4,6)].values),colums=['mpg','disp','hp','wt'])
cars_target=cars.ix[:,9].values
target_names=[0,1]
cars_df['group']=pd.series(cars_target,dtype='category')
sb.pairplot(cars_df,hue='group',palette='hls')

cars.boxplot(column='mpg',by='am')
cars.boxplot(column='wt',by='am')

sb.boxplot(x='am',y='mpg',data=cars,palette='hls')

