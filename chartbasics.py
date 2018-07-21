import numpy as np
from numpy.random import randn
import pandas as pd
from pandas import Series,DataFrame

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sb

rcParams['figure.figsize']=5,4
sb.set_style('whitegrid')

x=range(1,10)
y=[1,2,3,4,0,4,3,2,1]

plt.plot(x,y)

#address='C:/Users/give the loaction of the dataset/file.csv'
#cars=pd.read_csv(address)
#cars.column=['car_names','mpg','cyl'.....]
#mpg=cars['mpg']
#mpg.plot()
#df=cars[['cyl','wt','mpg']]
#df.plot()

#plt.bar(x,y)
#mpg.plot(kind='bar') #kind='barh'

x=[1,2,3,4,0.5]
#plt.pie(x)

#plt.savefig('pie.jpeg')
#plt.show()