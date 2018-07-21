import numpy as np
from numpy.random import randn
import pandas as pd
from pandas import Series,DataFrame

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sb

rcParams['figure.figsize']=8,4
sb.set_style='whitegrid'

x=range(1,10)
y=[1,2,3,4,0.5,4,3,2,1]
#plt.bar(x,y)
#plt.xlabel('xaxislabel')
#plt.ylabel('yaxislabel')

z=[1,2,3,4,0.5]
veh_type=['bicycle','motorbike','car','van','stroller']
#plt.pie(z,labels=veh_type)
#plt.show()

#using cars dataset
#mpg=cars.mpg

fig=plt.figure()
ax=fig.add_axes([.1,.1,1,1])

#mpg.plot()
#ax.ser_xticks(range(32))

#ax.set_xticklabels(cars.car_names,rotation=60,fontsize='medium')
#ax.set_title('cars is the title')
#ax.set_xlabel('carnames')
#ax.set_ylabel('miles')

plt.pie(z)
plt.legend(veh_type,loc='best')
plt.show()

#using cars mpg by oop method
#ax.legend(loc='best')

#mpg.max()

#ax.set_ylim([0,45])
#ax.annotate('toyota corolla',xy=(19,33.9),axtext(21,35),arrowprops=dict(facecolor='black','shrink=0.05'))
