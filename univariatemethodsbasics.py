import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
import sklearn
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn import datasets

rcParams['figure.figsize']=5,4

#iris dataset

df.columns=['Sepal Length','Sepal Width','Petal Length','Petal Width']

df.boxplot(return_type='dict')
plt.plot()

Sepal_Width=X[:,1]
iris_outliers=(Sepal_Width>4)
df[iris_outliers]

pd.options.display.float_format='{:.1f}'.format
X_df=pd.DataFrame(x)
print X_df.describe()




