import numpy as np
import pandas as pd
from pandas import DataFrame,Series

missing=np.nan
series_obj=Series(['row 1','row 2',missing,'row 4','row 5','row 6',missing,'row 8'])

print(series_obj.isnull())
np.random.seed(25)
DF_Obj=DataFrame(np.random.randn(36).reshape(6,6))

DF_Obj.loc[3:5,0]=missing
DF_Obj.loc[1:4,5]=missing
filled_DF=DF_Obj.fillna(0)


filled_DF=DF_Obj.fillna({0:0.1,5:1.25})
#print(filled_DF)
fill_DF=DF_Obj.fillna(method='ffill')
#print(fill_DF)

DF_obj=DataFrame(np.random.randn(36).reshape(6,6))

DF_obj.loc[3:5,0]=missing
DF_obj.loc[1:4,5]=missing
print(DF_obj)
print(DF_obj.isnull().sum())
DF_no_NAN=DF_obj.dropna(axis=1)
print(DF_no_NAN)

print(DF_obj.dropna(how='all'))