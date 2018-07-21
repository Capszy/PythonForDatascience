import numpy as np
import pandas as pd
from pandas import Series,DataFrame

df1=pd.DataFrame(np.arange(36).reshape(6,6))
print(df1)
df2=pd.DataFrame(np.arange(15).reshape(5,3))
print(df2)
df3=pd.concat([df1,df2],axis=1)
print(df3)
print(df3.drop([0,2],axis=1))

series=Series(np.arange(6))
series.name="added_variable"
print(series)

variable_added=DataFrame.join(df1,series)
print(variable_added)

added_datatable=variable_added.append(variable_added,ignore_index=True)
print(added_datatable)

print(df1.sort_values(by=[5],ascending=[True]))