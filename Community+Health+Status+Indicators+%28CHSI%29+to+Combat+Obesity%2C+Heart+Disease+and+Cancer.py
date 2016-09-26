
# coding: utf-8

# In[96]:

get_ipython().magic(u'matplotlib inline')


# In[97]:

import pandas as pd


# In[98]:

import numpy as pd
from sklearn.linear_model import LinearRegression
get_ipython().magic(u'pylab inline')
import matplotlib.pyplot as plt


# In[99]:

import pandas as pd
X = pd.read_csv('/Users/annettechiu/Desktop/Health_indicators/RISKFACTORSANDACCESSTOCARE.csv')


# In[100]:

X.head()


# In[101]:

X = X[X['No_Exercise'] > -100]


# In[102]:

X = X[X['Diabetes'] > -100]


# In[103]:

X.hist('No_Exercise');


# In[104]:

X.hist('Diabetes');


# In[105]:

#remove any points with a missing y value
filtered_data =X[~np.isnan(X["No_Exercise"])]
filtered_data.head(3)
filtered_data.columns


# In[106]:

filtered_data[['No_Exercise','Disabled_Medicare']].corr()


# In[107]:

filtered_data[['No_Exercise','High_Blood_Pres']].corr()


# In[108]:

filtered_data[['No_Exercise','Elderly_Medicare']].corr()


# In[109]:

filtered_data[['No_Exercise','Obesity']].corr()


# In[110]:

filtered_data[['No_Exercise','Diabetes']].corr()


# In[111]:

filtered_data[['No_Exercise','Prim_Care_Phys_Rate']].corr()


# In[113]:

npMatrix = np.matrix(filtered_data)
No_Exercise, Diabetes = npMatrix[:,0], npMatrix[:,1]
mdl = LinearRegression().fit(No_Exercise,Diabetes) # either this or the next line
#mdl = LinearRegression().fit(filtered_data[['x']],filtered_data.y)
m = mdl.coef_[0]
b = mdl.intercept_
print "formula: y = {0}x + {1}".format(m, b) # following slope intercept form 


# In[117]:

plt.scatter(No_Exercise,Diabetes, color='blue')
plt.plot([0,100],[b,m*100+b],'r')
plt.title('Linear Regression', fontsize = 20)
plt.xlabel('No_Exercise', fontsize = 15)
plt.ylabel('Diabetes', fontsize = 15)


# In[ ]:



