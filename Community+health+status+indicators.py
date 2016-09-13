
# coding: utf-8

# In[2]:

get_ipython().magic(u'matplotlib inline')


# In[ ]:

import pandas as pd
import numpy as pd
from sklearn.linear_model import LinearRegression
get_ipython().magic(u'pylab inline')
import matplotlib.pyplot as plt


# In[71]:

X = pd.read_csv('/Users/annettechiu/Desktop/Health_indicators/SUMMARY.csv')


# In[72]:

X.head()


# In[73]:

X = X[X['ALE'] > -100]


# In[74]:

X.hist('ALE');


# In[75]:

#remove any points with a missing y value
filtered_data =X[~np.isnan(X["ALE"])]
filtered_data.head(3)
filtered_data.columns


# In[81]:

filtered_data[['CI_Max_Health_Status','Strata_ID_Number']].corr()


# In[78]:

npMatrix = np.matrix(filtered_data)
ALE, US_Health_Status = npMatrix[:,0], npMatrix[:,1]
mdl = LinearRegression().fit(ALE,US_Health_Status) # either this or the next line
#mdl = LinearRegression().fit(filtered_data[['x']],filtered_data.y)
m = mdl.coef_[0]
b = mdl.intercept_
print "formula: y = {0}x + {1}".format(m, b) # following slope intercept form 


# In[79]:

plt.scatter(ALE,US_Health_Status, color='blue')
plt.plot([0,100],[b,m*100+b],'r')
plt.title('Linear Regression', fontsize = 20)
plt.xlabel('ALE', fontsize = 15)
plt.ylabel('US_Health_Status', fontsize = 15)


# In[80]:

plt.scatter(ALE,US_Health_Status, color='blue')
plt.plot([0,100],[b,m*100+b],'r')
plt.title('Linear Regression', fontsize = 20)
plt.xlabel('ALE', fontsize = 15)
plt.ylabel('US_Health_Status', fontsize = 15)


# In[49]:

import pandas as pd


# In[51]:



