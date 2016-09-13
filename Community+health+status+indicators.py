
# coding: utf-8

# In[49]:

get_ipython().magic(u'matplotlib inline')


# In[50]:

import pandas as pd
import numpy as pd
from sklearn.linear_model import LinearRegression
get_ipython().magic(u'pylab inline')
import matplotlib.pyplot as plt


# In[67]:

X = pd.read_csv('/Users/annettechiu/Desktop/Health_indicators/SUMMARY.csv')


# In[68]:

X.head()


# In[71]:

#remove any points with a missing y value
filtered_data =X[~np.isnan(X["ALE"])]
filtered_data.head(3)


# In[72]:

npMatrix = np.matrix(filtered_data)
ALE, US_Health_Status = npMatrix[:,0], npMatrix[:,1]
mdl = LinearRegression().fit(ALE,US_Health_Status) # either this or the next line
#mdl = LinearRegression().fit(filtered_data[['x']],filtered_data.y)
m = mdl.coef_[0]
b = mdl.intercept_
print "formula: y = {0}x + {1}".format(m, b) # following slope intercept form 


# In[74]:

plt.scatter(ALE,US_Health_Status, color='blue')
plt.plot([0,100],[b,m*100+b],'r')
plt.title('Linear Regression Example', fontsize = 20)
plt.xlabel('ALE', fontsize = 15)
plt.ylabel('US_Health_Status', fontsize = 15)


# In[ ]:



