#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install seaborn')
get_ipython().system('pip install scikit-learn')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split


# In[2]:


url="C:\\Users\\Asus\\Desktop\\medical_insurance_dataset.csv"
df=pd.read_csv(url,header=None)


# In[3]:


df


# In[4]:


df.head(10)


# In[5]:


list=["age","gender","bmi","no_of_children","smoker","region","charger"]
df.columns=list
df


# In[6]:


df.replace("?",np.nan,inplace=True)


# In[7]:


df.isna().sum()


# In[8]:


df.info()


# In[11]:


mean_age=df["age"].astype('float').mean()
df["age"].replace(np.nan,mean_age,inplace=True)
mean_bmi=df["bmi"].astype('float').mean()
df["bmi"].replace(np.nan,mean_bmi,inplace=True)
most_smoker=df["smoker"].value_counts().idxmax()
df["smoker"].replace(np.nan,most_smoker,inplace=True)
df["age"]=df["age"].astype("int")
df["smoker"]=df["smoker"].astype("int")
df["charger"]=np.round(df["charger"],2)
df


# In[12]:


df.info()


# In[13]:


sns.regplot(x="charger",y="bmi",data=df,line_kws={"color": "red"})


# In[14]:


sns.boxplot(x="smoker",y="charger",data=df)


# In[15]:


df.corr()


# In[16]:


lm=LinearRegression()
x=df[["smoker"]]
y=df["charger"]
lm.fit(x,y)
lm.score(x,y)


# In[26]:


z=df[["age","gender","bmi","no_of_children","smoker","region"]]
y=df["charger"]
lm.fit(z,y)
lm.score(z,y)


# In[28]:


Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model', LinearRegression())]
pipe=Pipeline(Input)
z= z.astype(float)
pipe.fit(z,Y)
ypipe=pipe.predict(z)
print(r2_score(Y,ypipe))


# In[35]:


input=[("scaler",StandardScaler()),("polinom",PolynomialFeatures()),("model",LinearRegression())]
pipe=Pipeline(input)
z=z.astype(float)
pipe.fit(z,y)
yhat=pipe.predict(z)
print(r2_score(y,yhat))


# In[39]:


x_train,x_test,y_train,y_test=train_test_split(z,y,random_state=1,test_size=0.2)


# In[41]:


RR=Ridge(alpha=0)
RR.fit(x_train,y_train)
yhat=RR.predict(x_test)
r2_score(y_test,yhat)


# In[43]:


pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.fit_transform(x_test)
RR.fit(x_train_pr, y_train)
y_hat =RR.predict(x_test_pr)
print(r2_score(y_test,y_hat))


# In[44]:


print("FİNAL")


# In[ ]:




