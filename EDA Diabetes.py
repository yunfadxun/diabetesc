#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df =pd.read_csv("diabetes.csv")
df.info()
df.shape
df.describe().T


# In[3]:


print(df.isnull().sum())
print()
print(df.nunique())


# In[7]:


for i in df.columns[0:8]:
    plt.figure()
    skewness= round(df[i].skew(),2)
    sns.displot(df[i],kde=True)
    plt.title(f"distribution of {i}, skew= {skewness}")


# In[10]:


for i in df.columns[0:8]:
    plt.figure()
    Q3= df[i].quantile(0.75)
    Q1= df[i].quantile(0.25)
    IQR= Q3=Q1
    sns.boxplot(df[i])
    plt.title(f"distribution of {i}, IQR= {IQR}")


# In[12]:


plt.figure()
sns.pairplot(df, hue="Outcome")
plt.suptitle("pairplot of diabetes")
plt.show()


# In[15]:


for i in df.columns[0:8]:
    plt.figure()
    sns.scatterplot(x= df[i],y=df["Outcome"], hue="Outcome", data=df,palette="husl")


# In[38]:


from scipy.stats import f_oneway

for i in df.columns[0:8]:
    group1= df[df["Outcome"]==0][i]
    group2= df[df["Outcome"]==1][i]
    result= f_oneway(group1, group2)
    formatted_p="{:3f}".format(result.pvalue)
    print(f"{i}, F Score: {result.statistic:.3f}, P-Value:{formatted_p}")


# In[42]:


sns.heatmap(df.corr(),annot=True)
#I'd like to odrop the blood pressure and diabetespedigreefunction


# In[ ]:




