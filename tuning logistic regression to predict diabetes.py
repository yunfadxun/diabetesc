#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df = pd.read_csv("diabetes.csv")
df.info()
df


# In[3]:


for i in df.columns[0:7]:
    plt.figure()
    sns.distplot(df[i])
    plt.figure(f"distribution plot of {i}")


# In[4]:


x= df.drop("Outcome",axis=1)
y= df["Outcome"].values.reshape(-1,1)


# In[5]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,random_state=44, train_size=0.7)


# In[6]:


from imblearn.over_sampling import SMOTE
sm=SMOTE()
x_train,y_train= sm.fit_resample(x_train,y_train)


# In[7]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train= scaler.fit_transform(x_train)
x_test= scaler.fit_transform(x_test)


# In[8]:


from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
y_base= logreg.predict(x_test)


# In[9]:


from sklearn.metrics import confusion_matrix, classification_report, recall_score
print(confusion_matrix(y_test,y_base))
print(classification_report(y_test,y_base))
print(recall_score(y_test,y_base))


# In[13]:


#tuning penalty
penalty1=["none", "l2"]
recall_penalty=[]
for i in penalty1:
    logreg_1= LogisticRegression(penalty=i)
    logreg_1.fit(x_train,y_train)
    y_penalty= logreg_1.predict(x_test)
    recall1=recall_score(y_test,y_penalty)
    recall_penalty.append(recall1)
    
d= pd.DataFrame({"penalty":pd.Series(penalty1),
                "specificity": pd.Series(recall_penalty)})
                 
plt.figure()
sns.lineplot(x="penalty",y="specificity", data=d)
plt.show()


# In[14]:


#tuning solver
solver=["lbfgs","newton-cg","sag","saga"]
solver_recall=[]

for i in solver:
    logreg_2=LogisticRegression(penalty="none",solver=i)
    logreg_2.fit(x_train,y_train)
    y_solver= logreg_2.predict(x_test)
    recall2=recall_score(y_test,y_solver)
    solver_recall.append(recall2)
d2= pd.DataFrame({"solver":pd.Series(solver),
                 "recall":pd.Series(solver_recall)})
plt.figure()
sns.lineplot("solver","recall",data=d2)


# In[15]:


max_iter=[5, 10,100,200,500,1000,10000]
iteration_recall=[]

for i in max_iter:
    logreg_3=LogisticRegression(penalty="none",solver="sag", max_iter=i)
    logreg_3.fit(x_train,y_train)
    y_iter=logreg_3.predict(x_test)
    recall3=recall_score(y_test,y_iter)
    iteration_recall.append(recall3)
d3= pd.DataFrame({"iteration":pd.Series(max_iter),
                 "recall":pd.Series(iteration_recall)})

plt.figure()
sns.lineplot("iteration","recall",data=d3)


# In[16]:


C=[1,2.5,5,10,12.5,15,17.5,20]
C_recall=[]

for i in C:
    logreg_4= LogisticRegression(penalty="none",solver="sag",max_iter=10, C=i)
    logreg_4.fit(x_train,y_train)
    y_C= logreg_4.predict(x_test)
    recall4= recall_score(y_test,y_C)
    C_recall.append(recall4)
d4= pd.DataFrame({"c":pd.Series(C),
                 "recall":pd.Series(C_recall)})
plt.figure()
sns.lineplot("c","recall",data=d4, marker="o")


# In[17]:


tol=[0.0001,0.001,0.01,1,10,100,1000]
tol_recall=[]

for i in tol:
    logreg_5= LogisticRegression(penalty="none",solver="sag",max_iter=10,C=1,tol=i)
    logreg_5.fit(x_train,y_train)
    y_tol=logreg_5.predict(x_test)
    recall5=recall_score(y_test,y_tol)
    tol_recall.append(recall5)
d5= pd.DataFrame({"tol":pd.Series(tol),
                 "recall":pd.Series(tol_recall)})

plt.figure()
sns.lineplot("tol","recall",data=d5,marker="o")


# In[18]:


intercept_scaling=[True,False]
intercept_recall=[]

for i in intercept_scaling:
    logreg_6= LogisticRegression(penalty="none",solver="sag",max_iter=10,C=1,tol= 0.0001, intercept_scaling=i)
    logreg_6.fit(x_train,y_train)
    y_intercept=logreg_6.predict(x_test)
    recall6=recall_score(y_test,y_intercept)
    intercept_recall.append(recall6)
d6= pd.DataFrame({"scaling":pd.Series(intercept_scaling),
                 "recall":pd.Series(intercept_recall)})
plt.figure()
sns.lineplot("scaling","recall",data=d6)


# In[19]:


multiclass=["ovr","multinomial","auto"]
multiclass_recall=[]

for i in multiclass:
    logreg_7= LogisticRegression(penalty="none",solver="sag",max_iter=10,C=1,tol= 0.0001, intercept_scaling=True, multi_class=i)
    logreg_7.fit(x_train,y_train)
    y_multiclass=logreg_7.predict(x_test)
    recall7=recall_score(y_test,y_multiclass)
    multiclass_recall.append(recall7)
d7= pd.DataFrame({"multiclass":pd.Series(multiclass),
                 "recall":pd.Series(multiclass_recall)})
plt.figure()
sns.lineplot("multiclass","recall",data=d7)


# In[20]:


verbose=[0.0001,0.001,0.01,0.1,1,10,100]
verbose_recall=[]

for i in verbose:
    logreg_8=LogisticRegression(penalty="none",solver="sag",max_iter=10,C=1,tol= 0.0001, 
                                intercept_scaling=True, multi_class="ovr",verbose=i)
    logreg_8.fit(x_train,y_train)
    y_verbose= logreg_8.predict(x_test)
    recall8=recall_score(y_test,y_verbose)
    verbose_recall.append(recall8)
d8= pd.DataFrame({"verbose":pd.Series(verbose),
                 "recall":pd.Series(verbose_recall)})
plt.plot()
sns.lineplot("verbose","recall",data=d8, marker="o")


# In[21]:


warmstart=[True,False]
warmstart_recall=[]

for i in warmstart:
    logreg_9=LogisticRegression(penalty="none",solver="sag",max_iter=10,C=1,tol= 0.0001, 
                                intercept_scaling=True, multi_class="ovr",verbose=0.001,warm_start=i)
    logreg_9.fit(x_train,y_train)
    y_warmstart= logreg_9.predict(x_test)
    recall9= recall_score(y_test,y_warmstart)
    warmstart_recall.append(recall9)
d9= pd.DataFrame({"warmstart":pd.Series(warmstart),
                 "recall":pd.Series(warmstart_recall)})
plt.figure()
sns.lineplot("warmstart","recall",data=d9)


# In[22]:


logreg_final=LogisticRegression(penalty="none",solver="sag",max_iter=10,C=1,tol= 0.0001, 
                                intercept_scaling=True, multi_class="ovr",verbose=0.001,warm_start=True)
logreg_final.fit(x_train,y_train)
y_final = logreg_final.predict(x_test)
print(classification_report(y_test,y_final))
print(classification_report(y_test,y_base))
print(confusion_matrix(y_test,y_final))
print(confusion_matrix(y_test,y_base))


# In[23]:


from sklearn.model_selection import RandomizedSearchCV
params=[{"penalty":["elasticnet","none","l1","l2"],
        "solver":["lbfgs","newton-cg","sag","saga"],
        "max_iter":[5, 10,100,200,500,1000,10000],
        "warm_start":[True,False],
        "verbose":[0.0001,0.001,0.01,1,10,100,1000],
        "C":[0.0001,0.001,0.01,1,10,100,1000],
        "multi_class":["ovr","multinomial","auto"],
        "intercept_scaling":[True,False],
        "tol":[0.0001,0.001,0.01,1,10,100,1000]}]


# In[26]:


auto_tuned=RandomizedSearchCV(LogisticRegression(), params, cv=5, n_jobs=-1)
auto_tuned.fit(x_train,y_train)
auto_tuned.best_params_


# In[29]:


automodel= auto_tuned.best_estimator_
y_automodel= automodel.predict(x_test)
print(confusion_matrix(y_test,y_automodel))


# In[ ]:




