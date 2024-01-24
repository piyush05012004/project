#!/usr/bin/env python
# coding: utf-8

# In[73]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings('ignore')


# In[9]:


df = pd.read_csv('spam.csv', encoding = "ISO-8859-1")
df


# In[10]:


df.sample(5)


# In[11]:


df.info()


# In[12]:


df.columns


# In[14]:


df.describe()


# In[15]:


df.isnull().sum()


# In[17]:


df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],inplace=True)


# In[18]:


df


# In[19]:


from sklearn.preprocessing import OrdinalEncoder


# In[20]:


od=OrdinalEncoder()
od.fit_transform(df[['v1','v2']])


# In[21]:


cat_col=df.select_dtypes(object).columns


# In[22]:


cat_col


# In[23]:


df[cat_col]=od.fit_transform(df[cat_col])


# In[24]:


df


# In[25]:


x=df[['v2']]
y=df['v1']


# In[26]:


plt.figure(figsize = (4,4))
sns.countplot(x="v1",data = df, hue ="v1", palette = "Pastel1")
plt.title("No.Of SMS")
plt.show()

print( '"The univariate graph, countplot indicates that the majority of SMS are in this dataset is Ham." ')

plt.figure(figsize=(8,6))
plt.scatter(x=df['v2'][0:20], y=df['v1'][0:20])
plt.title('SMS vs. Ham-Spam')
plt.xlabel('v2')
plt.ylabel('v1')
plt.xticks(rotation=90)
plt.yticks(rotation=90)
plt.grid()
plt.show()

print('"The Bivariate graph, Scatter plot represents that the distribution of sms msg and spam or non spam observation. which is required for analyze to develop a accurate model."')


# In[27]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=1)


# In[28]:


from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(xtrain,ytrain)
ypred=logreg.predict(xtest)


# In[29]:


from sklearn.metrics import classification_report, accuracy_score


# In[30]:


#Evaluating model

cr=classification_report(ytest,ypred)
print(cr)
ac= accuracy_score(ytest,ypred)
print("Accuracy Score: ",ac)


# In[31]:


logreg=LogisticRegression(solver='liblinear')
logreg.fit(xtrain,ytrain)
ypred=logreg.predict(xtest)


# In[32]:


cr=classification_report(ytest,ypred)
print(cr)


# In[33]:


logreg=LogisticRegression(solver='saga')
logreg.fit(xtrain,ytrain)
ypred=logreg.predict(xtest)


# In[34]:


cr=classification_report(ytest,ypred)
print(cr)


# In[35]:


logreg=LogisticRegression(solver='sag')
logreg.fit(xtrain,ytrain)
ypred=logreg.predict(xtest)


# In[36]:


cr=classification_report(ytest,ypred)
print(cr)


# In[37]:


from sklearn.tree import DecisionTreeClassifier


# In[38]:


dt= DecisionTreeClassifier()


# In[39]:


def mymodel(model):
    model.fit(xtrain,ytrain)
    ypred = model.predict(xtest)
    print(accuracy_score(ytest,ypred))
    print(classification_report(ytest,ypred))
    
    return model


# In[40]:


mymodel(dt)


# In[41]:


from sklearn import tree


# In[ ]:





# In[50]:


train = dt.score(xtrain,ytrain)
test = dt.score(xtest,ytest)
print(f"train score : {train} \n test score : {test}")


# In[51]:


for i in range(20,35):
    dt1 = DecisionTreeClassifier(max_depth = i)
    dt1.fit(xtrain,ytrain)
    ypred = dt1.predict(xtest)
    ac = accuracy_score(ytest,ypred)
    print(f"max_depth = {i} accuracy : {ac}")


# In[52]:


dt2 = DecisionTreeClassifier(max_depth = 24)
mymodel(dt2)


# In[53]:


train = dt2.score(xtrain,ytrain)
test = dt2.score(xtest,ytest)
print(f"train score : {train} \n test score : {test}")


# In[54]:


for i in range(90,110):
    dt3 = DecisionTreeClassifier(min_samples_split = i)
    dt3.fit(xtrain,ytrain)
    ypred = dt3.predict(xtest)
    ac = accuracy_score(ytest,ypred)
    print(f"min_sample_split = {i} accuracy : {ac}")


# In[55]:


dt4 = DecisionTreeClassifier(min_samples_split = 92)
mymodel(dt4)


# In[56]:


train = dt4.score(xtrain,ytrain)
test = dt4.score(xtest,ytest)
print(f"train score :{train} \n test score : {test}")


# In[57]:


for i in range(30,50):
    dt5 = DecisionTreeClassifier(min_samples_split = i)
    dt5.fit(xtrain,ytrain)
    ypred = dt5.predict(xtest)
    ac = accuracy_score(ytest,ypred)
    print(f"min_sample_split = {i} accuracy : {ac}")


# In[59]:


dt6 = DecisionTreeClassifier(min_samples_split = 44)
mymodel(dt6)


# In[61]:


train = dt6.score(xtrain,ytrain)
test = dt6.score(xtest,ytest)
print(f"train score :{train} \n test score : {test}")


# In[62]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5) # by default n_neighbors = 5
knn.fit(xtrain,ytrain)
ypred = knn.predict(xtest)


# In[63]:


#evaluate the model

from sklearn.metrics import accuracy_score
ac = accuracy_score(ytest,ypred)
print(ac)


# In[65]:


from sklearn.ensemble import RandomForestClassifier
rc = RandomForestClassifier()
rc.fit(xtrain,ytrain)
ypred = rc.predict(xtest)
print(classification_report(ytest,ypred))


# In[66]:


from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier()
ada.fit(xtrain,ytrain)
ypred = ada.predict(xtest)
print(classification_report(ytest,ypred))


# In[68]:


from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(xtrain,ytrain)
ypred = gbc.predict(xtest)
print(classification_report(ytest,ypred))


# In[93]:


pip install xgboost


# In[95]:


from sklearn.ensemble import BaggingClassifier
bg = BaggingClassifier(LogisticRegression())
bg.fit(xtrain,ytrain)
ypred = bg.predict(xtest)
print(classification_report(ytest,ypred))


# In[96]:


bg = BaggingClassifier(DecisionTreeClassifier())
bg.fit(xtrain,ytrain)
ypred = bg.predict(xtest)
print(classification_report(ytest,ypred))


# In[97]:


models =[]
models.append(("lr",LogisticRegression()))
models.append(("dt",DecisionTreeClassifier()))


# In[98]:


from sklearn.ensemble import VotingClassifier
vc = VotingClassifier(estimators = models)  # estimators ---> model name
vc.fit(xtrain,ytrain)
ypred = vc.predict(xtest)
print(classification_report(ytest,ypred))


# In[99]:


from sklearn.ensemble import VotingClassifier
vc = VotingClassifier(estimators = models,voting='soft')  # estimators ---> model name
vc.fit(xtrain,ytrain)
ypred = vc.predict(xtest)
print(classification_report(ytest,ypred))


# In[100]:


from sklearn.naive_bayes import GaussianNB


# In[101]:


nb_classifier = GaussianNB()
nb_classifier.fit(xtrain,ytrain)
ypred = nb_classifier.predict(xtest)


# In[102]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
accuracy = accuracy_score(ytest, ypred)
conf_matrix = confusion_matrix(ytest, ypred)
classification_rep = classification_report(ytest, ypred)


# In[103]:


print("Accuracy", accuracy)
print("\nConfusion Matrix: \n", conf_matrix)
print("\nClassification Report: \n",classification_rep)


# In[ ]:




