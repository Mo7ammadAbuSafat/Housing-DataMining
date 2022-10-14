#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
data= pd.read_csv("Housing.csv")
data.head()


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


data.dropna();
data.replace({'yes':1 , 'no':0}, inplace=True);
data.replace({'furnished':2 , 'semi-furnished':1,'unfurnished':0}, inplace=True);
data.head()


# In[ ]:


data=data.drop_duplicates()
data.info()


# In[ ]:


data.corr()


# In[ ]:


import numpy as np
import seaborn as see
import matplotlib.pyplot as plt

#to plot corr(لون فاتح يعني تشابه عالي)
see.heatmap(data.corr())


# In[ ]:


x= np.array([[1,2,3],[4.55,5,6]],dtype=np.int32)
x


# In[ ]:


y=np.zeros([4,5])
y1=np.ones([4,5])
y2=np.random.rand(4,5)

print(y)
print(y1)
print(y2)


# In[ ]:


output=np.array(data["price"],dtype=float).reshape(-1,1)
#حطينا الريشيب عشان ييضبط الدايمنشن 

input=np.array(data.iloc[:,1:13],dtype=float).reshape(-1,12)
print(output.shape)
print(input.shape)


# In[ ]:


plt.scatter(data["area"],data["price"])
plt.show()


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

g=MinMaxScaler();
outputnew= g.fit_transform(output);
inputnew= g.fit_transform(input);
print(inputnew)


# In[ ]:


mod1= LinearRegression();
x_train, x_test, y_train, y_test= train_test_split(inputnew,outputnew,train_size=0.7);
mod1.fit(x_train,y_train);
print(mod1.score(x_test,y_test));


# In[ ]:


output1=np.array(data["airconditioning"],dtype=float).reshape(-1,1)
input1=np.array(data.iloc[:,0:7],dtype=float).reshape(-1,7)

g1=MinMaxScaler();
output1new= g1.fit_transform(output1);
input1new= g1.fit_transform(input1);

mod2= LogisticRegression();

x1_train, x1_test, y1_train, y1_test= train_test_split(input1new,output1new,train_size=0.7);
mod2.fit(x1_train,y1_train);
print(mod2.score(x1_test,y1_test));


# In[ ]:


output2=np.array(data["price"],dtype=float).reshape(-1,1)
input2=np.array(data.iloc[:,1:13],dtype=float).reshape(-1,12)

g2=MinMaxScaler();
output2new= g2.fit_transform(output2);
input2new= g2.fit_transform(input2);

from sklearn.ensemble import RandomForestRegressor

mod3= RandomForestRegressor();

x2_train, x2_test, y2_train, y2_test= train_test_split(input2new,output2new,train_size=0.75);
mod3.fit(x2_train,y2_train);
print(mod3.score(x2_test,y2_test));


# In[ ]:


import pickle
model= pickle.dump(mod3,open("model","wb"));
model2=(pickle.load(open("model","rb")));

model2.score(x2_test,y2_test);
pred=model2.predict(x2_test);


# In[ ]:


from sklearn import metrics
contingecyMatrix = metrics.cluster.contingency_matrix(y2_test, pred)
print (contingecyMatrix)

