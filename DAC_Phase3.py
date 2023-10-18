#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
df = pd.read_csv('customer_churn.csv')
df.head(5)


# In[2]:


df.drop('customerID',axis='columns',inplace=True)
df.sample(5)


# In[3]:


df.dtypes


# In[4]:


df[pd.to_numeric(df.TotalCharges,errors='coerce').isnull()]


# In[5]:


df1 = df[df.TotalCharges!=' ']


# In[6]:


df1['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors='coerce')


# In[7]:


df1.dtypes


# In[8]:


df_tenure_no = df1[df1.Churn == 'No'].tenure
df_tenure_yes = df1[df1.Churn == 'Yes'].tenure
plt.hist([df_tenure_yes,df_tenure_no],color=['green','red'],label=['Churn=Yes','Churn=No'])
plt.legend()
plt.xlabel('Tenure')
plt.ylabel('Number of customers')
plt.title('Histogram based on Tenure of customers')


# In[9]:


df_total_charges_no = df1[df1.Churn == 'No'].TotalCharges
df_total_charges_yes = df1[df1.Churn == 'Yes'].TotalCharges
plt.hist([df_tenure_yes,df_tenure_no],color=['green','red'],label=['Churn=Yes','Churn=No'])
plt.legend()
plt.xlabel('Total Charges')
plt.ylabel('Number of customers')
plt.title('Histogram based on Total charges of customers')


# In[10]:


def print_unique_values(df):
    for cols in df.columns:
        print(cols,df[cols].unique())


# In[11]:


print_unique_values(df1)


# In[12]:


df1.replace('No phone service','No',inplace=True)


# In[13]:


df1.replace('No internet service','No',inplace=True)


# In[14]:


print_unique_values(df1)


# In[15]:


df2 = pd.get_dummies(data=df1,columns=['InternetService','Contract','PaymentMethod'])


# In[16]:


df2.shape


# In[17]:


print_unique_values(df2)


# In[18]:


df_true_false_cols = ['InternetService_DSL','InternetService_Fiber optic','InternetService_No','Contract_One year','Contract_Month-to-month','PaymentMethod_Bank transfer (automatic)','PaymentMethod_Credit card (automatic)','PaymentMethod_Electronic check','PaymentMethod_Mailed check','Contract_Two year']


# In[19]:


for cols in df_true_false_cols:
    df2[cols].replace({True:1,False:0},inplace=True)


# In[20]:


print_unique_values(df2)


# In[21]:


df_yes_no_cols = ['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']
for cols in df_yes_no_cols:
    df2[cols].replace({'Yes':1,'No':0},inplace=True)
print_unique_values(df2)


# In[22]:


df2['gender'].replace({'Female':1,'Male':0},inplace=True)
cols_to_scale = ['tenure','MonthlyCharges','TotalCharges']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])
print_unique_values(df2)


# In[23]:


df2.dtypes


# In[24]:


X = df2.drop('Churn',axis='columns')
y = df2['Churn']
print('Shape of X:',X.shape)
print('Shape of y:',y.shape)


# In[25]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=5)
print('Shape of X_train:',X_train.shape)
print('Shape of X_test:',X_test.shape)
print('Shape of y_train:',y_train.shape)
print('Shape of y_test:',y_test.shape)


# In[26]:


import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(20,input_shape=(26,),activation='relu'),
    keras.layers.Dense(1,activation='sigmoid')
])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=10)


# In[27]:


model.evaluate(X_test,y_test)


# In[28]:


from sklearn.metrics import classification_report,confusion_matrix
yp = model.predict(X_test)
yp


# In[29]:


yp.shape


# In[30]:


yp[:5]


# In[31]:


y_pred = []
for x in yp:
    if x > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)


# In[32]:


y_pred[:5]


# In[33]:


y_test[:5]


# In[34]:


print('Classification Report:',classification_report(y_test,y_pred))


# In[35]:


import seaborn as sns
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt='.2f')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')


# In[ ]:




