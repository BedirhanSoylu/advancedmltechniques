#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[2]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[3]:


train.info()


# In[4]:


train.Functional.value_counts()


# In[5]:


test.info()


# In[6]:


train.head()


# In[7]:


train.drop(["Condition1", "Condition2","Fence", "GarageYrBlt","BsmtHalfBath","BsmtFullBath","Heating","RoofMatl","LandSlope","Utilities","LandContour","Alley","Street","PoolQC"],axis = 1, inplace = True)
test.drop(["Condition1", "Condition2","Fence", "GarageYrBlt","BsmtHalfBath","BsmtFullBath","Heating","RoofMatl","LandSlope","Utilities","LandContour","Alley","Street","PoolQC"],axis = 1, inplace = True)


# In[8]:


train.info()


# In[9]:


train.drop_duplicates(inplace=True)
test.drop_duplicates(inplace=True)


# In[10]:


train.info()


# In[11]:


train.MiscFeature.value_counts()


# In[12]:


test.MiscFeature.value_counts()


# In[13]:


train.drop("MiscFeature", inplace = True,axis=1)
test.drop("MiscFeature", inplace = True,axis=1)


# In[14]:


train.info()


# In[15]:


plt.figure(figsize=[18,5])

plt.suptitle('Sold Date', fontsize = 16)

plt.subplot(1,2,1)
sns.histplot(data = train.MoSold, kde=True)
plt.title('Train')

plt.subplot(1,2,2)
sns.histplot(data = train.YrSold, kde=True)
plt.title('Test');


# In[16]:


train.YrSold = train.YrSold - train.YrSold.min()
test.YrSold = test.YrSold - test.YrSold.min()


# In[17]:


train.info()


# In[18]:


train.MoSold = train.MoSold.astype(np.float64)
test.MoSold = test.MoSold.astype(np.float64)


# In[19]:


train.MoSold = train.MoSold / 12
test.MoSold = test.MoSold / 12


# In[20]:


plt.figure(figsize=[18,5])

plt.suptitle('Sold Date', fontsize = 16)

plt.subplot(1,2,1)
sns.histplot(data = train.MoSold, kde=True)
plt.title('Month')

plt.subplot(1,2,2)
sns.histplot(data = train.YrSold, kde=True)
plt.title('Year');


# In[21]:


sell_age = train.MoSold + train.YrSold
sell_age2 = test.MoSold + test.YrSold


# In[22]:


train.insert(65,"sell_age",sell_age)
test.insert(65,"sell_age",sell_age2)


# In[23]:


plt.figure(figsize=[18,5])

plt.suptitle('Sellin Age', fontsize = 16)

plt.subplot(1,2,1)
sns.histplot(data = train.sell_age, kde=True)
plt.title('Train')

plt.subplot(1,2,2)
sns.histplot(data = test.sell_age, kde=True)
plt.title('Test');


# In[24]:


test.drop(["MoSold","YrSold"],axis = 1, inplace = True)
train.drop(["MoSold","YrSold"],axis = 1, inplace = True)


# In[25]:


train.info()


# In[26]:


test.info()


# In[27]:


train.SaleCondition.describe()


# In[28]:


train.SaleCondition.value_counts()


# In[29]:


test.SaleCondition.value_counts()


# In[30]:


plt.hist(train.SaleCondition);


# In[31]:


dummy5 = pd.get_dummies(train.SaleCondition)
train = pd.concat([train,dummy5],axis=1)
dummy6 = pd.get_dummies(test.SaleCondition)
test = pd.concat([test,dummy6],axis=1)


# In[32]:


dummy3 = pd.get_dummies(train.SaleType)
train = pd.concat([train,dummy3],axis=1)
dummy4 = pd.get_dummies(test.SaleType)
test = pd.concat([test,dummy4],axis=1)


# In[33]:


#get dummies uygulasak mı uygulamasak mı?
sns.histplot(train.SaleType)


# In[34]:


#burada belki yeni bir sütun açılıp saleprice ın üstüne direk ekle denilebilir
train.MiscVal.value_counts()


# In[35]:


train.info()


# In[36]:


train.PavedDrive.value_counts()


# In[37]:


dummy1 = pd.get_dummies(train.PavedDrive)
dummy2 = pd.get_dummies(test.PavedDrive)


# In[38]:


train = pd.concat([train,dummy1],axis = 1)
test = pd.concat([test,dummy2], axis = 1)


# In[39]:


train.info()


# In[40]:


train.drop("PavedDrive",axis = 1, inplace = True)
test.drop("PavedDrive", axis = 1, inplace = True)


# In[41]:


train.info()


# In[42]:


train.drop("SaleType", axis = 1, inplace = True)
test.drop("SaleType", axis = 1, inplace = True)


# In[43]:


train.info()


# In[44]:


train.drop("SaleCondition", axis = 1, inplace = True)
test.drop("SaleCondition", axis = 1, inplace = True)


# In[45]:


train.info()


# In[46]:


train.GarageFinish.value_counts()


# In[47]:


dummy7 = pd.get_dummies(train.GarageFinish)
train = pd.concat([train,dummy7],axis=1)
dummy8 = pd.get_dummies(test.GarageFinish)
test = pd.concat([test,dummy8],axis=1)
train.drop("GarageFinish", axis = 1, inplace = True)
test.drop("GarageFinish", axis = 1, inplace = True)


# In[48]:


train.info()


# In[49]:


dummy9 = pd.get_dummies(train.GarageType)
train = pd.concat([train,dummy9],axis=1)
dummy10 = pd.get_dummies(test.GarageType)
test = pd.concat([test,dummy10],axis=1)
train.drop("GarageType", axis = 1, inplace = True)
test.drop("GarageType", axis = 1, inplace = True)


# In[50]:


train.info()


# In[51]:


train.FireplaceQu.value_counts()


# In[52]:


train.FireplaceQu.loc[train.FireplaceQu == "Ex"] = 5
train.FireplaceQu.loc[train.FireplaceQu == "Gd"] = 4
train.FireplaceQu.loc[train.FireplaceQu == "TA"] = 3
train.FireplaceQu.loc[train.FireplaceQu == "Fa"] = 2
train.FireplaceQu.loc[train.FireplaceQu == "Po"] = 1
train.FireplaceQu.fillna(0 ,inplace = True)

test.FireplaceQu.loc[test.FireplaceQu == "Ex"] = 5
test.FireplaceQu.loc[test.FireplaceQu == "Gd"] = 4
test.FireplaceQu.loc[test.FireplaceQu == "TA"] = 3
test.FireplaceQu.loc[test.FireplaceQu == "Fa"] = 2
test.FireplaceQu.loc[test.FireplaceQu == "Po"] = 1
test.FireplaceQu.fillna(0 ,inplace = True)


# In[53]:


plt.figure(figsize=[18,5])

plt.suptitle('FirePlcQu', fontsize = 16)

plt.subplot(1,2,1)
sns.histplot(data = train['FireplaceQu'], kde=True)
plt.title('Train')

plt.subplot(1,2,2)
sns.histplot(data = test['FireplaceQu'], kde=True)
plt.title('Test');


# In[54]:


train.FireplaceQu.value_counts()


# In[55]:


train.info()


# In[56]:


train.Functional.value_counts()


# In[57]:


test.Functional.value_counts()


# In[58]:


train.head()


# In[59]:


#Label Coding Train Data
train.Functional.loc[train.Functional == "Typ"] = 5
train.Functional.loc[train.Functional == "Min1"] = 4
train.Functional.loc[train.Functional == "Min2"] = 3
train.Functional.loc[train.Functional == "Mod"] = 2
train.Functional.loc[train.Functional == "Maj1"] = 1
train.Functional.loc[train.Functional == "Maj2"] = 0
train.Functional.loc[train.Functional == "Sev"] = -1
train.Functional.loc[train.Functional == "Sal"] = -2
train.Functional = train.Functional.astype(np.int64)
#Label Coding Test Data
test.Functional.loc[test.Functional == "Typ"] = 5
test.Functional.loc[test.Functional == "Min1"] = 4
test.Functional.loc[test.Functional == "Min2"] = 3
test.Functional.loc[test.Functional == "Mod"] = 2
test.Functional.loc[test.Functional == "Maj1"] = 1
test.Functional.loc[test.Functional == "Maj2"] = 0
test.Functional.loc[test.Functional == "Sev"] = -1
test.Functional.loc[test.Functional == "Sal"] = -2
test.Functional.dropna(inplace = True)
test.Functional = test.Functional.astype(np.float64)


# In[60]:


train.info()


# In[61]:


test.info()


# In[62]:


test.Functional.value_counts()


# In[63]:


train.Electrical.value_counts()


# In[64]:


dummy11 = pd.get_dummies(train.Electrical)
train = pd.concat([train,dummy11],axis=1)
dummy12 = pd.get_dummies(test.Electrical)
test = pd.concat([test,dummy12],axis=1)
train.drop("Electrical", axis = 1, inplace = True)
test.drop("Electrical", axis = 1, inplace = True)


# In[65]:


train.info()


# In[66]:


train.CentralAir.value_counts()


# In[67]:


dummy13 = pd.get_dummies(train.CentralAir)
train = pd.concat([train,dummy13],axis=1)
dummy14 = pd.get_dummies(test.CentralAir)
test = pd.concat([test,dummy14],axis=1)
train.drop("CentralAir", axis = 1, inplace = True)
test.drop("CentralAir", axis = 1, inplace = True)


# In[68]:


train.info()


# In[69]:


#Label Coding Train Data
train.KitchenQual.loc[train.KitchenQual == "Ex"] = 5
train.KitchenQual.loc[train.KitchenQual == "Gd"] = 4
train.KitchenQual.loc[train.KitchenQual == "TA"] = 3
train.KitchenQual.loc[train.KitchenQual == "Fa"] = 2
train.KitchenQual.loc[train.KitchenQual == "Po"] = 1
#Label Coding Test Data
test.KitchenQual.loc[test.KitchenQual == "Ex"] = 5
test.KitchenQual.loc[test.KitchenQual == "Gd"] = 4
test.KitchenQual.loc[test.KitchenQual == "TA"] = 3
test.KitchenQual.loc[test.KitchenQual == "Fa"] = 2
test.KitchenQual.loc[test.KitchenQual == "Po"] = 1


# In[70]:


train.KitchenQual.value_counts()


# In[71]:


type(train.KitchenQual.loc[1])


# In[72]:


train.info()


# In[73]:


train.to_csv("asd.csv")
test.to_csv("fgh.csv")


# In[74]:


train2 = pd.read_csv("pcali.csv")
test2 = pd.read_csv("test_pcali.csv")


# In[75]:


train2.info()


# In[76]:


test2.info()


# In[77]:


train2.Foundation.value_counts()


# In[78]:


dummy = pd.get_dummies(train2.Foundation)
train2 = pd.concat([train2,dummy],axis=1)
dummy2 = pd.get_dummies(test2.Foundation)
test2 = pd.concat([test2,dummy2],axis=1)
train2.drop("Foundation", axis = 1, inplace = True)
test2.drop("Foundation", axis = 1, inplace = True)


# In[79]:


train2.info()


# In[83]:


def get_dummies(train, test, column):
    dummy = pd.get_dummies(train[column])
    train = pd.concat([train,dummy],axis=1)
    dummy2 = pd.get_dummies(test[column])
    test = pd.concat([test,dummy2],axis=1)
    train.drop(column, axis = 1, inplace = True)
    test.drop(column, axis = 1, inplace = True)


# In[90]:


dummy = pd.get_dummies(train2.MasVnrType)
train2 = pd.concat([train2,dummy],axis=1)
dummy2 = pd.get_dummies(test2.MasVnrType)
test2 = pd.concat([test2,dummy2],axis=1)
train2.drop("MasVnrType", axis = 1, inplace = True)
test2.drop("MasVnrType", axis = 1, inplace = True)


# In[91]:


train2.info()


# In[92]:


train2.Exterior2nd.value_counts()


# In[93]:


dummy = pd.get_dummies(train2.Exterior2nd)
train2 = pd.concat([train2,dummy],axis=1)
dummy2 = pd.get_dummies(test2.Exterior2nd)
test2 = pd.concat([test2,dummy2],axis=1)
train2.drop("Exterior2nd", axis = 1, inplace = True)
test2.drop("Exterior2nd", axis = 1, inplace = True)


# In[94]:


train2.info()


# In[95]:


train2.Exterior1st.value_counts()


# In[96]:


dummy = pd.get_dummies(train2.Exterior1st)
train2 = pd.concat([train2,dummy],axis=1)
dummy2 = pd.get_dummies(test2.Exterior1st)
test2 = pd.concat([test2,dummy2],axis=1)
train2.drop("Exterior1st", axis = 1, inplace = True)
test2.drop("Exterior1st", axis = 1, inplace = True)


# In[97]:


train2.info()


# In[99]:


train2.RoofStyle.value_counts()


# In[100]:


dummy = pd.get_dummies(train2.RoofStyle)
train2 = pd.concat([train2,dummy],axis=1)
dummy2 = pd.get_dummies(test2.RoofStyle)
test2 = pd.concat([test2,dummy2],axis=1)
train2.drop("RoofStyle", axis = 1, inplace = True)
test2.drop("RoofStyle", axis = 1, inplace = True)


# In[101]:


train2.HouseStyle.value_counts()


# In[102]:


dummy = pd.get_dummies(train2.HouseStyle)
train2 = pd.concat([train2,dummy],axis=1)
dummy2 = pd.get_dummies(test2.HouseStyle)
test2 = pd.concat([test2,dummy2],axis=1)
train2.drop("HouseStyle", axis = 1, inplace = True)
test2.drop("HouseStyle", axis = 1, inplace = True)


# In[103]:


train2.BldgType.value_counts()


# In[104]:


dummy = pd.get_dummies(train2.BldgType)
train2 = pd.concat([train2,dummy],axis=1)
dummy2 = pd.get_dummies(test2.BldgType)
test2 = pd.concat([test2,dummy2],axis=1)
train2.drop("BldgType", axis = 1, inplace = True)
test2.drop("BldgType", axis = 1, inplace = True)


# In[106]:


train2.columns


# In[107]:


dummy = pd.get_dummies(train2.LotShape)
train2 = pd.concat([train2,dummy],axis=1)
dummy2 = pd.get_dummies(test2.LotShape)
test2 = pd.concat([test2,dummy2],axis=1)
train2.drop("LotShape", axis = 1, inplace = True)
test2.drop("LotShape", axis = 1, inplace = True)


# In[108]:


train2.drop("Neighborhood", axis = 1, inplace = True)
test2.drop("Neighborhood", axis = 1, inplace = True)


# In[109]:


train2.drop("LotConfig", axis = 1, inplace = True)
test2.drop("LotConfig", axis = 1, inplace = True)


# In[110]:


test2.info()


# In[111]:


train2.info()


# In[112]:


train2.MSZoning.value_counts()


# In[113]:


dummy = pd.get_dummies(train2.MSZoning)
train2 = pd.concat([train2,dummy],axis=1)
dummy2 = pd.get_dummies(test2.MSZoning)
test2 = pd.concat([test2,dummy2],axis=1)
train2.drop("MSZoning", axis = 1, inplace = True)
test2.drop("MSZoning", axis = 1, inplace = True)


# In[114]:


train2.info()


# In[115]:


test2.info()


# In[116]:


train2.dropna(inplace = True)


# In[117]:


train2.info()


# In[139]:


plt.figure(figsize=[18,12])
sns.heatmap(train2.corr(), cmap="seismic", annot=True, vmin=-1, vmax=1, fmt='.1f');


# In[ ]:




