# %%

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import RandomOverSampler


# %%



cols = ["fLength" , "fWidth", "fSize", "fConc", "fConcl","fAsym", "fM3Long", "fM3Trans", "fAlpha","fDist", "class"]
print(len(cols))
df = pd.read_csv("magic04.data",names=cols)
df.head()


# %%

df['class'].unique()

# %%

df['class'] =  (df['class'] == "g").astype(int)
df


# %%

for label in cols[:-1]:
    df[df['class']==1][label].plot.hist(alpha=0.5)
    df[df['class']==0][label].plot.hist(alpha=0.5)
    

    # plt.hist(df[df['class']==1][label], color='blue', label="gamma" ,density=True)
    # plt.hist( df[df['class'] ==0][label] , color='red', label="hadron", density=True)
    plt.title(label)
    plt.ylabel("probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()



# %%

# train  ,  validation  , test    -->data sets
    
train, valid,test = np.split(df.sample(frac=1), [int(0.6 * len(df)) ,int(0.8 * len(df)) ])


print(len(train))
print(len(valid))
print(len(test))



# %%

def scale_dataset(dataframe, oversample=False):
    x = dataframe.iloc[:,:-1].values
    y = dataframe.iloc[:,-1].values
 
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    if oversample :
        ros = RandomOverSampler()
        x, y = ros.fit_resample(x,y)
    
    # horizontally stacking numpy arrays
    data = np.hstack((x, np.reshape(y, (len(y),1))))
    
    return data , x, y


# %%

train , x_train , y_train = scale_dataset(train,True)
valid , x_valid , y_valid = scale_dataset(valid,False)
test  , x_test  ,  y_test  = scale_dataset(test,False)



# %%

print(len(y_train))
print(len(x_train))
print(len(train))

# %%

print(sum(y_train==1))
print(sum(y_train==0))
# %%

# now our data is propery formatted
# now we will see different models

# first model is KNN 
# K nearest neighbours
# KNN algorithm --> memorizes the previous data
# also called memory based learning method

# While it’s not as popular as it once was, 
# it is still one of the first algorithms one learns in data science due to its simplicity and accuracy

# as a dataset grows, KNN becomes increasingly inefficient




# %%

# K nearest neighbours

# we will be not coding all stuffs of algo
# since it will have all bugs, and also may be slow
# algo is very simple
# only think which algo should be applied in a model

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# %%

knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(x_train, y_train)

# %%

# now , predicting
y_pred = knn_model.predict(x_test)

y_pred


# %%

y_test
# %%


print(classification_report(y_test,y_pred))
# %%


