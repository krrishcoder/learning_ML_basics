# %%

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 



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
    



    