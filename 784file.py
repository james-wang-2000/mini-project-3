from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, cache=True)
print(mnist.data.shape)
import numpy as np
#print(digits['target'][0])

import matplotlib as mpl
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(6,6)) # figure size in inches
fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)

from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest = train_test_split(mnist.data,mnist.target,random_state=0)

#for idx in range(16):
#    ex=mnist.data[idx]
#    plt.imshow(ex.reshape((28,28)),cmap=plt.cm.binary)
#    plt.show()

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=1000)
model.fit(Xtrain,ytrain)

from sklearn.metrics import accuracy_score
ypred=model.predict(Xtest)
print(accuracy_score(ypred,ytest))




