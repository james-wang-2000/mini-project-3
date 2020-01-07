from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, cache=True)
#print(mnist.data.shape)
import numpy as np
print("data done")

import matplotlib as mpl
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10,1)) # figure size in inches
fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)

from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest = train_test_split(mnist.data,mnist.target,random_state=0)

from PIL import Image

def convert(image):
    data = list()
    bmp_image = Image.open(image)

    for j in range(28):
        for i in range(28):
            gray = (255 - bmp_image.getpixel((i,j))[0])
            data.append(gray)

    return data

#for idx in range(16):
#    ex=mnist.data[idx]
#    plt.imshow(ex.reshape((28,28)),cmap=plt.cm.binary)
#    plt.show()

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=1000)
model.fit(Xtrain,ytrain)

print('trained')

#from sklearn.metrics import accuracy_score
#ypred=model.predict(Xtest)
#print(accuracy_score(ypred,ytest))
for idx in range(10):
    file='test%d.png'%(idx+1)
    ex=convert(file)
    ex=np.array([int(color*0.9)+15 for color in ex])
    ax=fig.add_subplot(1,10,idx+1,xticks=[],yticks=[])
    ax.imshow(ex.reshape((28,28)),cmap=plt.cm.binary,interpolation='nearest')
    result=model.predict(ex.reshape((1,784)))[0]
    ax.text(0,12,str(result))
    
plt.show()




