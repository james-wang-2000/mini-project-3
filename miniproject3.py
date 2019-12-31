from sklearn.datasets import load_digits
digits = load_digits()
import numpy as np
#print(digits['target'][0])

import matplotlib as mpl
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(6,6)) # figure size in inches
fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)

from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest = train_test_split(digits.data,digits.target,random_state=0)

zero = []
one = []
two = []
three = []
four = []
five = []
six = []
seven = []
eight = []
nine = []

for i in range(len(digits.data)):
    if digits.target[i] == 0:
        zero.append(digits.data[i])
    if digits.target[i] == 1:
        one.append(digits.data[i])
    if digits.target[i] == 2:
        two.append(digits.data[i])
    if digits.target[i] == 3:
        three.append(digits.data[i])
    if digits.target[i] == 4:
        four.append(digits.data[i])
    if digits.target[i] == 5:
        five.append(digits.data[i])
    if digits.target[i] == 6:
        six.append(digits.data[i])
    if digits.target[i] == 7:
        seven.append(digits.data[i])
    if digits.target[i] == 8:
        eight.append(digits.data[i])
    if digits.target[i] == 9:
        nine.append(digits.data[i])

ex = np.array(two[50]).reshape((8,8))
def change_one(x):#1[50]
    for i in range(8):
        for j in range(8):
            if x[i][j] == 0:
                pass
            elif x[i][j] > 8:
                x[i][j] = x[i][j] - 2
            else:
                x[i][j] = x[i][j] + 2
def move_left(x):#9[5],0[2]
    for i in range(8):
        for j in range(7):
            x[i][j] = x[i][j+1]
def change_three(x): #three[50]
    for i in range(4):
        for j in range(7,0,-1):
            #if x[i][j] != 0:
                if j-1 > 0:
                    x[i][j] = x[i][j-1]
    for i in range(4,8):
        for j in range(7):
            x[i][j] = x[i][j+1]
def blurr(x): #four[50][20],six[20],
    for i in range(8):
        for j in range(7):
            x[i][j] = x[i][j]+10

print(ex)
change_one(ex)
print(ex)
a = np.array([0,0,0,0,6,0,0,0,
               0,0,2,6,6,0,0,0,
               0,0,6,2,6,0,0,0,
               0,0,0,0,6,0,0,0,
               0,0,0,0,6,0,0,0,
               0,0,0,0,6,0,0,0,
               0,0,6,6,6,6,6,0,
               0,0,0,0,0,0,0,0])

np.random.seed(42)
#ex = np.random.normal(ex,2)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=1000)
model.fit(Xtrain,ytrain)

print(model.predict(ex.reshape(1,64)))

plt.imshow(ex.reshape(8,8),cmap = plt.cm.binary)
plt.show()

