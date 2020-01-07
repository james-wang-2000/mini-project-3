from sklearn.datasets import load_digits
from masaic import convert
digits = load_digits()
import numpy as np

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

ex = np.array(three[60]).reshape((8,8))

ex = np.array(convert(r"C:\Users\Admin\Desktop\Python\計程實驗\mini_project_3\9.png"))

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=1000)
model.fit(Xtrain,ytrain)

print(model.predict(ex.reshape(1,64)))

plt.imshow(ex.reshape(8,8),cmap = plt.cm.binary)
plt.show()