from sklearn.datasets import load_digits
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

# classify images
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
        
# example image   
ex = np.array(two[50]).reshape((8,8))

# some image processing functions ('#' is the example misrecognized image)
def dark_down(x):    #1[50]
    """ up lighter(-2), down darker(+2)"""
    for i in range(8):
        for j in range(8):
            if x[i][j] == 0:
                pass
            elif x[i][j] > 8:
                x[i][j] = x[i][j] - 2
            else:
                x[i][j] = x[i][j] + 2
                
def move_left(x):    #9[5],0[2]
    """ whole picture move left 1 """
    for i in range(8):
        for j in range(7):
            x[i][j] = x[i][j+1]
            
def twist(x):    #3[50]
    """ up move right 1, down move left 1 """
    for i in range(4):
        for j in range(7,0,-1):
            #if x[i][j] != 0:
                if j-1 > 0:
                    x[i][j] = x[i][j-1]
    for i in range(4,8):
        for j in range(7):
            x[i][j] = x[i][j+1]
            
def blurr(x):   #4[50][20],6[20]
    """ all darker (+10) """
    for i in range(8):
        for j in range(7):
            x[i][j] = x[i][j]+10


print(ex)
dark_down(ex)
print(ex)
np.random.seed(42)
#ex = np.random.normal(ex,2)  

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=1000)
model.fit(Xtrain,ytrain)

print(model.predict(ex.reshape(1,64)))

plt.imshow(ex.reshape(8,8),cmap = plt.cm.binary)
plt.show()
