import cv2
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import train_test_split
import os

path1 = "D:/Tristan/Python_Stuff/SimpleML/data/books/"
path2 = "D:/Tristan/Python_Stuff/SimpleML/data/candy/"
length = 64
width = 48

books = os.listdir(path1)
for i in books:
    print (i)

candy = os.listdir(path2)
for i in candy:
    print(i)

data = []
labels = []

for i in books:
    image = cv2.imread(path1+i,cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image,(length,width))
    data.append(image)
    labels.append(1)

for i in candy:
    image = cv2.imread(path2+i,cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image,(length,width))
    data.append(image)
    labels.append(0)

data = np.array(data)
labels = np.array(labels)

print (data.shape)
data = data.reshape(len(data),length*width)
print (data.shape)

x_train,x_test,y_train,y_test = train_test_split(data,labels,test_size=0.2)
x_train = x_train/255
x_test = x_test/255

classifier = neighbors.KNeighborsClassifier()
classifier.fit(x_train,y_train)
accuracy = classifier.score(x_test,y_test)

print(accuracy)