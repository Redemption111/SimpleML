import cv2
import numpy as np
from sklearn import preprocessing, neighbors
import sklearn.model_selection
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

databooks = "D:/Tristan/Python_Stuff/SimpleML/data/books/"
books = os.listdir("D:/Tristan/Python_Stuff/SimpleML/data/books/")
for i in books:
    print (i)

datacandy = "D:/Tristan/Python_Stuff/SimpleML/data/candy/"
candy = os.listdir("D:/Tristan/Python_Stuff/SimpleML/data/candy/")
for i in candy:
    print(i) 


# databooks = "D:/Tristan/Python_Stuff/Youngwonks ML/cats/"
# books = os.listdir("D:/Tristan/Python_Stuff/Youngwonks ML/cats")
# for i in books:
#     print (i)

# datacandy = "D:/Tristan/Python_Stuff/Youngwonks ML/dogs/"
# candy = os.listdir("D:/Tristan/Python_Stuff/Youngwonks ML/dogs")
# for i in candy:
#     print(i)

data = []
labels = []

for i in books:
    
    image = cv2.imread(databooks+i,cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image,(64,48))
    data.append(image)
    labels.append(1)

for i in candy:
    
    image = cv2.imread(datacandy+i,cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image,(64,48))

    data.append(image)
    labels.append(0)

data = np.array(data)
labels = np.array(labels)

# Neural network

model = Sequential()
model.add(Dense(3072, input_dim=3072, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print (data.shape)
data = data.reshape(len(data),3072)
print (data.shape)

# Data Preprocessing
data = data / 255.0

# Shuffle data and labels
random_indices = np.random.permutation(len(data))
data = data[random_indices]
labels = labels[random_indices]



labels = to_categorical(labels)

x_train,x_test,y_train,y_test = train_test_split(data,labels,test_size=0.2)

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=64)

y_pred = model.predict(x_test)
#Converting predictions to label
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))

#Converting one hot encoded test label to label
test = list()
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))

from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
pred = np.argmax(y_pred, axis=1)
true_labels = np.argmax(y_test, axis=1)
accuracy = accuracy_score(pred, true_labels)
print('Accuracy is:', accuracy * 100)

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()