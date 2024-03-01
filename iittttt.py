#!/usr/bin/env python
# coding: utf-8

# In[1]:


# DIGIT RECOGNITION SYSTEM


# In[2]:


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


# In[3]:


# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[4]:


print(x_train.shape, y_train.shape)


# In[5]:


#Preprocess the data
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)


# In[6]:


# convert class vectors to binary class matrices
import tensorflow as tf
from keras import utils as np_utils 

num_classes=10
y_train =  tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


# In[7]:


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# In[8]:


#Create the model
batch_size = 128
num_classes = 10
epochs = 10


# In[9]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])


# In[10]:


#Train the model
hist = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
print("The model has successfully trained")


# In[11]:


model.save('mnist.h5')
print("Saving the model as mnist.h5")


# In[12]:


#Evaluate the model

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:


##from tensorflowTesting import testing
import tensorflow as tf
from keras.models import load_model
import cv2
import numpy as np
import os

from PIL import ImageTk, Image, ImageDraw
import PIL
import tkinter as tk
from tkinter import *

classes=[0,1,2,3,4,5,6,7,8,9]
width = 500
height = 500
center = height//2
white = (255, 255, 255)
green = (0,128,0)

def paint(event):
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    cv.create_oval(x1, y1, x2, y2, fill="black",width=40)
    draw.line([x1, y1, x2, y2],fill="black",width=40)
def model():
    filename = "image.png"
    image1.save(filename)
    pred=testing()
    print('argmax',np.argmax(pred[0]),'\n',
          pred[0][np.argmax(pred[0])],'\n',classes[np.argmax(pred[0])])
    txt.insert(tk.INSERT,"{}\nAccuracy: {}%".format(classes[np.argmax(pred[0])],round(pred[0][np.argmax(pred[0])]*100,3)))
    
def clear():
    cv.delete('all')
    draw.rectangle((0, 0, 500, 500), fill=(255, 255, 255, 0))
    txt.delete('1.0', END)

root = Tk()
##root.geometry('1000x500') 

root.resizable(0,0)
cv = Canvas(root, width=width, height=height, bg='white')
cv.pack()

# PIL create an empty image and draw object to draw on
# memory only, not visible
image1 = PIL.Image.new("RGB", (width, height), white)
draw = ImageDraw.Draw(image1)

txt=tk.Text(root,bd=3,exportselection=0,bg='WHITE',font='Helvetica',
            padx=10,pady=10,height=5,width=20)

cv.pack(expand=YES, fill=BOTH)
cv.bind("<B1-Motion>", paint)

##button=Button(text="save",command=save)
btnModel=Button(text="Predict",command=model)
btnClear=Button(text="clear",command=clear)
##button.pack()
btnModel.pack()
btnClear.pack()
txt.pack()
root.title('digit recognizer---- PRIYANKA/VAIDEHI')
root.mainloop()


# In[1]:


import tensorflow as tf
from keras.models import load_model
import cv2
import numpy as np

classes=[0,1,2,3,4,5,6,7,8,9]

model=tf.keras.models.load_model('digit_recog_cnn.h5')
def testing():
    img=cv2.imread('image.png',0)
    img=cv2.bitwise_not(img)
##    cv2.imshow('img',img)
    img=cv2.resize(img,(28,28))
    img=img.reshape(1,28,28,1)
    img=img.astype('float32')
    img=img/255.0

    pred=model.predict(img)
    return pred


# In[ ]:




