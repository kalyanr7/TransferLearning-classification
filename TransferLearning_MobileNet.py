#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Loading and preprocessing data
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True
)
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)


train_data = train_gen.flow_from_directory('./Rock-Paper-Scissors/train',target_size=(224, 224),
        batch_size=32)
test_data = test_gen.flow_from_directory('./Rock-Paper-Scissors/test',target_size=(224, 224),
        batch_size=32)


# In[12]:


train_data.class_indices


# In[16]:


# Building the model
model = tf.keras.Sequential()
model.add(tf.keras.applications.mobilenet.MobileNet(input_shape=(224,224,3), include_top=False, pooling='avg'))
for layer in model.layers:
    layer.trainable = False
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dense(3, activation='softmax'))          


# In[18]:


model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])


# In[19]:


model.fit(x=train_data, validation_data=test_data, batch_size=32, epochs=10)


# ## Predicting on 3 single images

# In[57]:


# Loading new images to predict
# predicting paper sign images

for i in range(3):
    img = tf.keras.utils.load_img('C:/Users/91630/Pictures/Rock-Paper-Scissors/validation/paper'+ str(i+1) + '.png', target_size=(224,224))
    img = tf.keras.utils.img_to_array(img)
    img = img/255 
    img_n = tf.expand_dims(img, axis=0)
    prediction = model.predict(img_n)
    prediction = prediction[0]
    p = max(prediction)
    index = np.where(prediction == p)[0]
    predicted = ''
    if index == 0:
        predicted = 'paper'
    elif index == 1:
        predicted = 'rock'
    elif index == 2:
        predicted = 'scissor'
    else:
        predicted = 'no prediction'
    
    plt.figure()
    plt.imshow(img)
    plt.title('Predicted sign : '+ predicted)


# ## Predicting on entire folder

# In[58]:


import os
path = 'C:/Users/91630/Pictures/Rock-Paper-Scissors/validation/'
image_paths = []

for filename in os.listdir(path):
    image_paths.append(path + filename)
    
for image_path in image_paths:
    img = tf.keras.utils.load_img(image_path, target_size=(224,224)) 
    img = tf.keras.utils.img_to_array(img)
    img = img/255 
    img_n = tf.expand_dims(img, axis=0)
    prediction = model.predict(img_n)
    prediction = prediction[0]
    p = max(prediction)
    index = np.where(prediction == p)[0]
    predicted = ''
    if index == 0:
        predicted = 'paper'
    elif index == 1:
        predicted = 'rock'
    elif index == 2:
        predicted = 'scissor'
    else:
        predicted = 'no prediction'
    
    plt.figure()
    plt.imshow(img)
    plt.title('Predicted sign : '+ predicted)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




