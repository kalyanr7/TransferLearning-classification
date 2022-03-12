#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Flatten, Dense
from tensorflow.keras.applications.vgg16 import VGG16


# In[3]:


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
# "C:\Users\91630\Pictures\Rock-Paper-Scissors"


# In[5]:


import matplotlib.pyplot as plt


# In[6]:


train_data.class_indices


# In[7]:


test_data.class_indices


# In[10]:


# for each in train_data:
#     for i in range(10):
#         fig = plt.figure(figsize=(10,10))
#         fig.add_subplot(5,2,i+1)
#         plt.axis('off')
#         plt.imshow(each[0][i])
#     break    


# In[11]:


# Building the model
model = tf.keras.Sequential()
model.add(VGG16(include_top=False, weights='imagenet', pooling='avg'))


# In[12]:


for layer in model.layers:
    layer.trainable = False


# In[13]:


model.summary()


# In[14]:


model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(3, activation='softmax'))


# In[15]:


model.summary()


# In[16]:


model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])


# In[17]:


model.fit(train_data, validation_data=test_data, epochs=10)


# In[31]:


import os
import numpy as np

path = 'C:/Users/91630/Pictures/Rock-Paper-Scissors/validation/'
image_paths = []

for filename in os.listdir(path):
    image_paths.append(path + filename)

for (i,image_path) in enumerate(image_paths):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224,224))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = image_array/255
    image_array_n = tf.expand_dims(image_array, 0)
 
    prediction = model.predict(image_array_n)
    prediction = prediction[0]
#     print(prediction)
    pred = max(prediction)
#     print(pred)
    pred_ind = np.where(prediction == pred)[0]
#     print(pred_ind)
    if pred_ind == 0:
        p = 'paper'
    elif pred_ind == 1:
        p = 'rock'
    elif pred_ind == 2:
        p = 'scissors'
    else: 
        p = 'no prediction'
        
    plt.figure()
    plt.title('predicted: '+ p)
    plt.imshow(image_array)
        
       


# In[ ]:


# import os
# import numpy as np

# path = 'C:/Users/91630/Pictures/Rock-Paper-Scissors/validation/'
# image_paths = []

# for filename in os.listdir(path):
#     image_paths.append(path + filename)

# for (i,image_path) in enumerate(image_paths):
#     image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224,224))
#     image_array = tf.keras.preprocessing.image.img_to_array(image)
#     image_array = image_array/255
#     img_vis = image_array
#     image_array = tf.expand_dims(image_array, 0)
 
#     prediction = model.predict(image_array)

#     pred = prediction.max(1)

#     pred_ind = np.where(prediction == pred)
    
#     pred_ind = int(pred_ind[1][0])
#     if pred_ind == 0:
#         p = 'paper'
#     elif pred_ind == 1:
#         p = 'rock'
#     elif pred_ind == 2:
#         p = 'scissors'
        
#     fig = plt.figure(figsize=(33,33))
#     fig.add_subplot(6,6,i+1)
#     plt.title('predicted: '+ p)
#     plt.imshow(img_vis)

