#!/usr/bin/env python
# coding: utf-8

# In[68]:


import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


# # Attempting to predict the median price of homes in a given Boston suburb
# # in the mid 1970s, given data points about the suburb at the time

# In[69]:


#Loading Boston Housing dataset
from keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()


# In[70]:


print(train_data.shape)
print(test_data.shape)


# In[71]:


train_data[1]


# In[47]:


len(train_targets)


# # Preparing the data

# In[39]:


#Normalizing the data via feature-wise normalization
mean = train_data.mean(axis = 0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std


# # Building your network

# In[40]:


#Model definition
from keras import models
from keras import layers

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape = (train_data.shape[1],)))
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model
#final layer has no activation which is typical for a scalar regression (a regression where you're trying to
# predict a single continuous value). If you point sigmoid as the activator it would try to predict values between
# 0 and 1
# Loss function is mse which is the widely used loss function in regression problems


# # Validating your approach using K-fold validation

# In[41]:


k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []


for i in range(k):
    print('proccessing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples] 
    # line above prepares the validation data: data from partition #k
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    
    #Prepares the training data " data from all other proportions"
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
        train_data[(i + 1) * num_val_samples:]], axis = 0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
        train_targets[(i + 1) * num_val_samples:]], axis = 0)
    
    model = build_model() #builds keras model(already compiled)
    model.fit(partial_train_data, partial_train_targets, epochs = num_epochs, batch_size = 1,  verbose = 0)
    # line above trains the model (in silence mode, verbose = 0)
    val_mse, val_mae, = model.evaluate(val_data, val_targets, verbose = 0) #evaluates the model on validation data
    all_scores.append(val_mae)
    print(all_scores)


# In[42]:


all_scores


# In[49]:


np.mean(all_scores)


# # Based on the results of the model, the predictions are off by an average of 2540 dollars

# # Hyperparameter Tuning

# In[50]:


num_epochs = 500
all_mae_histories = []


for i in range(k):
    print('proccessing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples] 
    # line above prepares the validation data: data from partition #k
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    
    #Prepares the training data " data from all other proportions"
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
        train_data[(i + 1) * num_val_samples:]], axis = 0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
        train_targets[(i + 1) * num_val_samples:]], axis = 0)
    
    model = build_model() #builds keras model(already compiled)
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs = num_epochs, batch_size = 1,  verbose = 0)
    # line above trains the model (in silence mode, verbose = 0)
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)


# In[52]:


average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]


# In[53]:


plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


# # Plotting validation scores, excluding the first 10 data points

# In[64]:


def smooth_curve(points, factor = 0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
             smoothed_points.append(point)
    return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


# ## according to this plot, validation MAE stops improving significantly after 80 epochs. Past that point is overfitting

# # Training the final model

# In[65]:


model = build_model()
model.fit(train_data, train_targets, epochs = 80, batch_size = 16, verbose = 0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)


# In[66]:


test_mae_score


# # off by $2,680 dollars

# In[ ]:




