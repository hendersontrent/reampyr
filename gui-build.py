#--------------------------------------
# This script sets out to produce a GUI
# application that loads, visualises
# and transforms a raw DI guitar signal
# into that of a given amplifier using
# machine learning.
#--------------------------------------

#--------------------------------------
# Author: Trent Henderson, 17 June 2020
#--------------------------------------

#%%
import tkinter as tk
from tkinter import *
from tkinter.ttk import *
import numpy as np
import pandas as pd
import os
import math
from scipy.io import wavfile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns; sns.set(style = "darkgrid")

from subprocess import check_output
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#%%
#----------------READ IN DATA--------------------

# Read in raw DI and amp file

samplerate, di_data = wavfile.read("/Users/trenthenderson/Documents/Python/reampyr/data/ScathingRhythmDI.wav")
times_di = np.arange(len(di_data))/float(samplerate)

samplerate, amp_data = wavfile.read("/Users/trenthenderson/Documents/Python/reampyr/data/ScathingRhythmAmp.wav")
times_amp = np.arange(len(amp_data))/float(samplerate)

# Merge together

merged_data = np.concatenate((di_data, amp_data), axis = 1)

#%%
#----------------VISUALISATION-------------------

# Plot every 100th point to avoid matplotlib overflow errors

fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, figsize = (11,7))

ax1.fill_between(times_di[::100], di_data[::100,0], di_data[::100,1], color = '#57DBD8')
ax1.set_ylabel('Amplitude')
ax2.set_ylim(-40000,40000)
ax1.set_title('Raw Guitar DI')

ax2.fill_between(times_amp[::100], amp_data[::100,0], amp_data[::100,1], color = '#F84791')
ax2.set_title('Guitar Amplifier Head')
ax2.set_ylabel('Amplitude')
ax2.set_xlabel('Time (s)')
ax2.set_xlim(times_di[0], times_di[-1])
ax2.set_ylim(-40000,40000)

fig.savefig('/Users/trenthenderson/Documents/Python/reampyr/output/raw_signal.png', dpi = 1000)
fig.show()

#%%
#----------------WRITE LSTM ML ALGORITHMS-----------

# Scale data ready for algorithm

shorter_data = np.delete(merged_data,1,1)
shorter_data = np.delete(shorter_data,2,1)

scaler = MinMaxScaler(feature_range = (0, 1))
trans_data = scaler.fit_transform(shorter_data)

#%%
# Split into train and test data

train_size = int(len(trans_data) * 0.80)
test_size = len(trans_data) - train_size
train, test = trans_data[0:train_size,:], trans_data[train_size:len(trans_data),:]
print(len(train), len(test))

# Convert function to convert an array of values into a dataset matrix

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# Use function to reshape data into X=t and Y=t+1 - this frames 
# data as a SUPERVISED LEARNING PROBLEM with inputs and outputs side-by-side

look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

#%%
#-----------------MODEL FITTING---------------------

# Instantiate network

model = Sequential()

# Layer 1

model.add(LSTM(units = 32, input_shape = (1, look_back), activation = 'relu', 
               return_sequences = True))
model.add(Dropout(0.2))

# Layer 2

model.add(LSTM(units = 32, return_sequences = True, activation = 'relu'))
model.add(Dropout(0.2))

# Layer 3

model.add(LSTM(units = 32, return_sequences = True, activation = 'relu'))
model.add(Dropout(0.2))

# Layer 4

model.add(LSTM(units = 32, return_sequences = True, activation = 'relu'))
model.add(Dropout(0.2))

# Layer 5

model.add(LSTM(units = 32, return_sequences = True, activation = 'relu'))
model.add(Dropout(0.2))

# Final model components

model.add(Dense(units = 1, activation = 'relu')) # Predict single output value
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['acc'])

#%%
# Train the model

model.fit(trainX, trainY, epochs = 1000, batch_size = 512, verbose = 2)

#%%
# Visualise loss

history = model.fit(trainX, trainY, epochs = 1000, batch_size = 512, verbose = 2,
                    validation_data = (testX, testY))

#%%
plt.plot(history.history['loss'], label = "Train", color = '#57DBD8')
plt.plot(history.history['val_loss'], label = "Validation", color= '#F84791')
plt.title('Model train vs validation loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc = 'upper right')
plt.show()

#%%
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

#%%
# Reshape Y data

train_shape = trainY.shape
trainYr = trainY.reshape((train_shape[0]))

test_shape = testY.shape
testYr = testY.reshape((test_shape[0]))

# Invert predictions

trainPredict = scaler.inverse_transform(trainPredict)
trainYr = scaler.inverse_transform([trainYr])
testPredict = scaler.inverse_transform(testPredict)
testYr = scaler.inverse_transform([testYr])

# Calculate root mean squared error

trainScore = math.sqrt(mean_squared_error(trainYr[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testYr[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

