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

# Create pandas version for easy plotting

merged_data_pd = merged_data

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

scaler = MinMaxScaler(feature_range = (0, 1))
merged_data = scaler.fit_transform(merged_data)

# Split into train and test data

train_size = int(len(merged_data) * 0.80)
test_size = len(merged_data) - train_size
train, test = merged_data[0:train_size,:], merged_data[train_size:len(merged_data),:]
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

# Create and fit the LSTM network

model = Sequential()
model.add(LSTM(15, input_shape = (1, look_back), dropout = 0.5, activation = 'sigmoid'))
model.add(Dense(2, activation = 'sigmoid'))
opt = Adam(learning_rate = 0.01)
model.compile(loss = 'mean_squared_error', optimizer = opt, metrics = ['acc'])
early_stop = EarlyStopping(monitor = 'loss', patience = 10, verbose = 1) # Stops after no improvement across 10 epochs
model.fit(trainX, trainY, epochs = 1000, batch_size = 512, verbose = 2, callbacks = [early_stop])

#%%
# Visualise loss

history = model.fit(trainX, trainY, epochs = 100, batch_size = 512, verbose = 2, callbacks = [early_stop],
                    validation_data = (testX, testY))
plt.plot(history.history['loss'], label = "Train", color = '#B2D9EA')
plt.plot(history.history['val_loss'], label = "Validation", color= '#F4DCD6')
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

#%%

#----------------BUILD BASIC GUI LAYOUT--------------

# Root window

root = tk.Tk()

# Set window parameters

root.title('ReamPyr - Machine Learning Software (v0.1)')
root.configure(background = "#B2D9EA")
root.geometry("900x500")

# Add application title

main_title = tk.Label(root, text = "ReamPyr" ,
                      font = ('Arial', 54, 'bold'),
                      fg = 'white', bg = "#B2D9EA",
                      highlightthickness = 0)
main_title.pack(side = tk.TOP, fill = tk.X, expand = False)

# Add data visualisation

#fig = audio_plotter()
#canvas = FigureCanvasTkAgg(fig, master = root)
#canvas.get_tk_widget().configure(background = '#B2D9EA', highlightcolor = '#B2D9EA')
#canvas.draw()
#canvas.get_tk_widget().pack(side = tk.RIGHT, expand = False)

# Exit button

def _quit():
    root.quit()
    root.destroy()

button = tk.Button(master = root, text = "Exit", command = _quit)
button.pack(side = tk.BOTTOM, expand = False)

# Run application

root.mainloop()
