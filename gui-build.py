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
from scipy.io import wavfile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns; sns.set()

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

# Read in raw DI and amp file and plot it

def audio_plotter():
    samplerate, data = wavfile.read("/Users/trenthenderson/Documents/Python/reampyr/data/Dry Guitar.wav")
    times = np.arange(len(data))/float(samplerate)
    
    # Plot it
    
    f, ax = plt.subplots(figsize=(7, 3))
    
    plt.fill_between(times, data[:,0], data[:,1], color = '#619196') 
    plt.xlim(times[0], times[-1])
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    return f

#%%

#----------------WRITE ML ALGORITHMS----------------

#--------------
# LSTM CNN
#--------------

# Scale data ready for algorithm

scaler = MinMaxScaler(feature_range=(0, 1))
DATA = scaler.fit_transform(DATA)

# Split into train and test data

train_size = int(len(DATA) * 0.80)
test_size = len(DATA) - train_size
train, test = DATA[0:train_size,:], DATA[train_size:len(DATA),:]
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
# data as a SUPERVISED LEARNING PROBLEM
# with inputs and outputs side-by-side

look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

#--------------
# FFT
#--------------



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
main_title.pack(side = tk.TOP)

# Add data visualisation

fig = audio_plotter()
canvas = FigureCanvasTkAgg(fig, master = root)
canvas.draw()
canvas.get_tk_widget().pack(side = tk.RIGHT, fill = tk.BOTH, expand = True)

# Exit button

def _quit():
    root.quit()
    root.destroy()

button = tk.Button(master = root, text = "Exit", command = _quit)
button.pack(side = tk.BOTTOM)

# Run application

root.mainloop()
