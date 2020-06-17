#--------------------------------
#
#--------------------------------

#--------------------------------
#
#--------------------------------

#%%
import tkinter as tk
from tkinter import *
from tkinter.ttk import * 
from argparse import ArgumentParser
import numpy as np
import os
import soundfile as sf
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

#----------------DEFINE COLOUR PALETTE--------------------

two_palette = ['#619196', '#B2D9EA']
three_palette = ['#619196', '#B2D9EA', '#F4DCD6']

#%%

#----------------READ IN DATA--------------------

# Plotter for raw data

sns.lineplot(x = "x", y = "y",
             hue = "group", style = "event",
             data = data, palette = two_palette)

# Plotter for trained neural network data

sns.lineplot(x = "x", y = "y",
             hue = "group", style = "event",
             data = data, palette = three_palette)

# Read in raw DI and amp file

di_data, samplerate = sf.read("filehere.wav")
amp_data, samplerate = sf.read("filehere.wav")

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
main_title.grid(row = 1, column = 0)

# Add input panel label

inputs_title = tk.Label(root, text = "Inputs",
                        font = ('Arial', 42, 'bold'),
                        fg = '#619196', bg = "#B2D9EA",
                        highlightthickness = 0)
inputs_title.grid(row = 10, column = 0)

# Add inputs



# Add data visualisation



# Run application

root.mainloop()
