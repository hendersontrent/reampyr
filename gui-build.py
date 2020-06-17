#--------------------------------
#
#--------------------------------

#--------------------------------
#
#--------------------------------

#%%
import tkinter as tk
from tkinter.ttk import * 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

#----------------BUILD BASIC GUI LAYOUT--------------

# Root window

root = tk.Tk()

# Set window parameters

root.title('ReamPyr - Machine Learning Software (v0.1)')
root.configure(background = "#B2D9EA")
root.geometry("700x600")

# Add application title

main_title = tk.Text(root, height = 20, width = 40,
                     font = ('Arial', 54, 'bold'),
                         foreground = 'white', background = "#B2D9EA",
                         highlightthickness = 0)
main_title.insert(tk.END,'ReamPyr')
main_title.pack(side = tk.LEFT)

# Run application

root.mainloop()
