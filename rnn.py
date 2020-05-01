#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 22:49:44 2020

@author: yash
"""

import matplotlib.pyplot as plt
import collections
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

model = tf.keras.Sequential()
model.add(layers.Embedding(input_dim=1000,output_dim=64))
model.add(layers.LSTM(128))
model.add(layers.Dense(10))
model.summary()

model = tf.keras.Sequential()
model.add(layers.Embedding(input_dim = 1000,output_dim=64))
model.add(layers.GRU(256,return_sequences=True))
model.add(layers.SimpleRNN(128))
model.add(layers.Dense(10))
model.summary()