#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 16:34:44 2020

@author: kartikey
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflowrbm import BBRBM, GBRBM
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST/data',one_hot = True)
mnist_images = mnist.train.images

bbrbm = BBRBM(n_visible=784,n_hidden=64,learning_rate = 0.01,momentum = 0.95,use_tqdm = True)
errs = bbrbm.fit(mnist_images,n_epochs=30,batch_size=10)
plt.plot(errs)
plt.show()