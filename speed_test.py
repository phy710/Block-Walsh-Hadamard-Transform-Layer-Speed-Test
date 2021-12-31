# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 03:57:16 2021

@author: Zephyr
"""

import numpy as np
import tensorflow as tf
from time import time

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model_fhwt = tf.keras.models.load_model('./fwht/saved_model/')
model_conv = tf.keras.models.load_model('./conv/saved_model/')
model_sq = tf.keras.models.load_model('./sq/saved_model/')

lite_model_fhwt = tf.lite.Interpreter(model_path="./fwht/lite_model.tflite")
lite_model_conv = tf.lite.Interpreter(model_path="./conv/lite_model.tflite")
lite_model_sq = tf.lite.Interpreter(model_path="./sq/lite_model.tflite")

input_shape = model_fhwt.input.shape.as_list()
input_size = input_shape[-2]
num_features = input_shape[-1]
num = 10

while True:
    x = np.random.rand(num, input_size, input_size, num_features).astype(np.float32)
    yy = np.zeros((num, input_size, input_size, num_features)).astype(np.float32)
    
    start = time()
    y = model_conv.predict(x)
    end = time()
    time1 = end-start
    
    start = time()
    for i in range(num):
        lite_model_conv.allocate_tensors()
        lite_model_conv.set_tensor(lite_model_conv.get_input_details()[0]['index'], x[i:i+1, :, :])
        lite_model_conv.invoke()
        yy[i:i+1, :, :] = lite_model_conv.get_tensor(lite_model_conv.get_output_details()[0]['index'])
    end = time()
    time2 = end-start
    
    print('3x3 Conv')
    print('PB took', time1, 'S;')
    print('TFLite took', time2, 'S.')
    
    
    start = time()
    y = model_fhwt.predict(x)
    end = time()
    time1 = end-start
    
    start = time()
    for i in range(num):
        lite_model_fhwt.allocate_tensors()
        lite_model_fhwt.set_tensor(lite_model_fhwt.get_input_details()[0]['index'], x[i:i+1, :, :])
        lite_model_fhwt.invoke()
        yy[i:i+1, :, :] = lite_model_fhwt.get_tensor(lite_model_fhwt.get_output_details()[0]['index'])
    end = time()
    time2 = end-start
    
    print('fwht')
    print('PB took', time1, 'S;')
    print('TFLite took', time2, 'S.')
    
    start = time()
    y = model_sq.predict(x)
    end = time()
    time1 = end-start
    
    start = time()
    for i in range(num):
        lite_model_sq.allocate_tensors()
        lite_model_sq.set_tensor(lite_model_sq.get_input_details()[0]['index'], x[i:i+1, :, :])
        lite_model_sq.invoke()
        yy[i:i+1, :, :] = lite_model_sq.get_tensor(lite_model_sq.get_output_details()[0]['index'])
    end = time()
    time2 = end-start
    
    print('sq')
    print('PB took', time1, 'S;')
    print('TFLite took', time2, 'S.')
    
